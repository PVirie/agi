import os
import sys
import logging
import random
import argparse
import numpy as np
import shutil
import asyncio
import time
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

from utilities.package_install import install

install("minigrid")
install("gymnasium[other]")

from utilities import format_float
from utilities.minigrid.environments import Multi_Environment, BonusType
from utilities.minigrid.babyai_language import shorten_mission
from utilities.tokenizer import Text_Tokenizer
from utilities.episode_recorder import Episode_Recorder, CUMSUM, prefixed_headers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import model_82
from implementations.networks.torch.policy.babyai_82_core import Policy_Core
from implementations.learning_algorithms.torch.ppo_graph_split import PPO
from implementations.collectors.states import State_Sequence as Collector
from implementations.memories.graph_memory import NP_Graph_Memory as Graph_Memory

torch.autograd.set_detect_anomaly(False)

# logged every 2 rollouts: PPO.last_stats (aux_loss = the prediction loss, only when a prediction is enabled) and,
# averaged over the steps since the last log where they count:
#   prediction_reward: the intrinsic reward (whitened scores by default, so ~0 on average)
#   obs_score, reward_score: the raw observation / reward prediction scores in [0, 1]
#   reward_event_score: the reward score on steps that carried a reward, the one that shows whether rewards are anticipated
PREDICTION_KEYS = ["prediction_reward", "obs_score", "reward_score", "reward_event_score"]
DIAGNOSTIC_KEYS = PREDICTION_KEYS + ["entropy", "approx_kl", "pg_loss", "v_loss", "aux_loss"]

# per-env episode statistics, in the order run() fills stat_row;
# env_steps receives each episode's length and is summed so the count continues across resumed runs
STAT_COLUMNS = [("env_steps", CUMSUM), "return", "length", "success", "excess_degree", "graph_nodes"]
DIAGNOSTIC_COLUMNS = [("env_steps", CUMSUM)] + DIAGNOSTIC_KEYS

async def run(env, agent, rollout_length=16, verbose=False, env_seed=None):

    observations, info = env.reset(seed=env_seed)
    rewards = [np.zeros(env.reward_size, dtype=np.float32) for _ in observations]
    last_idle = [False for _ in observations]
    last_done = [False for _ in observations]
    last_truncated = [False for _ in observations]
    last_reset = [False for _ in observations]

    total_returns = [0 for _ in observations]
    # per mission sub-goal count: [successful, total] finished episodes, pooled over envs since the last log
    sub_goal_outcomes = {}
    session_return_update_alpha = 0.9
    start_time = time.perf_counter()
    steps = 0
    last_diagnostic_env_steps = 0
    prediction_sums = {k: [0.0, 0] for k in PREDICTION_KEYS}  # key -> [sum, count]
    while True:
        elapsed_time = time.perf_counter() - start_time
        should_stop = elapsed_time > max_running_time  # run for the specified max time

        actions, metrics = agent.choose_action(
            last_idles=last_idle,
            last_dones=last_done,
            last_truncates=last_truncated,
            last_resets=last_reset,
            latest_frames=observations,
            rewards=rewards,
            next_available_actions=env.get_available_actions(),
            force_train=steps % rollout_length == 0 or should_stop,
        )
        actions = [int(a[0].item()) if a is not None else None for a in actions]
        # the scores judge the transition into this call: its reward is `rewards`, and an episode end leaves the observation unscored
        reward_events = np.array([abs(r[0]) > 0.01 for r in rewards])
        for key, values, counted in [
            ("prediction_reward", metrics[2], None),
            ("obs_score", metrics[3], ~np.array(last_done)),
            ("reward_score", metrics[4], None),
            ("reward_event_score", metrics[4], reward_events),
        ]:
            values = values if counted is None else values[counted]
            prediction_sums[key][0] += float(np.sum(values))
            prediction_sums[key][1] += len(values)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        last_idle = [a is None for a in actions]
        last_done = [terminations[i] or truncations[i] for i in range(len(observations))]
        last_truncated = [truncations[i] for i in range(len(observations))]
        last_reset = [terminations[i] or truncations[i] for i in range(len(observations))]

        stat_row = []
        for i in range(len(observations)):
            if terminations[i] or truncations[i]:
                total_score = infos[i]["episode"]["r"]
                outcome = sub_goal_outcomes.setdefault(infos[i]["episode"]["m"], [0, 0])
                outcome[0] += int(infos[i]["episode"]["s"])
                outcome[1] += 1
                total_returns[i] = (
                    session_return_update_alpha * total_returns[i]
                    + (1 - session_return_update_alpha) * total_score
                )
                stat_row.extend([
                    infos[i]["episode"]["e"],
                    infos[i]["episode"]["r"], 
                    infos[i]["episode"]["l"], 
                    infos[i]["episode"]["s"],
                    metrics[0][i],
                    metrics[1][i],
                ])
            else:
                stat_row.extend([None] * len(STAT_COLUMNS))
        stat_recorder.record(stat_row)

        steps += 1
        # log extrinsic rewards only
        if any([r[0] != 0 for r in rewards]) and verbose:
            logging.info(f"{steps}| Rewards: {', '.join([format_float(r[0]) for r in rewards])}")

        # constant LR on purpose: a wall-clock anneal would give the arms different schedules
        # at the same environment step, which is the axis they are compared on

        if steps % (rollout_length * 2) == 0 or should_stop:
            logging.info(f"{steps}| Returns: {', '.join([format_float(s) for s in total_returns])}")
            if sub_goal_outcomes:
                logging.info(f"{steps}| Success by sub-goals: " + ", ".join(f"{k}: {s}/{n}" for k, (s, n) in sorted(sub_goal_outcomes.items())))
                sub_goal_outcomes = {}
            logging.info(f"{steps}| Selected actions: {actions}")

            diagnostics = {
                **{k: s / n for k, (s, n) in prediction_sums.items() if n > 0},
                **ppo_learner.last_stats,
            }
            prediction_sums = {k: [0.0, 0] for k in PREDICTION_KEYS}
            logging.info(f"{steps}| Diagnostics: " + ", ".join(f"{k}={diagnostics[k]:.4g}" for k in DIAGNOSTIC_KEYS if k in diagnostics))
            # env_steps is a cumsum column: record this interval's steps so the count continues across resumed runs
            diagnostic_recorder.record([env.total_env_steps() - last_diagnostic_env_steps] + [diagnostics.get(k) for k in DIAGNOSTIC_KEYS])
            last_diagnostic_env_steps = env.total_env_steps()
            diagnostic_recorder.write()

            # save 
            policy_core.save()
            ppo_learner.save()
            tokenizer.save(f"{experiment_path}/parameters")

        if steps % (rollout_length * 10) == 0:
            # compute estimated time left
            logging.info(f"Completed {steps} steps, {env.total_env_steps()} environment steps.")
            logging.info(f"Current elapsed time: {elapsed_time:.2f} seconds.")
            logging.info(f"Expected time left: {max_running_time - elapsed_time:.2f} seconds.")

        if steps % 1000 == 0:
            stat_recorder.write()

        if should_stop:
            # arms differ in throughput, so this total is the truncation point for comparison
            logging.info(f"Max running time reached after {env.total_env_steps()} environment steps, stopping the experiment.")
            break

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",                  "-r",   action="store_true")
    parser.add_argument("--hours",                  "-hr",  type=float, default=0.05, help="Number of hours to train the agent. Fractional hours allowed.")
    parser.add_argument("--scale",                  "-s",   type=str, default="medium", choices=["small", "medium", "large"], help="The scale of the neural network. Default is 'medium'.")
    parser.add_argument("--st-train",               "-stt",   action="store_true", help="Use spatio-temporal training for the agent. Default is False.")
    parser.add_argument("--silent",                 "-silent", action="store_true", help="Disable reward logging for cleaner output.")
    parser.add_argument("--cell",                   "-cell", type=str, default="gru", choices=["lstm", "gru", "residual"], help="Recurrent cell of the policy memory. Default is 'gru'.")
    parser.add_argument("--ent-coef",               "-ent", type=float, default=0.01, help="PPO entropy coefficient, over both the ext and the internal action. Default is 0.01.")
    parser.add_argument("--no-predict-obs",         "-npo", action="store_true", help="Disable the next-observation prediction and its intrinsic reward. Default is enabled.")
    parser.add_argument("--no-predict-reward",      "-npr", action="store_true", help="Disable the step-reward prediction and its intrinsic reward. Default is enabled.")
    parser.add_argument("--aux-coef",               "-aux", type=float, default=1.0, help="Coefficient of the prediction loss that trains the predictors. Default is 1.0.")
    parser.add_argument("--gamma-int",              "-gint", type=float, default=0.95, help="Discount of the intrinsic stream; the prediction reward is dense (every step), so a shorter horizon keeps its return scale near the extrinsic one. Default is 0.95.")
    args = parser.parse_args()


    # print summary of arguments that are not default
    logging.info("With arguments:")
    for arg in vars(args):
        value = getattr(args, arg)
        default = parser.get_default(arg)
        if value != default:
            logging.info(f"  --{arg}: {value} (default: {default})")

    max_running_time = int(args.hours * 3600.0)  # in seconds
    hours = max_running_time // 3600
    minutes = (max_running_time % 3600) // 60
    seconds = max_running_time % 60
    logging.info(f"The experiment will be run for {hours} hours, {minutes} minutes, and {seconds} seconds.")

    predict_observation = not args.no_predict_obs
    predict_reward = not args.no_predict_reward
    if not (predict_observation or predict_reward):
        logging.warning("Both predictions are disabled: the intrinsic reward is zero, so the internal actions get no learning signal.")

    # For reproducibility (https://docs.pytorch.org/docs/stable/notes/randomness.html)
    seed = 20261124
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    # the predictors change the checkpoint layout, so each prediction setting gets its own experiment
    experiment_path = f"{APP_ROOT}/experiments/babyai_82_size_{args.scale}_cell_{args.cell}"
    if args.st_train:
        experiment_path += "_stmean"
    if not predict_observation:
        experiment_path += "_noobs"
    if not predict_reward:
        experiment_path += "_norwd"

    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    vocab_size = 256
    tokenizer = Text_Tokenizer(max_vocab_size=vocab_size)
    tokenizer.load(f"{experiment_path}/parameters")

    game_ids = ["BabyAI-MiniBossLevel-v0"] * 512
    # compact missions: longest possible BossLevel mission is 27 tokens, total vocab ~58 words
    env = Multi_Environment(
        game_ids=game_ids,
        tokenizer=tokenizer,
        mission_max_len=32,
        full_mdp=False,  # the observation predictor's classes (obs_classes) assume the partial 7x7 view
        include_inventory=False,
        bonus_type=BonusType.NONE,
        # the env gives the extrinsic reward only; model_82 adds the prediction reward as the intrinsic stream
        has_intrinsic_reward=False,
        mission_translator=shorten_mission,
    )

    mission_size = env.mission_max_len
    inventory_size = 3 # inventory include 1 slot for current direction and 2 slots for items
    
    stat_recorder = Episode_Recorder(f"{experiment_path}/statistics", headers=prefixed_headers(game_ids, STAT_COLUMNS))
    diagnostic_recorder = Episode_Recorder(f"{experiment_path}/diagnostics", headers=DIAGNOSTIC_COLUMNS)
    
    if args.scale == "small":
        hidden_size = 128
        embedding_dim = 32
        node_dim = 32
        C = 3
        layers = [1, 1, 1]
        memory_dim = hidden_size
        cnn_channels = 64
        cnn_blocks = 2
        minibatch_size = 8
        rollout_length = 84
    elif args.scale == "medium":
        hidden_size = 256
        embedding_dim = 32
        node_dim = 32
        C = 7
        layers = [1, 1, 1, 1]
        memory_dim = hidden_size
        cnn_channels = 128
        cnn_blocks = 2
        minibatch_size = 12
        rollout_length = 256
    else:
        hidden_size = 256
        embedding_dim = 32
        node_dim = 32
        C = 7
        layers = [1, 1, 1, 1]
        memory_dim = hidden_size
        cnn_channels = 128
        cnn_blocks = 2
        minibatch_size = 24
        rollout_length = 256

    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)
    policy_core = Policy_Core(
        int_action_size=len(model_82.INT_FLAG_OPS), ext_action_size=7 + 1,  # last ext slot = idle (no env step)
        node_dim=node_dim,
        internal_state_size=1 + C,
        goal_size=mission_size, inventory_size=inventory_size,
        width=7, height=7, channel=3,
        dict_size=vocab_size, embedding_dim=embedding_dim, pad_token_id=0,
        hidden_size=hidden_size, layers=layers, 
        memory_dim=memory_dim,
        cnn_channels=cnn_channels, cnn_blocks=cnn_blocks,
        cell=args.cell,
        # edges and write values count only on the memory ops that use them
        component_flags=model_82.COMPONENT_FLAGS,
        predict_observation=predict_observation,
        predict_reward=predict_reward,
        device=device, persistence_path=parameters_path
    ).to(device)
    ppo_learner = PPO(
        # the ext action learns from the extrinsic return, the internal actions from the intrinsic (prediction) return
        policy_model=policy_core,
        device=device, persistence_path=parameters_path,
        lr=1e-4, minibatch_size=minibatch_size,
        gamma=0.99,
        gamma_intrinsic=args.gamma_int,
        gae_lambda=0.95,
        # each stream's advantage is normalised on its own and drives disjoint heads, so equal weights
        ext_coef=1.0,
        int_coef=1.0,
        ent_coef=args.ent_coef,
        # the prediction loss (logged as aux_loss); None without predictions, as the policy then has no aux loss
        aux_coef=args.aux_coef if (predict_observation or predict_reward) else None,
        use_graph=args.st_train,
        # the recurrence and the graph memory restart each episode, so internal actions cannot earn the next one's predictions
        episodic_intrinsic=True,
    )
    memory = Graph_Memory(
        num_batches=len(game_ids),
        num_nodes=4096,
        max_edges_per_node=C,
        node_dim=node_dim,
        # node values are the policy's sampled write values in [0, 1]^node_dim
        start_node_value=0.0,
        empty_value=0.0,
    )
    agent = model_82.Model(
        policy_model=policy_core,
        trainer=ppo_learner,
        context_collector=Collector(max_history=0),
        action_collector=Collector(max_history=0),
        valid_action_collector=Collector(max_history=0),
        graph_memory=memory,
        has_intrinsic_reward=env.has_intrinsic_reward
    )

    asyncio.run(run(env, agent, rollout_length, verbose=not args.silent, env_seed=seed))
