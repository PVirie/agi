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

from utilities import scatter, format_float
from utilities.minigrid.environments import Multi_Environment, BonusType
from utilities.minigrid.babyai_language import shorten_mission
from utilities.tokenizer import Text_Tokenizer
from utilities.episode_recorder import Episode_Recorder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import model_76
from implementations.networks.torch.policy.babyai_cont import Projector
from implementations.networks.torch.policy.babyai_cont import Policy_Core
from implementations.learning_algorithms.torch.ppo_graph import PPO
from implementations.collectors.states import State_Sequence as Collector
from implementations.memories.graph_memory import NP_Graph_Memory as Graph_Memory

torch.autograd.set_detect_anomaly(False)

# logged every 2 rollouts, see PPO.last_stats; aux_loss is only present when the learner has aux_coef set
DIAGNOSTIC_KEYS = ["entropy", "approx_kl", "pg_loss", "v_loss", "aux_loss"]

async def run(env, agent, rollout_length=16, verbose=False, env_seed=None):

    observations, info = env.reset(seed=env_seed)
    rewards = [np.zeros(env.reward_size, dtype=np.float32) for _ in observations]
    last_idle = [False for _ in observations]
    last_done = [False for _ in observations]
    last_truncated = [False for _ in observations]
    last_reset = [False for _ in observations]

    total_returns = [0 for _ in observations]
    session_return_update_alpha = 0.9
    start_time = time.perf_counter()
    steps = 0
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

        observations, rewards, terminations, truncations, infos = env.step(actions)

        last_idle = [a is None for a in actions]
        last_done = [terminations[i] or truncations[i] for i in range(len(observations))]
        last_truncated = [truncations[i] for i in range(len(observations))]
        last_reset = [terminations[i] or truncations[i] for i in range(len(observations))]

        stat_row = []
        for i in range(len(observations)):
            if terminations[i] or truncations[i]:
                total_score = infos[i]["episode"]["r"]
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
                stat_row.extend([None, None, None, None, None, None])
        stat_recorder.record(stat_row)

        steps += 1
        # log extrinsic rewards only; with the position bonus the training reward is nonzero on almost every step
        if any([r[0] != 0 for r in rewards]) and verbose:
            logging.info(f"{steps}| Rewards: {', '.join([format_float(r[0]) for r in rewards])}")

        # constant LR on purpose: a wall-clock anneal would give the arms different schedules
        # at the same environment step, which is the axis they are compared on

        if steps % (rollout_length * 2) == 0 or should_stop:
            logging.info(f"{steps}| Returns: {', '.join([format_float(s) for s in total_returns])}")
            logging.info(f"{steps}| Selected actions: {actions}")

            diagnostics = ppo_learner.last_stats
            logging.info(f"{steps}| Diagnostics: " + ", ".join(f"{k}={diagnostics[k]:.4g}" for k in DIAGNOSTIC_KEYS if k in diagnostics))
            diagnostic_recorder.record([env.total_env_steps()] + [diagnostics.get(k) for k in DIAGNOSTIC_KEYS])
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
    parser.add_argument("--max-thought-steps",      "-mts", type=int, default=2, help="Maximum number of thought steps the agent can take before being forced to act externally.")
    parser.add_argument("--scheme",                 "-sch", type=str, default="full", help="The scheme to use for the agent's decision making. Default is 'full'.")
    parser.add_argument("--st-train",               "-stt",   action="store_true", help="Use spatio-temporal training for the agent. Default is False.")
    parser.add_argument("--silent",                 "-silent", action="store_true", help="Disable reward logging for cleaner output.")
    parser.add_argument("--ent-coef",               "-ent", type=float, default=0.01, help="PPO entropy coefficient. Default is 0.01.")
    parser.add_argument("--aux-coef",               "-aux", type=float, default=None, help="Coefficient of the graph-transition auxiliary loss (U-Net: obs_t + delta graph -> obs at the next non-thought step). Default is off.")
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

    # For reproducibility (https://docs.pytorch.org/docs/stable/notes/randomness.html)
    seed = 20260718
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    experiment_path = f"{APP_ROOT}/experiments/babyai_76_cont_size_{args.scale}_scheme_{args.scheme}_mts_{args.max_thought_steps}"
    if args.st_train:
        experiment_path += "_stmean"
    if args.aux_coef is not None:
        # the aux decoder adds parameters, so its checkpoints are not interchangeable with the plain arm
        experiment_path += "_aux"

    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    vocab_size = 256
    tokenizer = Text_Tokenizer(max_vocab_size=vocab_size)
    tokenizer.load(f"{experiment_path}/parameters")

    # game_ids = ["BabyAI-MiniBossLevel-v0"] * 256 + ["BabyAI-GoToLocalS8N7-v0"] * 128 + ["BabyAI-UnlockPickupDist-v0"] * 128
    # game_ids = ["BabyAI-MiniBossLevel-v0"] * 512
    game_ids = ["BabyAI-MiniBossLevel-v0"] * 512
    # compact missions: longest possible BossLevel mission is 27 tokens, total vocab ~58 words
    env = Multi_Environment(
        game_ids=game_ids,
        tokenizer=tokenizer,
        mission_max_len=32,
        full_mdp=False,
        include_inventory=False,
        bonus_type=BonusType.NONE,
        # extrinsic only: single-stream ppo_graph with a single value head (policy value_size=1)
        has_intrinsic_reward=False,
        mission_translator=shorten_mission
    )

    mission_size = env.mission_max_len
    inventory_size = 3 # inventory include 1 slot for current direction and 2 slots for items
    
    stat_recorder = Episode_Recorder(f"{experiment_path}/statistics", headers=[f"{gid}/{stat}" for gid in game_ids for stat in ["env_steps", "return", "length", "success", "excess_degree", "graph_nodes"]])
    diagnostic_recorder = Episode_Recorder(f"{experiment_path}/diagnostics", headers=["env_steps"] + DIAGNOSTIC_KEYS)
    
    if args.scale == "small":
        hidden_size = 256
        embedding_dim = 32
        node_dim = 8
        C = 4
        layers = [1, 1, 1]
        cnn_channels = 64
        cnn_blocks = 2
        minibatch_size = 4
        rollout_length = 512
    elif args.scale == "medium":
        # 19GB VRAM
        hidden_size = 256
        embedding_dim = 32
        node_dim = 8
        C = 8
        layers = [1, 1, 1, 1]
        cnn_channels = 128
        cnn_blocks = 2
        minibatch_size = 8
        rollout_length = 512
    else:
        # large 25GB VRAM
        hidden_size = 384
        embedding_dim = 32
        node_dim = 8
        C = 15
        layers = [1, 1, 1, 1]
        cnn_channels = 128
        cnn_blocks = 2
        minibatch_size = 16
        rollout_length = 512

    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)
    policy_core = Policy_Core(
        int_action_size=7, ext_action_size=7,
        node_dim=node_dim,
        internal_state_size=1 + C,
        goal_size=mission_size, inventory_size=inventory_size,
        width=7, height=7, channel=3,
        dict_size=vocab_size, embedding_dim=embedding_dim, pad_token_id=0,
        hidden_size=hidden_size, layers=layers,
        cnn_channels=cnn_channels, cnn_blocks=cnn_blocks,
        empty_value=-1.0,
        use_aux=args.aux_coef is not None,
        external_flags=model_76.EXTERNAL_FLAGS,
        aux_span=args.max_thought_steps + 1,
        device=device, persistence_path=parameters_path
    ).to(device)
    ppo_learner = PPO(
        policy_model=Projector(
            policy_core, [0, 1, 2, 3, 4],
            component_flags=model_76.COMPONENT_FLAGS,
        ),
        device=device, persistence_path=parameters_path,
        lr=1e-4, minibatch_size=minibatch_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        use_graph=args.st_train,
        aux_coef=args.aux_coef,
    )
    memory = Graph_Memory(
        num_batches=len(game_ids),
        num_nodes=4096,
        max_edges_per_node=C,
        node_dim=node_dim,
        # node values live in [0, 1]^D; empty slots sit outside that range so they stay distinguishable
        start_node_value=0.5,
        empty_value=-1.0,
    )
    model_module = model_76
    agent_class = model_76.Model
    agent = agent_class(
        policy_model=policy_core,
        trainer=ppo_learner,
        context_collector=Collector(max_history=0),
        action_collector=Collector(max_history=0),
        valid_action_collector=Collector(max_history=0),
        graph_memory=memory,
        max_num_thought_steps=args.max_thought_steps,
        do_supervision=False,
        scheme=model_module.Scheme(args.scheme),
        has_intrinsic_reward=env.has_intrinsic_reward
    )

    asyncio.run(run(env, agent, rollout_length, verbose=not args.silent, env_seed=seed))
