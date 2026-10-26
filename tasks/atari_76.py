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

install("opencv-python-headless") 
install("ale-py")
install("gymnasium[atari, other]")
install("colorama")

import ale_py
from utilities.atari.environments import Multi_Atari_Environment, ATARI_ACTIONS
from utilities.episode_recorder import Episode_Recorder
from colorama import Fore, Back, Style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import model_76
from implementations.agents import model_77
from implementations.networks.torch.policy.atari import Policy_Core, Projector
from implementations.learning_algorithms.torch.ppo_combine import PPO
from implementations.learning_algorithms.torch.ppo_graph import PPO as PPO_Graph
from implementations.collectors.states import State_Sequence as Collector
from implementations.memories.graph_memory import NP_Graph_Memory as Graph_Memory

torch.autograd.set_detect_anomaly(False)

def format_float(f):
    if f > 0:
        return Fore.GREEN + "{: .1f}".format(f) + Style.RESET_ALL
    elif f < 0:
        return Fore.RED + "{: .1f}".format(f) + Style.RESET_ALL
    else:
        return "{: .1f}".format(f)

# logged every 2 rollouts, see Multi_Atari_Environment.diagnostics and PPO.last_stats
DIAGNOSTIC_KEYS = [
    "game_reward_per_step", "rewarded_step_fraction",
    "rnd_error", "rnd_reward", "episodic_reward", "modulator", "intrinsic_reward",
    "cells_seen", "categorical_seen",
    "entropy", "approx_kl", "pg_loss", "v_loss",
]

async def run(env, agent, rollout_length=16, verbose=False, env_seed=None):

    observations, info = env.reset(seed=env_seed)
    rewards = [np.float32(0) for _ in observations]
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
            rewards=[r for r in rewards],
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
                    metrics[0][i],
                    metrics[1][i],
                ])
            else:
                stat_row.extend([None, None, None, None, None])
        stat_recorder.record(stat_row)

        steps += 1
        # log extrinsic rewards only; with RND the training reward is nonzero on almost every step
        extrinsic_rewards = [infos[i]["original_reward"] if actions[i] is not None else 0 for i in range(len(observations))]
        if any([r != 0 for r in extrinsic_rewards]) and verbose:
            logging.info(f"{steps}| Rewards: {', '.join([format_float(r) for r in extrinsic_rewards])}")

        # constant LR on purpose: a wall-clock anneal would give the arms different schedules
        # at the same environment step, which is the axis they are compared on

        if steps % (rollout_length * 2) == 0 or should_stop:
            logging.info(f"{steps}| Returns: {', '.join([format_float(s) for s in total_returns])}")
            logging.info(f"{steps}| Selected actions: {actions}")

            diagnostics = {**env.diagnostics(), **ppo_learner.last_stats}
            logging.info(f"{steps}| Diagnostics: " + ", ".join(f"{k}={diagnostics[k]:.4g}" for k in DIAGNOSTIC_KEYS if k in diagnostics))
            diagnostic_recorder.record([env.total_env_steps()] + [diagnostics.get(k) for k in DIAGNOSTIC_KEYS])
            diagnostic_recorder.write()
            if torch.cuda.is_available():
                # running maxima since process start, never reset: this is the probe measurement
                logging.info(f"{steps}| Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB reserved")

            # save 
            policy_core.save()
            ppo_learner.save()
            env.save()

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


# Peak *reserved* VRAM, measured at 512 envs and rollout_length 512 with two minibatch sizes per scale:
#   small   16 -> 1.70 GB,  48 -> 4.42 GB
#   medium   8 -> 2.64 GB,  24 -> 7.19 GB
#   large    4 -> 2.99 GB,  12 -> 7.56 GB
MEMORY_FIT = {  # scale: (fixed_gb, gb_per_minibatch), at 512 envs and rollout_length 512
    "small":  (0.34, 0.085),
    "medium": (0.365, 0.284),
    "large":  (0.705, 0.571),
}


def fit_rollout_and_minibatch(scale, max_thought_steps, num_envs, target_gb):
    # Linear model, both terms grow in proportion to rollout_length:
    #     peak_gb = rollout_length / 512 * (fixed_gb * num_envs / 512 + gb_per_minibatch * minibatch_size)
    # Scaling the whole intercept with rollout_length over-reserves a little, since the parameter and
    # optimizer share of fixed_gb (~0.35 GB at large, ~0 at small) is constant. That errs toward not OOMing.
    # rollout_length scales with max_thought_steps + 1 so a rollout keeps the same number of game steps.
    fixed_gb, gb_per_minibatch = MEMORY_FIT[scale]
    fixed_gb *= num_envs / 512

    rollout_length = int(512 * (max_thought_steps + 1) * 4 / 3)
    # longest rollout at which minibatch 1 still fits the target
    rollout_length = min(rollout_length, int(512 * target_gb / (fixed_gb + gb_per_minibatch)))

    minibatch_size = int((target_gb * 512 / rollout_length - fixed_gb) / gb_per_minibatch)
    return rollout_length, max(1, minibatch_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",                  "-r",   action="store_true")
    parser.add_argument("--hours",                  "-hr",  type=float, default=0.05, help="Number of hours to train the agent. Fractional hours allowed.")
    parser.add_argument("--scale",                  "-s",   type=str, default="medium", choices=["small", "medium", "large"], help="The scale of the neural network. Default is 'medium'.")
    parser.add_argument("--max-thought-steps",      "-mts", type=int, default=2, help="Maximum number of thought steps the agent can take before being forced to act externally.")
    parser.add_argument("--scheme",                 "-sch", type=str, default="full", help="The scheme to use for the agent's decision making. Default is 'full'.")
    parser.add_argument("--st-train",               "-stt",   action="store_true", help="Use spatio-temporal training for the agent. Default is False.")
    parser.add_argument("--sie",                    "-sie",   action="store_true", help="Use separate internal-external attention (model 77). Default is False.")
    parser.add_argument("--silent",                 "-silent", action="store_true", help="Disable reward logging for cleaner output.")
    parser.add_argument("--low-var",                "-lv", action="store_true", help="Use low-variance (lower GAE-lambda) advantage estimation. Default is False.")
    parser.add_argument("--rnd",                    "-rnd", action="store_true", help="Add Random Network Distillation intrinsic reward. Default is False.")
    parser.add_argument("--rnd-weight",             "-rndw", type=float, default=0.5, help="Weight of the intrinsic reward. Default is 0.5.")
    parser.add_argument("--rnd-lr",                 "-rndlr", type=float, default=1e-5, help="RND predictor learning rate; lower makes the intrinsic reward decay slower. Default is 1e-5.")
    parser.add_argument("--rnd-episodic",           "-rnde", action="store_true", help="NGU-style intrinsic reward: per-episode RAM-cell novelty, modulated by RND. Default is False.")
    parser.add_argument("--rnd-cell-bin",           "-rndcb", type=int, default=8, help="Bin size of the position bytes in an episodic RAM cell. Default is 8.")
    parser.add_argument("--rnd-stats-window",       "-rndsw", type=int, default=0, help="Agent-env steps over which the RND reward statistics are averaged; 0 keeps the whole history. Default is 0.")
    parser.add_argument("--ent-coef",               "-ent", type=float, default=0.01, help="PPO entropy coefficient. Default is 0.01.")
    parser.add_argument("--minibatch-size",         "-mb", type=int, default=0, help="Envs per PPO minibatch; 0 fits it to the scale's VRAM target. Default is 0.")
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
    seed = 20261004
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    experiment_path = f"{APP_ROOT}/experiments/atari_76_size_{args.scale}_scheme_{args.scheme}_mts_{args.max_thought_steps}_v"
    if args.st_train:
        experiment_path += "_stmean"
    if args.sie:
        experiment_path += "_sie"
    if args.low_var:
        experiment_path += "_lv"
    if args.rnd:
        experiment_path += f"_rnd{args.rnd_weight:g}"
        if args.rnd_lr != parser.get_default("rnd_lr"):
            experiment_path += f"_lr{args.rnd_lr:g}"
        if args.rnd_episodic:
            experiment_path += "_ep"
            if args.rnd_cell_bin != parser.get_default("rnd_cell_bin"):
                experiment_path += f"{args.rnd_cell_bin}"
        if args.rnd_stats_window > 0:
            experiment_path += f"_sw{args.rnd_stats_window}"
    if args.ent_coef != parser.get_default("ent_coef"):
        experiment_path += f"_ent{args.ent_coef:g}"

    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)
    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)

    game_ids = ["ALE/MontezumaRevenge-v5"] * 512
    # game_ids = ["ALE/Pitfall-v5"] * 512
    # game_ids = ["ALE/Adventure-v5"] * 512
    
    vocab_size = 256

    env = Multi_Atari_Environment(
        game_ids=game_ids,
        episodic_life=False,
        prepend_objective=False,
        rnd_reward=args.rnd,
        rnd_weight=args.rnd_weight,
        rnd_lr=args.rnd_lr,
        rnd_episodic=args.rnd_episodic,
        rnd_cell_bin_size=args.rnd_cell_bin,
        rnd_stats_window_steps=args.rnd_stats_window if args.rnd_stats_window > 0 else None,
        device=device,
        persistence_path=parameters_path,
    )

    stat_recorder = Episode_Recorder(f"{experiment_path}/statistics", headers=[f"{gid}/{stat}" for gid in game_ids for stat in ["env_steps", "return", "length", "edges_per_node", "graph_nodes"]])
    diagnostic_recorder = Episode_Recorder(f"{experiment_path}/diagnostics", headers=["env_steps"] + DIAGNOSTIC_KEYS)
    
    if args.scale == "small":
        # 12/16 GB
        hidden_size = 128
        embedding_dim = 8
        C = 4
        layers = [1, 1, 1]
        target_gb = 12
    elif args.scale == "medium":
        # 20/24 GB
        hidden_size = 256
        embedding_dim = 16
        C = 8
        layers = [1, 1, 1, 1]
        target_gb = 20
    else:
        # 28/32 GB
        hidden_size = 512
        embedding_dim = 32
        C = 8
        layers = [1, 1, 1, 1]
        target_gb = 28

    policy_core = Policy_Core(
        int_action_size=7, ext_action_size=len(ATARI_ACTIONS),
        write_action_size=16,
        internal_state_size=1 + C,
        external_state_size=env.observation_size,
        dict_size=vocab_size, embedding_dim=embedding_dim, pad_token_id=0,
        hidden_size=hidden_size, layers=layers,
        device=device, persistence_path=parameters_path
    ).to(device)

    rollout_length, minibatch_size = fit_rollout_and_minibatch(
        scale=args.scale,
        max_thought_steps=args.max_thought_steps,
        num_envs=len(game_ids),
        target_gb=target_gb,
    )
    if args.minibatch_size > 0:
        # the fit over-reserves at long rollouts (its whole intercept scales with rollout_length), so it can be raised by hand
        minibatch_size = args.minibatch_size

    # gamma is per agent step: spread a 0.997-per-game-step discount over up to max_thought_steps + 1 agent steps
    gamma = 0.997 ** (1.0 / (args.max_thought_steps + 1))
    logging.info(f"Rollout length: {rollout_length}, minibatch size: {minibatch_size}, gamma: {gamma:.5f}")

    if args.st_train:
        ppo_learner = PPO_Graph(
            policy_model=Projector(policy_core, [0, 1, 2, 3, 4]),
            device=device, persistence_path=parameters_path,
            lr=1e-4, minibatch_size=minibatch_size,
            gamma=gamma,
            gae_lambda=0.95 if not args.low_var else 0.90,
            ent_coef=args.ent_coef,
        )
    else:
        ppo_learner = PPO(
            policy_model=Projector(policy_core, [0, 1, 2, 3, 4]),
            device=device, persistence_path=parameters_path,
            lr=1e-4, minibatch_size=minibatch_size,
            gamma=gamma,
            gae_lambda=0.95 if not args.low_var else 0.90,
            ent_coef=args.ent_coef,
        )
    memory = Graph_Memory(
        num_batches=len(game_ids),
        num_nodes=4096,
        max_edges_per_node=C,
        node_dim=1,
        start_node_value=vocab_size - 1
    )
    model_module = model_77 if args.sie else model_76
    agent_class = model_77.Model_77 if args.sie else model_76.Model_76
    agent = agent_class(
        policy_model=policy_core,
        trainer=ppo_learner,
        context_collector=Collector(max_history=0),
        action_collector=Collector(max_history=0),
        valid_action_collector=Collector(max_history=0),
        graph_memory=memory,
        max_num_thought_steps=args.max_thought_steps,
        do_supervision=False,
        scheme=model_module.Scheme(args.scheme)
    )

    asyncio.run(run(env, agent, rollout_length, verbose=not args.silent, env_seed=seed))
