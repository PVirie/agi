import os
import sys
import subprocess
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

install("ale-py")
install("colorama")

from ale_py.vector_env import AtariVectorEnv
from colorama import Fore, Back, Style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import random_agent, model_base
from implementations.networks.torch.policy.dualism import Policy_Core, Projector
from implementations.networks.torch.value.mix_noxy import Value_Core
from implementations.learning_algorithms.torch.ppo import PPO
from implementations.collectors.states import State_Sequence as Collector

torch.autograd.set_detect_anomaly(True)

def format_float(f):
    if f > 0:
        return Fore.GREEN + "{: .1f}".format(f) + Style.RESET_ALL
    elif f < 0:
        return Fore.RED + "{: .1f}".format(f) + Style.RESET_ALL
    else:
        return "{: .1f}".format(f)
    
async def run(env, agent, rollout_length=16):

    observations, info = env.reset()
    rewards = [np.float32(0) for _ in observations]
    last_idle = [False for _ in observations]
    last_done = [False for _ in observations]
    last_truncated = [False for _ in observations]
    last_reset = [False for _ in observations]

    total_returns = [0 for _ in observations]
    session_return_stat = 0
    session_return_update_alpha = 0.95
    start_time = time.perf_counter()
    steps = 0
    while True:
        elapsed_time = time.perf_counter() - start_time
        should_stop = elapsed_time > max_running_time  # run for the specified max time

        actions = agent.choose_action(
            last_idles=last_idle,
            last_dones=last_done,
            last_truncates=last_truncated,
            last_resets=last_reset,
            latest_frames=[obs.astype(np.float32) / 255.0 for obs in observations],
            rewards=[r.item() for r in rewards],
            next_available_actions=[
                list(range(action_space_size)) for _ in observations
            ],
            force_train=steps % rollout_length == 0 or should_stop,
        )
        actions = [int(a[0].item()) if a is not None else None for a in actions]

        observations, rewards, terminations, truncations, infos = env.step(np.array(actions, dtype=np.int32))

        last_idle = [False for _ in observations]
        last_done = [terminations[i] or truncations[i] for i in range(len(observations))]
        last_truncated = [truncations[i] for i in range(len(observations))]
        last_reset = [False for _ in observations]

        for i in range(len(observations)):
            total_returns[i] += rewards[i].item()
            if terminations[i] or truncations[i]:
                session_return_stat = session_return_update_alpha * total_returns[i] + (1 - session_return_update_alpha) * session_return_stat
                total_returns[i] = 0

        steps += 1
        if any([r != 0 for r in rewards]):
            logging.info(f"{steps}| Rewards: {', '.join([format_float(r) for r in rewards])}")

        if steps % rollout_length == 0:
            ppo_learner.update_learning_rate(time=elapsed_time / max_running_time)

        if steps % (rollout_length * 2) == 0 or should_stop:
            logging.info(f"{steps}| Session return stat: {session_return_stat}")
            logging.info(f"{steps}| Selected actions: {actions}")

            # save 
            policy_core.save()
            value_core.save()
            ppo_learner.save()

        if steps % (rollout_length * 10) == 0:
            # compute estimated time left
            logging.info(f"Completed {steps} steps.")
            logging.info(f"Current elapsed time: {elapsed_time:.2f} seconds.")
            logging.info(f"Expected time left: {max_running_time - elapsed_time:.2f} seconds.")

        if should_stop:
            logging.info("Max running time reached, stopping the experiment.")
            break

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",                  "-r",   action="store_true")
    parser.add_argument("--hours",                  "-hr",  type=float, default=0.05, help="Number of hours to train the agent. Fractional hours allowed.")
    parser.add_argument("--scale",                  "-s",   type=str, default="medium", choices=["small", "medium", "large"], help="The scale of the neural network. Default is 'medium'.")
    parser.add_argument("--max-thought-steps",      "-mts", type=int, default=0, help="Maximum number of thought steps the agent can take before being forced to act externally.")
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
    random.seed(20260221)  
    torch.manual_seed(20260221)
    np.random.seed(20260221)
    torch.use_deterministic_algorithms(True)

    experiment_path = f"{APP_ROOT}/experiments/atari_64"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    env = AtariVectorEnv(
        game="pong",                # The ROM id not name, i.e., camel case compared to `gymnasium.make` name versions
        num_envs=16,                # Number of parallel environments
        img_height=64,              # Height to resize frames to
        img_width=32,               # Width to resize frames to
        maxpool=True,               # 1. Solves "Invisibility" (Flickering)
        stack_num=4,                # 2. Solves "Motion" (Velocity)
        frameskip=4,                # 3. Standard time resolution
        grayscale=True,             # 4. Removes noise (Color is usually irrelevant in Atari)
        episodic_life=True,         # Recommended for harder games (Breakout/Montezuma)
        reward_clipping=True,
    )
    action_space_size = env.single_action_space.n.item()

    random_agent = random_agent.Random_Agent("01")

    if args.scale == "small":
        history_steps = 0
        hidden_size = 64
        conv_layers = [16, 32, 32] # basic impala
        rollout_length = 128
        minibatch_size = 8
        position_size = 2
    elif args.scale == "medium":
        history_steps = 0
        hidden_size = 64
        conv_layers = [16, 32, 64] # medium impala
        rollout_length = 256
        minibatch_size = 8
        position_size = 8
    else:  # large
        history_steps = 0
        hidden_size = 256
        conv_layers = [32, 64, 128, 128, 256, 256] # large impala
        rollout_length = 256
        minibatch_size = 8
        position_size = 16

    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)
    policy_core = Policy_Core(
        int_action_size=2, ext_action_size=action_space_size, 
        position_size=position_size,
        width=32, height=64, channel=4,
        hidden_size=hidden_size, layers=conv_layers,
        history_steps=history_steps, max_temporal_len=rollout_length,
        device=device, persistence_path=parameters_path
    ).to(device)
    value_core = Value_Core(
        int_action_size=2, ext_action_size=action_space_size, position_size=position_size,
        width=32, height=64, channel=4,
        hidden_size=hidden_size, layers=conv_layers,
        history_steps=history_steps, max_temporal_len=rollout_length,
        device=device, persistence_path=parameters_path
    ).to(device)
    ppo_learner = PPO(
        policy_model=Projector(policy_core, [1]), value_model=value_core,
        device=device, persistence_path=parameters_path, minibatch_size=minibatch_size,
        aux_coef=0.1
    )
    agent = model_base.Model_Base(
        policy_model=policy_core, value_model=value_core,
        trainer=ppo_learner,
        context_collector=Collector(max_history=history_steps),
        action_collector=Collector(max_history=history_steps),
        valid_action_collector=Collector(max_history=history_steps),
        max_num_thought_steps=args.max_thought_steps,
        do_supervision=False
    )

    asyncio.run(run(env, agent, rollout_length))
