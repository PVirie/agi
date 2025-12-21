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

import utilities
from utilities.arcagi3.environments import Action_Type, Game_State_Type, ARCAGI3_Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import random_agent, model_53
from implementations.core.torch.sfstct_core import SF_STCT_Core as Core
from implementations.core.states import State_Sequence as Collector
from implementations.learning_algorithms.torch.ppo import PPO
from implementations.learning_algorithms.torch.supervised import Basic_Learner

torch.autograd.set_detect_anomaly(True)


async def run(env, agent):

    all_games_info = await env.list_games()
    actions = [(Action_Type.RESET, ) for _ in range(4)]

    await env.start(selected_game_ids=[game["game_id"] for game in all_games_info[:4]])

    start_time = time.perf_counter()
    steps = 0
    while True:
        states = await env.execute(actions)
        next_done = [
            s.state == Game_State_Type.WIN or s.state == Game_State_Type.GAME_OVER for s in states
        ]
        last_truncated = [
            s.state == Game_State_Type.TRUNCATED for s in states
        ]
        actions = agent.choose_action(
            last_idles=[s.state == Game_State_Type.IDLE for s in states],
            next_dones=next_done,
            last_truncates=last_truncated,
            latest_frames=[state.frame for state in states],
            scores=[state.score for state in states],
            next_available_actions=[state.next_available_actions for state in states],
            force_train=steps % 10 == 9
        )
        actions = [
            (Action_Type.RESET if d else a[0], a[1], a[2]) if a is not None else None
            for a,d in zip(actions, next_done)
        ]
        await asyncio.sleep(1)

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > max_running_time:  # run for the specified max time
            logging.info("Max running time reached, stopping the experiment.")
            break

        steps += 1
        if steps % 10 == 0:
            logging.info(f"{steps}| Selected actions: {actions}; Scores: {[s.score for s in states]}")

        if steps % 100 == 0:
            # compute estimated time left
            logging.info(f"Completed {steps} steps.")
            logging.info(f"Current elapsed time: {elapsed_time:.2f} seconds.")
            logging.info(f"Expected time left: {max_running_time - elapsed_time:.2f} seconds.")


    await env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",                  "-r",   action="store_true")
    parser.add_argument("--hours",                  "-hr",  type=float, default=0.05, help="Number of hours to train the agent. Fractional hours allowed.")
    parser.add_argument("--scale",                  "-s",   type=str, default="medium", choices=["small", "medium", "large"], help="The scale of the neural network. Default is 'medium'.")
    parser.add_argument("--with-supervision",       "-svl", action="store_true",                help="Enable supervised learning along with PPO.")
    parser.add_argument("--no-thought",             "-nth", action="store_true",                help="Disable thoughts in favor of fixed steps.")
    parser.add_argument("--no-reference",           "-nrf", action="store_true",                help="Disable reference and use traditional PE.")
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
    random.seed(20251118)  
    torch.manual_seed(20251118)
    np.random.seed(20251118)
    torch.use_deterministic_algorithms(True)

    experiment_path = f"{APP_ROOT}/experiments/arcagi3"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    env = ARCAGI3_Environment()

    random_agent = random_agent.Random_Agent("agent_001")

    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)
    agent_core = Core(
        action_size=6, position_size=16,
        width=64, height=64, channel=4,
        hidden_size=128, layers=2,
        device=device, 
        persistence_path=parameters_path
    ).to(device)
    ppo_learner = PPO(
        agent=agent_core, device=device,
        persistence_path=parameters_path
    )
    supervised_learner = Basic_Learner(
        agent=agent_core, device=device,
        persistence_path=parameters_path
    )
    model_53_agent = model_53.Model_53(
        agent_core=agent_core, 
        trainer=ppo_learner, supervised_trainer=supervised_learner,
        context_collector=Collector(max_history=8),
        action_collector=Collector(max_history=8),
        do_supervision=args.with_supervision
    )

    asyncio.run(run(env, model_53_agent))
