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
from utilities.arcagi3.environments import Game_State, Action_Type, Game_State_Type, ARCAGI3_Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import random_agent, model_54
from implementations.core.torch.core import Action_Content_Core as Core
from implementations.learning_algorithms.torch.ppo import PPO
from implementations.learning_algorithms.torch.supervised import Basic_Learner
from implementations.core.states import State_Sequence as Collector
from implementations.core.energy_memory import Energy_Memory as Memory

torch.autograd.set_detect_anomaly(True)


def get_env_action(agent_action, state: Game_State):
    if agent_action is None:
        return None
    
    if state.state == Game_State_Type.WIN   :
        return (Action_Type.RESET, )
    elif state.state == Game_State_Type.GAME_OVER:
        return (Action_Type.RESTART, )
    else:
        return (Action_Type(agent_action[0].item()), agent_action[1].item(), agent_action[2].item())


async def run(env, agent):

    all_games_info = await env.list_games()
    all_public_game_ids = [game["game_id"] for game in all_games_info if game.get("game_type") == "public"]
    # now duplicate game to have at least 3 games for each id
    selected_game_ids = all_public_game_ids * 3
    await env.start(selected_game_ids=selected_game_ids)

    actions = [(Action_Type.RESET, ) for _ in range(len(selected_game_ids))]
    auxiliary_state = [["explore", 200] for _ in range(len(selected_game_ids))]
    start_time = time.perf_counter()
    steps = 0
    while True:
        has_event, states = await env.execute(actions)
        actions = agent.choose_action(
            latest_frames=[state.frame for state in states]
        )
        actions = [get_env_action(a, s, x) for a, s, x in zip(actions, states, auxiliary_state)]
        await asyncio.sleep(1)

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > max_running_time:  # run for the specified max time
            logging.info("Max running time reached, stopping the experiment.")
            break

        steps += 1
        if steps % 10 == 0 or has_event:
            log_str = "; ".join([s.short_str() for s in states])
            logging.info(f"{steps}| States: [{log_str}]")
            logging.info(f"{steps}| Selected actions: {actions}")

        if steps % 100 == 0:
            # compute estimated time left
            logging.info(f"Completed {steps} steps.")
            logging.info(f"Current elapsed time: {elapsed_time:.2f} seconds.")
            logging.info(f"Expected time left: {max_running_time - elapsed_time:.2f} seconds.")


    report = await env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",                  "-r",   action="store_true")
    parser.add_argument("--hours",                  "-hr",  type=float, default=0.05, help="Number of hours to train the agent. Fractional hours allowed.")
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
    random.seed(20260104)  
    torch.manual_seed(20260104)
    np.random.seed(20260104)
    torch.use_deterministic_algorithms(True)

    experiment_path = f"{APP_ROOT}/experiments/arcagi3_navigator"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    env = ARCAGI3_Environment()

    random_agent = random_agent.Random_Agent("01")

    asyncio.run(run(env, random_agent))
