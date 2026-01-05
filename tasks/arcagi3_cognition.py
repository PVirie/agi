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

from implementations.agents import random_agent, model_53
from implementations.core.torch.core import Action_Content_Core as Core
from implementations.learning_algorithms.torch.ppo import PPO
from implementations.learning_algorithms.torch.supervised import Basic_Learner
from implementations.core.states import State_Sequence as Collector
from implementations.core.energy_memory import Energy_Memory as Memory

torch.autograd.set_detect_anomaly(True)


def get_state_reward(state: Game_State) -> int:
    reward = state.delta_score
    state_type = state.state
    if state_type == Game_State_Type.WIN:
        reward = 10
    elif state_type == Game_State_Type.GAME_OVER:
        reward = -1
    elif state_type == Game_State_Type.IDLE:
        reward = 0
    elif state_type == Game_State_Type.NOT_FINISHED:
        reward = state.delta_score
        if not state.diff_from_last:
            reward = -0.1
    else:
        reward = 0
    return reward - 0.01  # small step penalty
    

async def run(env, agent):

    all_games_info = await env.list_games()
    all_public_game_ids = [game["game_id"] for game in all_games_info if game.get("game_type") == "public"]
    # now duplicate game to have at least 3 games for each id
    selected_game_ids = all_public_game_ids * 3
    await env.start(selected_game_ids=selected_game_ids)

    actions = [(Action_Type.RESET, ) for _ in range(len(selected_game_ids))]
    start_time = time.perf_counter()
    steps = 0
    while True:
        has_event, states = await env.execute(actions)
        last_idle = [
            s.state == Game_State_Type.IDLE for s in states
        ]
        next_done = [
            s.state == Game_State_Type.WIN or s.state == Game_State_Type.GAME_OVER for s in states
        ]
        last_truncated = [
            s.state == Game_State_Type.TRUNCATED or s.state == Game_State_Type.RESET for s in states
        ]
        last_reset = [
            s.state == Game_State_Type.RESET for s in states
        ]
        actions = agent.choose_action(
            last_idles=last_idle,
            next_dones=next_done,
            last_truncates=last_truncated,
            last_resets=last_reset,
            latest_frames=[state.frame for state in states],
            rewards=[get_state_reward(s) for s in states],
            next_available_actions=[
                [a.value for a in state.next_available_actions] for state in states
            ],
            force_train=steps % 10 == 9
        )
        actions = [
            ((Action_Type(a[0].item()), a[1].item(), a[2].item()) if a is not None else None) if not d else (Action_Type.RESET, )
            for a, d in zip(actions, next_done)
        ]
        await asyncio.sleep(1)

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > max_running_time:  # run for the specified max time
            logging.info("Max running time reached, stopping the experiment.")
            break

        steps += 1
        if steps % 10 == 0 or has_event:
            log_str = "; ".join([s.short_str() for s in states])
            logging.info(f"{steps}| States: [{log_str}]")
            logging.info(f"{steps}| Rewards: {[get_state_reward(s) for s in states]}")
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
    parser.add_argument("--scale",                  "-s",   type=str, default="medium", choices=["small", "medium", "large"], help="The scale of the neural network. Default is 'medium'.")
    parser.add_argument("--max-thought-steps",      "-mts", type=int, default=2, help="Maximum number of thought steps the agent can take before being forced to act externally.")
    parser.add_argument("--use-memory",             "-um",  action="store_true",                help="Enable the use of memory in the agent.")
    parser.add_argument("--with-supervision",       "-svl", action="store_true",                help="Enable supervised learning along with PPO.")
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

    experiment_path = f"{APP_ROOT}/experiments/arcagi3_cognition"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    env = ARCAGI3_Environment()

    random_agent = random_agent.Random_Agent("01")

    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)
    agent_core = Core(
        action_size=7, position_size=16,
        width=64, height=64, channel=4,
        hidden_size=64, layers=4,
        max_temporal_range=32, device=device, 
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
    memory = Memory(
        sizes=(1, 16, agent_core.content_size),
        max_slot_size=128
    )
    model_53_agent = model_53.Model_53(
        agent_core=agent_core, 
        trainer=ppo_learner, supervised_trainer=supervised_learner,
        context_collector=Collector(max_history=8),
        action_collector=Collector(max_history=8),
        valid_action_collector=Collector(max_history=8),
        memory=memory,
        max_num_thought_steps=args.max_thought_steps,
        do_supervision=args.with_supervision,
        use_memory=args.use_memory,
    )

    asyncio.run(run(env, model_53_agent))
