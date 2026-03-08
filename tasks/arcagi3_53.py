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
from utilities.arcagi3.environments import Game_State, Action_Type, Game_State_Type, ARCAGI3_Remote_Environment

install("arc-agi")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import random_agent, model_53
from implementations.networks.torch.policy.base_xy import Policy_Core, Projector
from implementations.networks.torch.value.conv import Value_Core
from implementations.learning_algorithms.torch.ppo import PPO
from implementations.collectors.states import State_Sequence as Collector
from implementations.memories.energy_memory import Energy_Memory as Memory

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
        if state.matched_relative_index >= 0:
            reward = -0.1 # small penalty for action that causes looping
    else:
        reward = 0
    return reward
    

async def run(env, agent, rollout_length=16, verbose=False):

    all_games_info = await env.list_games()
    all_public_game_ids = [game["game_id"] for game in all_games_info if game.get("game_type") == "public"]
    # now duplicate game to have at least 3 games for each id
    selected_game_ids = all_public_game_ids * 8
    await env.start(selected_game_ids=selected_game_ids)

    actions = [(Action_Type.RESET, ) for _ in range(len(selected_game_ids))]
    start_time = time.perf_counter()
    steps = 0
    while True:
        elapsed_time = time.perf_counter() - start_time
        should_stop = elapsed_time > max_running_time  # run for the specified max time

        has_event, states = await env.execute(actions)
        last_idle = [
            s.state == Game_State_Type.IDLE for s in states
        ]
        last_done = [
            s.state == Game_State_Type.WIN or s.state == Game_State_Type.GAME_OVER or s.state == Game_State_Type.TRUNCATED or s.state == Game_State_Type.RESET
            for s in states
        ]
        last_truncated = [
            s.state == Game_State_Type.TRUNCATED or s.state == Game_State_Type.RESET for s in states
        ]
        last_reset = [
            s.state == Game_State_Type.RESET for s in states
        ]
        actions = agent.choose_action(
            last_idles=last_idle,
            last_dones=last_done,
            last_truncates=last_truncated,
            last_resets=last_reset,
            latest_frames=[state.frame for state in states],
            rewards=[get_state_reward(s) for s in states],
            next_available_actions=[
                [a.value for a in state.next_available_actions] for state in states
            ],
            force_train=steps % rollout_length == 0 or should_stop,
        )
        actions = [
            (Action_Type(a[0].item()), a[1].item(), a[2].item()) if a is not None else None 
            for a in actions
        ]
        await asyncio.sleep(1)

        if should_stop:
            logging.info("Max running time reached, stopping the experiment.")
            break

        steps += 1
        if steps % (rollout_length) == 0:
            ppo_learner.update_learning_rate(time=elapsed_time / max_running_time)

        if (steps % (rollout_length) == 0 or has_event) and verbose:
            log_str = "; ".join([s.short_str() for s in states])
            logging.info(f"{steps}| States: [{log_str}]")
            logging.info(f"{steps}| Rewards: {[get_state_reward(s) for s in states]}")
            logging.info(f"{steps}| Selected actions: {actions}")

        if steps % (rollout_length) == 0:
            # save 
            policy_core.save()
            value_core.save()
            ppo_learner.save()

        if steps % (rollout_length * 10) == 0:
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
    parser.add_argument("--max-thought-steps",      "-mts", type=int, default=2, help="Maximum number of thought steps the agent can take before being forced to act externally.")
    parser.add_argument("--scheme",                 "-sch", type=str, default="reactive", help="The scheme to use for the agent's decision making. Default is 'reactive'.")
    parser.add_argument("--with-auxiliary",         "-aux", action="store_true", help="Enable auxiliary loss along with PPO.")
    parser.add_argument("--silent",                 "-silent", action="store_true", help="Disable reward logging for cleaner output.")
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

    experiment_path = f"{APP_ROOT}/experiments/arcagi3_53"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    env = ARCAGI3_Remote_Environment()

    random_agent = random_agent.Random_Agent("01")

    if args.scale == "small":
        history_steps = 3
        hidden_size = 64
        conv_layers = [64, 64, 64] # basic impala
        rollout_length = 64
        minibatch_size = 4
        position_size = 4
    elif args.scale == "medium":
        history_steps = 7
        hidden_size = 128
        conv_layers = [64, 64, 64, 64] # medium impala
        rollout_length = 64
        minibatch_size = 4
        position_size = 8
    else:  # large
        history_steps = 15
        hidden_size = 256
        conv_layers = [64, 64, 128, 128, 256, 256] # large impala
        rollout_length = 64
        minibatch_size = 4
        position_size = 16

    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)
    policy_core = Policy_Core(
        int_action_size=6, ext_action_size=7, position_size=position_size,
        width=64, height=64, channel=4,
        hidden_size=hidden_size, layers=conv_layers,
        history_steps=history_steps, max_temporal_len=rollout_length,
        device=device, persistence_path=parameters_path
    ).to(device)
    value_core = Value_Core(
        position_size=position_size,
        width=64, height=64, channel=4,
        output_dims=3,
        layers=conv_layers,
        device=device, persistence_path=parameters_path
    ).to(device)
    ppo_learner = PPO(
        policy_model=Projector(policy_core, [0, 1, 2, 3, 4]), value_model=value_core,
        device=device, persistence_path=parameters_path, minibatch_size=minibatch_size,
        aux_coef=0.1 if args.with_auxiliary else None
    )
    memory = Memory(
        sizes=(1, position_size, policy_core.content_size),
        max_slot_size=256
    )
    agent = model_53.Model_53(
        policy_model=policy_core, value_model=value_core,
        trainer=ppo_learner,
        context_collector=Collector(max_history=history_steps),
        action_collector=Collector(max_history=history_steps),
        valid_action_collector=Collector(max_history=history_steps),
        memory=memory,
        max_num_thought_steps=args.max_thought_steps,
        do_supervision=args.with_auxiliary,
        scheme=model_53.Scheme(args.scheme)
    )

    asyncio.run(run(env, agent, rollout_length, verbose=not args.silent))
