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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

from utilities.package_install import install

install("minigrid")
install("gymnasium[other]")

from utilities import scatter
from utilities.minigrid.environments import Multi_Environment
from utilities.tokenizer import Text_Tokenizer
from utilities.episode_recorder import Episode_Recorder
from colorama import Fore, Back, Style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import model_76
from implementations.networks.torch.policy.minigrid import Projector
from implementations.networks.torch.policy.minigrid import Policy_Core
from implementations.learning_algorithms.torch.ppo_graph import PPO
from implementations.collectors.states import State_Sequence as Collector
from implementations.memories.graph_memory import NP_Graph_Memory as Graph_Memory

torch.autograd.set_detect_anomaly(True)

def format_float(f):
    if f > 0:
        return Fore.GREEN + "{: .1f}".format(f) + Style.RESET_ALL
    elif f < 0:
        return Fore.RED + "{: .1f}".format(f) + Style.RESET_ALL
    else:
        return "{: .1f}".format(f)

async def run(env, agent, rollout_length=16, verbose=False):

    observations, info = env.reset()
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

        actions, _ = agent.choose_action(
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

        last_idle = [False for _ in observations]
        last_done = [terminations[i] or truncations[i] for i in range(len(observations))]
        last_truncated = [truncations[i] for i in range(len(observations))]
        last_reset = [False for _ in observations]

        stat_row = []
        for i in range(len(observations)):
            if terminations[i] or truncations[i]:
                total_score = infos[i]["episode"]["r"]
                total_returns[i] = (
                    session_return_update_alpha * total_returns[i]
                    + (1 - session_return_update_alpha) * total_score
                )
                stat_row.extend([infos[i]["episode"]["r"]])
            else:
                stat_row.extend([None])
        stat_recorder.record(stat_row)

        steps += 1
        if any([r != 0 for r in rewards]) and verbose:
            logging.info(f"{steps}| Rewards: {', '.join([format_float(r) for r in rewards])}")

        if steps % rollout_length == 0:
            ppo_learner.update_learning_rate(time=elapsed_time / max_running_time)

        if steps % (rollout_length * 2) == 0 or should_stop:
            logging.info(f"{steps}| Returns: {', '.join([format_float(s) for s in total_returns])}")
            logging.info(f"{steps}| Selected actions: {actions}")

            # save 
            policy_core.save()
            ppo_learner.save()
            tokenizer.save(f"{experiment_path}/parameters")

        if steps % (rollout_length * 10) == 0:
            # compute estimated time left
            logging.info(f"Completed {steps} steps.")
            logging.info(f"Current elapsed time: {elapsed_time:.2f} seconds.")
            logging.info(f"Expected time left: {max_running_time - elapsed_time:.2f} seconds.")

        if steps % 1000 == 0:
            stat_recorder.write()

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
    parser.add_argument("--max-thought-steps",      "-mts", type=int, default=2, help="Maximum number of thought steps the agent can take before being forced to act externally.")
    parser.add_argument("--scheme",                 "-sch", type=str, default="full", help="The scheme to use for the agent's decision making. Default is 'reactive'.")
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
    random.seed(20260625)  
    torch.manual_seed(20260625)
    np.random.seed(20260625)
    torch.use_deterministic_algorithms(True)

    experiment_path = f"{APP_ROOT}/experiments/minigrid_76_graph_size_{args.scale}_scheme_{args.scheme}"
    if args.reset:
        # clear the experiment path
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        exit()
    os.makedirs(experiment_path, exist_ok=True)

    vocab_size = 256
    tokenizer = Text_Tokenizer(max_vocab_size=vocab_size)
    tokenizer.load(f"{experiment_path}/parameters")

    game_ids=["BabyAI-MiniBossLevel-v0"]*64
    env = Multi_Environment(
        game_ids=game_ids,
        tokenizer=tokenizer,
        mission_max_len=16,
        full_mdp=False,
        include_inventory=False
    )

    mission_size = env.mission_max_len
    inventory_size = 3 # inventory include 1 slot for current direction and 2 slots for items
    content_size = mission_size + inventory_size + 7 * 7 * 3 # mission + inv + image
    
    stat_recorder = Episode_Recorder(f"{experiment_path}/statistics", headers=[f"minigrid/return" for _ in list(range(env.batch_size))])
    
    if args.scale == "small":
        hidden_size = 128
        embedding_dim = 16
        C = 16
        layers = [16, 32, 32] # basic impala
        minibatch_size = 32
        rollout_length = 128
    elif args.scale == "medium":
        hidden_size = 128
        embedding_dim = 16
        C = 16
        layers = [16, 32, 64, 64] # medium impala
        minibatch_size = 32
        rollout_length = 128
    else:  # large
        hidden_size = 128
        embedding_dim = 16
        C = 16
        layers = [16, 32, 64, 128, 128] # large impala
        minibatch_size = 32
        rollout_length = 128

    parameters_path = f"{experiment_path}/parameters"
    os.makedirs(parameters_path, exist_ok=True)
    policy_core = Policy_Core(
        int_action_size=5, ext_action_size=7,
        write_action_size=16,
        internal_state_size=1 + C,
        goal_size=mission_size, inventory_size=inventory_size,
        width=7, height=7, channels=3,
        dict_size=vocab_size, embedding_dim=embedding_dim, pad_token_id=0,
        hidden_size=hidden_size, layers=layers,
        device=device, persistence_path=parameters_path
    ).to(device)
    ppo_learner = PPO(
        policy_model=Projector(policy_core, [0, 1, 2, 3, 4]),
        device=device, persistence_path=parameters_path, minibatch_size=minibatch_size
    )
    memory = Graph_Memory(
        num_batches=env.batch_size,
        num_nodes=4096,
        max_edges_per_node=C,
        node_dim=1
    )
    agent = model_76.Model_76(
        policy_model=policy_core,
        trainer=ppo_learner,
        context_collector=Collector(max_history=0),
        action_collector=Collector(max_history=0),
        valid_action_collector=Collector(max_history=0),
        graph_memory=memory,
        max_num_thought_steps=args.max_thought_steps,
        do_supervision=False,
        scheme=model_76.Scheme(args.scheme)
    )

    asyncio.run(run(env, agent, rollout_length, verbose=not args.silent))
