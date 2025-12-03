import os
import sys
import subprocess
import logging
import random
import argparse
import numpy as np

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

import utilities

arcagi_path = f"{APP_ROOT}/cache/ARC-AGI-3-Agents"
os.makedirs(arcagi_path, exist_ok=True)
if len(os.listdir(arcagi_path)) == 0:
    subprocess.run(["git", "clone", "https://github.com/arcprize/ARC-AGI-3-Agents.git", f"{arcagi_path}"])
    utilities.install_pyproject_toml(f"{arcagi_path}/pyproject.toml")
    utilities.install("langchain-openai")

sys.path.append(os.path.join(arcagi_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import register_agent_class, random_agent, model_53
from implementations.core.torch.sfstct_core import SF_STCT_Core as Core
from implementations.core.states import State_Sequence as Collector
from implementations.rl_algorithms.torch.ppo import PPO as Learner

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",                  "-r",   action="store_true")
    parser.add_argument("--scale",                  "-s",   type=str, default="medium", choices=["small", "medium", "large"], help="The scale of the neural network. Default is 'medium'.")
    parser.add_argument("--no-thought",             "-nth", action="store_true",                help="Disable thoughts in favor of fixed steps.")
    parser.add_argument("--no-reference",           "-nrf", action="store_true",                help="Disable reference and use traditional PE.")
    args = parser.parse_args()

    # For reproducibility (https://docs.pytorch.org/docs/stable/notes/randomness.html)
    random.seed(20251118)  
    torch.manual_seed(20251118)
    np.random.seed(20251118)
    torch.use_deterministic_algorithms(True)

    agent_001 = random_agent.Random_Agent("agent_001")
    register_agent_class("small_agent", agent_001)

    agent_core = Core(action_size=6, position_size=16,
        width=64, height=64, channel=4,
        hidden_size=128, heads=8, layers=2,
        device=device
    ).to(device)
    learner = Learner(agent=agent_core, device=device)
    model_53_agent = model_53.Model_53(
        agent_core=agent_core, trainer=learner, 
        context_collector=Collector(max_history=8),
        action_collector=Collector(max_history=8)
    )
    register_agent_class("model_53", model_53_agent)

    from main import main

    # override system argument
    sys.argv = [sys.argv[0], "--agent", "model_53", "--game", "ls20"]

    main()