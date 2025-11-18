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

from implementations.agents import register_agent_class, my_awesome_agent

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

    agent_001 = my_awesome_agent.MyAwesomeAgent("agent_001")
    register_agent_class("small_agent", agent_001)

    from main import main

    # override system argument
    sys.argv = [sys.argv[0], "--agent", "small_agent", "--game", "ls20"]

    main()