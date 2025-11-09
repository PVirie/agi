import os
import sys
import subprocess
import logging
import random

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

import utilities

utilities.install('argparse')

arcagi_path = f"{APP_ROOT}/cache/ARC-AGI-3-Agents"
os.makedirs(arcagi_path, exist_ok=True)
if len(os.listdir(arcagi_path)) == 0:
    subprocess.run(["git", "clone", "https://github.com/arcprize/ARC-AGI-3-Agents.git", f"{arcagi_path}"])
    utilities.install_pyproject_toml(f"{arcagi_path}/pyproject.toml")
    utilities.install("langchain-openai")

sys.path.append(os.path.join(arcagi_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from implementations.agents import my_awesome_agent

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from main import main

    # override system argument
    sys.argv = [sys.argv[0], "--agent", "MyAwesomeAgent", "--game", "ls20"]

    main()