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
install("gymnasium[atari]")

from ale_py.vector_env import AtariVectorEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a vector environment with 4 parallel instances of Breakout
envs = AtariVectorEnv(
    game="breakout",  # The ROM id not name, i.e., camel case compared to `gymnasium.make` name versions
    num_envs=4,
)

# Reset all environments
observations, info = envs.reset()

# Measure total steps took in 1 second
start_time = time.perf_counter()
total_steps = 0
for _ in range(10000):
    # Take random actions in all environments
    actions = envs.action_space.sample()
    observations, rewards, terminations, truncations, infos = envs.step(actions)
    total_steps += len(actions)
    elapsed_time = time.perf_counter() - start_time
    if elapsed_time >= 1.0:
        break

print(f"Throughput: {total_steps / elapsed_time:.2f} steps/second")

# Close the environment when done
envs.close()