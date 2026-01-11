import os
import sys
import time
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

from utilities.package_install import install

install("ale-py")
install("gymnasium[atari]")

from ale_py.vector_env import AtariVectorEnv

# Create a vector environment with 4 parallel instances of Breakout
envs = AtariVectorEnv(
    game="pong",  # The ROM id not name, i.e., camel case compared to `gymnasium.make` name versions
    num_envs=4,
    img_height=64,           # Height to resize frames to
    img_width=64,            # Width to resize frames to
    maxpool=True,               # 1. Solves "Invisibility" (Flickering)
    stack_num=4,                # 2. Solves "Motion" (Velocity)
    frameskip=4,                # 3. Standard time resolution
    grayscale=True,             # 4. Removes noise (Color is usually irrelevant in Atari)
    episodic_life=True,         # Recommended for harder games (Breakout/Montezuma)
    reward_clipping=True,
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

# now draw one frame from each environment and save as images to /artifacts/log
artifacts_path = f"{APP_ROOT}/log"
os.makedirs(artifacts_path, exist_ok=True)
for i, frame in enumerate(observations):
    img = Image.fromarray(frame[-1, ...], mode='L')  # 'L' mode for grayscale
    img.save(f"{artifacts_path}/pong_env_{i}.png")

# Close the environment when done
envs.close()