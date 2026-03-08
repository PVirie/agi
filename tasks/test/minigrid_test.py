import os
import sys
import time
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

from utilities.package_install import install

install("minigrid")
install("gymnasium[other]")

from utilities.minigrid.environments import Multi_Environment

# Create a vector environment with 4 parallel instances of Breakout
envs = Multi_Environment(
    game_ids=["BabyAI-GoToRedBall-v0", "BabyAI-GoToSeqS5R2-v0", "MiniGrid-SimpleCrossingS11N5-v0", "MiniGrid-GoToDoor-8x8-v0"]
)

# Reset all environments
observations, info = envs.reset()

# Measure total steps took in 1 second
start_time = time.perf_counter()
total_steps = 0
for _ in range(10000):
    # Take random actions in all environments
    actions = envs.sample_valid_actions()
    observations, rewards, terminations, truncations, infos = envs.step(actions)
    total_steps += len(actions)
    elapsed_time = time.perf_counter() - start_time
    if elapsed_time >= 1.0:
        break

print(f"Throughput: {total_steps / elapsed_time:.2f} steps/second")

# now draw one frame from each environment and save as images to /artifacts/log
artifacts_path = f"{APP_ROOT}/log"
os.makedirs(artifacts_path, exist_ok=True)
for i, obs in enumerate(observations):
    direction = obs['direction']
    frame = obs['image']
    mission = obs['mission']
    for j in range(frame.shape[0]):
        img = Image.fromarray(frame[j, ...], mode='L')  # 'L' mode for grayscale
        img.save(f"{artifacts_path}/env_{i}_{j}.png")

# Close the environment when done
envs.close()