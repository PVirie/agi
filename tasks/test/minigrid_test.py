import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

from utilities.package_install import install

install("minigrid")
install("gymnasium[other]")
install("pillow")

from PIL import Image

from utilities.minigrid.environments import Multi_Environment
from utilities.minigrid import constants
from utilities.tokenizer import Text_Tokenizer

tokenizer = Text_Tokenizer(max_vocab_size=100)

# Create a vector environment with 4 parallel instances of Breakout
envs = Multi_Environment(
    game_ids=["BabyAI-Unlock-v0", "BabyAI-GoToSeqS5R2-v0", "MiniGrid-FourRooms-v0", "MiniGrid-ObstructedMaze-Full-v0"],
    mission_max_len=12,
    full_mdp=True,
    full_mdp_width=22,
    full_mdp_height=22,
    tokenizer=tokenizer
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
    mission = obs[:envs.mission_max_len]
    # decode mission
    mission_text = tokenizer.decode(mission)
    print(f"Environment {i} Mission: {mission_text}")

    frame = obs[envs.mission_max_len:].reshape((3, envs.full_mdp_height, envs.full_mdp_width)).astype(np.uint8)
    # transpose to channel last
    frame = frame.transpose(1, 2, 0)  # shape (full_mdp_height, full_mdp_width, 3)
    
    # convert frame to color using IDX_TO_COLOR and gather
    color_indices = frame[..., 1]
    # map color indices to RGB using constants.IDX_TO_COLOR
    color_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    for color_idx, color in constants.IDX_TO_COLOR.items():
        color_frame[color_indices == color_idx] = constants.COLORS[color]
    frame = color_frame

    for j in range(frame.shape[0]):
        img = Image.fromarray(frame, mode='RGB')
        # resize image four times
        img = img.resize((img.width * 4, img.height * 4), resample=Image.NEAREST)
        img.save(f"{artifacts_path}/env_{i}_{j}.png")

# Close the environment when done
envs.close()