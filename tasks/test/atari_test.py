import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utilities.package_install import install

install("ale-py")
install("gymnasium[atari, other]")

from utilities.atari.environments import Multi_Atari_Environment

game_ids = ["ALE/Pong-v5", "ALE/AirRaid-v5", "ALE/SpaceInvaders-v5", "ALE/Assault-v5"]


def run(rnd_reward, prepend_objective):
    envs = Multi_Atari_Environment(
        game_ids=game_ids,
        episodic_life=False,        # True ends the episode on every lost life and restarts the game; keep False for hard-exploration games
        reward_clipping=True,
        prepend_objective=prepend_objective,
        rnd_reward=rnd_reward,
    )

    observations, info = envs.reset(seed=0)
    for i, obs in enumerate(observations):
        assert obs.shape == (envs.observation_size,) and obs.dtype == np.uint8, (obs.shape, obs.dtype)
        if prepend_objective:
            assert obs[0] == list(dict.fromkeys(game_ids)).index(game_ids[i]), (i, obs[0])

    # Measure throughput over 1 second of random play
    start_time = time.perf_counter()
    total_steps = 0
    intrinsic = []
    while True:
        actions = envs.sample_valid_actions()
        observations, rewards, terminations, truncations, infos = envs.step(actions)
        intrinsic.extend(info["intrinsic_reward"] for info in infos)
        total_steps += len(actions)
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= 1.0:
            break

    for obs in observations:
        assert obs.shape == (envs.observation_size,) and obs.dtype == np.uint8, (obs.shape, obs.dtype)

    print(f"rnd={rnd_reward} objective={prepend_objective} obs_size={envs.observation_size} "
          f"throughput: {total_steps / elapsed_time:.2f} steps/second, mean intrinsic reward: {np.mean(intrinsic):.4f}")
    envs.close()


run(rnd_reward=False, prepend_objective=False)
run(rnd_reward=True, prepend_objective=True)
