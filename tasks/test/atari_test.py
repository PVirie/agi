import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utilities.package_install import install

install("ale-py")
install("gymnasium[atari, other]")

from utilities.atari.environments import Multi_Atari_Environment
from utilities.atari.rnd import RAM_Mask, RND_RAM_MASKS

game_ids = ["ALE/Pong-v5", "ALE/AirRaid-v5", "ALE/SpaceInvaders-v5", "ALE/Assault-v5"]


def test_rnd_ram_mask():
    # mixed batch: the masked games read only their selected bytes, zero-padded to the longest mask
    mixed_ids = ["ALE/MontezumaRevenge-v5", "ALE/Adventure-v5", "ALE/MontezumaRevenge-v5"]
    ram_mask = RAM_Mask(mixed_ids, ram_size=128)
    assert ram_mask.feature_size == len(RND_RAM_MASKS["Adventure"])

    rams = np.random.default_rng(0).integers(0, 256, size=(len(mixed_ids), 128), dtype=np.uint8)
    features = ram_mask(rams, [0, 1, 2])
    assert features.shape == (3, ram_mask.feature_size) and features.dtype == np.uint8
    montezuma = RND_RAM_MASKS["MontezumaRevenge"]
    for row in (0, 2):
        assert np.array_equal(features[row, :len(montezuma)], rams[row, montezuma])
        assert not features[row, len(montezuma):].any()
    assert np.array_equal(features[1], rams[1, RND_RAM_MASKS["Adventure"]])

    # unlisted games keep the full RAM
    assert RAM_Mask(game_ids[:1], ram_size=128).feature_size == 128
    print("rnd ram mask: ok")


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


test_rnd_ram_mask()
run(rnd_reward=False, prepend_objective=False)
run(rnd_reward=True, prepend_objective=True)
