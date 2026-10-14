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
    # mixed batch: every (game, byte) is its own field; each kind is padded to the longest mask of that kind
    mixed_ids = ["ALE/MontezumaRevenge-v5", "ALE/Adventure-v5", "ALE/MontezumaRevenge-v5", "ALE/Pitfall-v5"]
    ram_mask = RAM_Mask(mixed_ids, ram_size=128)
    montezuma, adventure, pitfall = (RND_RAM_MASKS[g] for g in ("MontezumaRevenge", "Adventure", "Pitfall"))
    assert ram_mask.categorical_size == len(adventure["categorical"])
    assert ram_mask.continuous_size == len(adventure["continuous"])
    assert ram_mask.num_categorical_fields == len(montezuma["categorical"]) + len(adventure["categorical"])

    rams = np.random.default_rng(0).integers(0, 256, size=(len(mixed_ids), 128), dtype=np.uint8)
    tokens, values, fields = ram_mask(rams, [0, 1, 2, 3])
    assert tokens.shape == (4, ram_mask.categorical_size) and values.shape == fields.shape == (4, ram_mask.continuous_size)
    for row in (0, 2):
        n = len(montezuma["categorical"])
        # same game, same fields: the token only depends on the byte value
        assert np.array_equal(tokens[row, :n] % 256, rams[row, montezuma["categorical"]])
        assert (tokens[row, n:] == ram_mask.padding_token).all()
        assert np.array_equal(values[row, :len(montezuma["continuous"])], rams[row, montezuma["continuous"]])
    assert np.array_equal(tokens[0] // 256, tokens[2] // 256)
    # different games never share a field
    assert not set(tokens[0, :len(montezuma["categorical"])] // 256) & set(tokens[1] // 256)
    assert not set(fields[0]) & set(fields[1])
    assert (tokens[3] == ram_mask.padding_token).all() and (fields[3] >= 0).sum() == len(pitfall["continuous"])

    # unlisted games read the full RAM as continuous
    full = RAM_Mask(game_ids[:1], ram_size=128)
    assert full.categorical_size == 0 and full.continuous_size == 128
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
