import re

# Compact BabyAI mission language. Every phrase collapses into a single token so missions
# fit in a short mission_max_len while the vocabulary stays far below 256 words.
#
#   go to the red box on your left, then pick up a ball
#   -> gth r-box on-left then pa ball
#
# Action + determiner: <verb><th|a>
#   go to -> g, pick up -> p, open -> o, put -> m (move), next to -> n
# Object: <color>-<type>, or just <type> when no color is given
#   red r, green g, blue b, purple p, yellow y, grey gy; "object" -> obj
# Location: in-front, behind, on-left, on-right
# Connectives: then, after, and (commas are dropped)

VERBS = {"go to": "g", "pick up": "p", "open": "o", "put": "m", "next to": "n"}
DETERMINERS = {"the": "th", "a": "a"}
COLORS = {"red": "r", "green": "g", "blue": "b", "purple": "p", "yellow": "y", "grey": "gy"}
TYPES = {"box": "box", "ball": "ball", "key": "key", "door": "door", "object": "obj"}
LOCATIONS = {"in front of you": "in-front", "behind you": "behind", "on your left": "on-left", "on your right": "on-right"}
CONNECTIVES = {", then": " then", " after you": " after"}

_verb_re = re.compile(r"\b(" + "|".join(VERBS) + r") (" + "|".join(DETERMINERS) + r")\b")
_obj_re = re.compile(r"\b(?:(" + "|".join(COLORS) + r") )?(" + "|".join(TYPES) + r")\b")
_loc_re = re.compile(r"\b(" + "|".join(LOCATIONS) + r")\b")


def shorten_mission(mission):
    for long, short in CONNECTIVES.items():
        mission = mission.replace(long, short)
    mission = mission.replace(",", "")
    mission = _loc_re.sub(lambda m: LOCATIONS[m.group(1)], mission)
    mission = _verb_re.sub(lambda m: VERBS[m.group(1)] + DETERMINERS[m.group(2)], mission)
    mission = _obj_re.sub(lambda m: (COLORS[m.group(1)] + "-" if m.group(1) else "") + TYPES[m.group(2)], mission)
    return " ".join(mission.split())


def compact_vocabulary():
    words = [v + d for v in VERBS.values() for d in DETERMINERS.values()]
    words += [t for t in TYPES.values()] + [c + "-" + t for c in COLORS.values() for t in TYPES.values()]
    words += list(LOCATIONS.values()) + ["then", "after", "and"]
    return words


_expand_table = {}
_expand_table.update({v + d: lv + " " + ld for lv, v in VERBS.items() for ld, d in DETERMINERS.items()})
_expand_table.update({t: lt for lt, t in TYPES.items()})
_expand_table.update({c + "-" + t: lc + " " + lt for lc, c in COLORS.items() for lt, t in TYPES.items()})
_expand_table.update({s: l for l, s in LOCATIONS.items()})
_expand_table.update({s.strip(): l.strip() for l, s in CONNECTIVES.items()})


def expand_mission(short_mission):
    # Inverse of shorten_mission; unknown tokens pass through unchanged.
    mission = " ".join(_expand_table.get(w, w) for w in short_mission.split())
    return mission.replace(" , then", ", then")


if __name__ == "__main__":
    import io
    import sys
    import contextlib
    import gymnasium as gym
    import minigrid  # noqa: F401  (registers BabyAI envs)

    seeds_per_env = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    vocabulary = set(compact_vocabulary())
    env_ids = sorted(e for e in gym.registry if e.startswith("BabyAI-"))

    checked, failures, oov = 0, [], set()
    seen = set()
    for env_id in env_ids:
        env = gym.make(env_id)
        for seed in range(seeds_per_env):
            with contextlib.redirect_stdout(io.StringIO()):
                obs, _ = env.reset(seed=seed)
            mission = obs["mission"]
            if mission in seen:
                continue
            seen.add(mission)
            short = shorten_mission(mission)
            restored = expand_mission(short)
            checked += 1
            oov.update(w for w in short.split() if w not in vocabulary)
            if restored != mission:
                failures.append((env_id, mission, short, restored))
        env.close()

    print(f"\nchecked {checked} unique missions from {len(env_ids)} envs")
    print(f"out-of-vocabulary short tokens: {sorted(oov) if oov else 'none'}")
    for env_id, mission, short, restored in failures[:20]:
        print(f"[FAIL] {env_id}\n  original: {mission}\n  short:    {short}\n  restored: {restored}")
    print(f"round-trip failures: {len(failures)}")
    sys.exit(1 if failures else 0)
