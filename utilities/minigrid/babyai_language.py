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
