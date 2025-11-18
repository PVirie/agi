from agents import AVAILABLE_AGENTS


def register_agent_class(name, instantiable_object):
    AVAILABLE_AGENTS[name] = instantiable_object