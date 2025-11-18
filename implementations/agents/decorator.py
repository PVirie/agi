from agents import AVAILABLE_AGENTS, Agent # Make sure to change from `..` imports
from agents.structs import FrameData, GameAction, GameState # Make sure to change from `..` imports
from functools import wraps


class Agent_Implemented(Agent):

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return False  # Placeholder, will be overridden

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        return GameAction.RESET  # Placeholder, will be overridden


class Instantiable_Agent:

    def __call__(self, *args, **kwargs):
        agent_obj = Agent_Implemented(*args, **kwargs)

        # now assign agent_obj with this subclass methods and parameters
        for attr_name in dir(self):
            if not attr_name.startswith("__"):
                attr_value = getattr(self, attr_name)
                setattr(agent_obj, attr_name, attr_value)

        return agent_obj
    
    def load_parameters(self, path: str):
        raise NotImplementedError("load_parameters method must be implemented by the subclass")
    
    def save_parameters(self, path: str):
        raise NotImplementedError("save_parameters method must be implemented by the subclass")