from agents.structs import FrameData, GameAction, GameState # Make sure to change from `..` imports
from .decorator import Instantiable_Agent
import logging
import random

from interfaces.learning import Learner
from interfaces.agent import Agent_Core


class Model_53(Instantiable_Agent):
    
    def __init__(self, agent_core: Agent_Core, trainer: Learner):
        self.agent_core = agent_core
        self.trainer = trainer

        self.trainer.reset(time=0.0)

        self.last_obs = None
        self.last_value = None
        self.last_action = None
        self.last_action_log_prob = None


    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Your logic to determine if the game is finished
        return latest_frame.state is GameState.WIN


    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:

        # now collect
        self.trainer.collect(self.last_obs, self.last_value, self.last_action, self.last_action_log_prob, reward, 0, 0)

        # learn RL and Supervised content
        last_values = self.agent_core.get_value(frames_tensor)
        last_value = last_values[:, -1]
        self.trainer.learn(lastest_value, 1.0, 0)
        self.trainer.reset(time=0.0)

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = f"Game is over or not played, choosing RESET"
            return action
    
        while True:
            # Choose a random action (except RESET)
            ext_flag, action, content, value, log_prob, entropy = self.agent_core.get_action_and_value(frames_tensor)

            self.last_obs = frames_tensor
            self.last_value = value
            self.last_action = action
            self.last_action_log_prob = log_prob

            if ext_flag:
                break

            self.trainer.collect(obs, value, action, logprob, 0, 0, 0)

    
        # Add reasoning for simple actions
        if action.is_simple():
            action.reasoning = f"Chose {action.value} randomly"
        # For complex actions, set coordinates
        elif action.is_complex():
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })
            action.reasoning = {"action": action.value, "reason": "Random choice"}
        
        return action