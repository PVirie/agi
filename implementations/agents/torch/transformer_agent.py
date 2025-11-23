from interfaces.agent import Agent as Agent_Interface
from agents.structs import FrameData, GameAction, GameState # Make sure to change from `..` imports
from ..decorator import Instantiable_Agent
import random
import logging

from implementations.rl_algorithms.torch.ppo import PPO


class Transformer_Agent(Instantiable_Agent):
    def __init__(self, trainer: PPO):
        self.trainer = trainer
    
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Your logic to determine if the game is finished
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:

        # Your custom decision-making logic goes here
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Start or restart the game
            action = GameAction.RESET
        else:
            # Choose a random action (except RESET)
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])
        
        # decide to learn or not
        # convert frames to tensors
        


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