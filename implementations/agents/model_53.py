from agents.structs import FrameData, GameAction, GameState # Make sure to change from `..` imports
from .decorator import Instantiable_Agent
import logging
import random

from interfaces.learning import Learner
from interfaces.core import Core, Context_Collector

from .utils import extract_frame


class Model_53(Instantiable_Agent):
    
    def __init__(self, agent_core: Core, trainer: Learner, context_collector: Context_Collector):
        self.agent_core = agent_core
        self.trainer = trainer
        self.obs = context_collector

        self.trainer.reset(time=0.0)
        self.obs.clear()
        self.packed_actions = []
        self.logprobs = []
        self.rewards = []
        self.next_dones = []
        self.values = []

        self.current_score = 0

        self.last_position = None


    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Your logic to determine if the game is finished
        return latest_frame.state is GameState.WIN


    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:

        game_state, obs, score, next_done = extract_frame(latest_frame)
        reward = score - self.current_score
        self.current_score = score

        self.obs.append(obs, self.last_position, reward)

        # compute last value from the current context (past observation) and the recent observation
        last_value = self.agent_core.get_latest_value(self.obs)

        if self.last_position is not None:
            # learn RL and Supervised content
            self.trainer.learn(self.obs[:-1], self.packed_actions, self.logprobs, self.rewards, self.values, self.next_dones, last_value, next_done)
            
            self.trainer.reset(time=0.0)
            self.obs.reset()
            self.packed_actions = []
            self.logprobs = []
            self.rewards = []
            self.next_dones = []
            self.values = []

        if game_state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = f"Game is over or not played, choosing RESET"
            return action
    
        while True:
            # Choose a random action (except RESET)
            packed_action, newlogprob, _, newvalue = self.agent_core.get_action_and_value(self.obs[:-1])

            self.packed_actions.append(packed_action)
            self.logprobs.append(newlogprob)
            self.rewards.append(0)
            self.next_dones.append(False)
            self.values.append(newvalue)

            # extract output here
            ext_flag, action_data, position, content = unpack_action(packed_action)

            if ext_flag:
                self.last_position = position
                break
            else:
                self.obs.append(content, position, 0)

        action = GameAction(action_data)

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