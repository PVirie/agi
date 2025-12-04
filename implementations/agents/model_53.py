from agents.structs import FrameData, GameAction, GameState # Make sure to change from `..` imports
from .decorator import Instantiable_Agent
import logging
import random
import numpy as np

from interfaces.learning import Learner
from interfaces.core import Core, Context_Collector

from .utils import extract_frame


game_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]

class Model_53(Instantiable_Agent):
    
    def __init__(self, agent_core: Core, trainer: Learner, context_collector: Context_Collector, action_collector: Context_Collector):
        self.agent_core = agent_core
        self.trainer = trainer
        self.obs = context_collector
        self.actions = action_collector

        self.trainer.reset(time=0.0)
        self.obs.clear()
        self.actions.clear()
        self.logprobs = []
        self.rewards = []
        self.next_dones = []
        self.values = []

        self.current_score = 0

        self.last_position = None
        self.last_content = None


    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Your logic to determine if the game is finished
        return latest_frame.state is GameState.WIN


    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:

        game_state, content_, score = extract_frame(latest_frame)
        reward = score - self.current_score
        self.current_score = score
        next_done = game_state in [GameState.GAME_OVER, GameState.WIN]

        # content must be batch leading tensor (1, ...)
        last_position = self.last_position
        if last_position is None:
            last_position = np.zeros((1, 16), dtype=np.float32)
        self.obs.append(np.array([[reward]], dtype=np.float32), last_position, np.reshape(content_, (1, -1)))

        if self.last_position is not None:

            # compute last value from the current context (past observation) and the recent observation
            # this one return batch leading tensors (batch)
            last_value = self.agent_core.get_latest_value(
                self.obs.make_batch(batch_led=True),
                self.actions.make_batch(batch_led=True, append_last=True),
            )
            
            # learn RL and Supervised content
            self.trainer.learn(
                self.obs[:-1].make_batch(batch_led=True), 
                self.actions.make_batch(batch_led=True), 
                self.logprobs, 
                self.rewards, 
                self.values, 
                self.next_dones, 
                last_value, [next_done],
                masks=self.actions.make_mask(batch_led=True)
            )
            
            self.trainer.reset(time=0.0)
            self.obs.mark(skip_last=True)
            self.actions.mark()
            # self.logprobs = []
            # self.rewards = []
            # self.next_dones = []
            # self.values = []

        if game_state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = f"Game is over or not played, choosing RESET"
            self.obs.clear()
            self.actions.clear()
            self.last_position = None
            self.last_content = None
            return action
    
        thought_steps = 1
        while True:
            # Choose a random action (except RESET)
            # this one return batch leading tensors (batch, 1, ...)
            packed_action, position, newlogprob, _, newvalue = self.agent_core.get_action_and_value(
                self.obs.make_batch(batch_led=True),
                self.actions.make_batch(batch_led=True, append_last=True),
                use_action=False
            )

            position = position[:, -1, ...]
            self.actions.append(packed_action[:, -1, ...])
            self.logprobs.append(newlogprob[:, -1, ...])
            self.rewards.append([0])
            self.next_dones.append([False])
            self.values.append(newvalue[:, -1, ...])

            # extract output here
            ext_flag, a, x, y, content = self.agent_core.unpack_action(packed_action[:, -1, ...])

            self.last_position = position
            self.last_content = content
        
            # Decide whether to execute action or think more
            if ext_flag.item() > 0.5 or thought_steps >= 4:
                break

            self.obs.append(np.zeros((1, 1), dtype=np.float32), position, content)
            thought_steps += 1


        action = game_actions[a.item()]

        # Add reasoning for simple actions
        if action.is_simple():
            # action.reasoning = f"Chose action {action.name}."
            pass
        # For complex actions, set coordinates
        elif action.is_complex():
            action.set_data({
                "x": x.item(),
                "y": y.item()
            })
        
        return action