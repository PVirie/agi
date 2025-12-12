import numpy as np
from interfaces.learning import PPO_Learner, Supervised_Learner
from interfaces.core import Core, Context_Collector

from .utils import extract_frame, pad

class Model_53:
    
    def __init__(self, 
                 agent_core: Core, 
                 trainer: PPO_Learner, supervised_trainer: Supervised_Learner, 
                 context_collector: Context_Collector, action_collector: Context_Collector,
                 do_supervision: bool = False
                 ):
        self.agent_core = agent_core
        self.trainer = trainer
        self.supervised_trainer = supervised_trainer
        self.obs = context_collector
        self.actions = action_collector

        self.do_supervision = do_supervision

        self.trainer.reset(time=0.0)
        self.obs.clear()
        self.actions.clear()
        self.logprobs = []
        self.rewards = []
        self.next_dones = []
        self.values = []

        self.current_score = 0

        self.last_position = None


    def choose_action(self, latest_frames, states, scores, next_available_actions):

        game_state, content_, score = extract_frame(latest_frame)
        reward = score - self.current_score
        self.current_score = score
        next_done = game_state in [GameState.GAME_OVER, GameState.WIN]

        # content must be batch leading tensor (1, ...)
        last_position = self.last_position
        if last_position is None:
            last_position = np.zeros((1, self.agent_core.position_size), dtype=np.float32)
        self.obs.append(np.array([[reward]], dtype=np.float32), last_position, np.reshape(content_, (1, -1)))

        if reward != 0 or len(self.rewards) % 10 == 9 or next_done:

            # compute last value from the current context (past observation) and the recent observation
            # this one return batch leading tensors (batch)
            last_value = self.agent_core.get_latest_value(
                self.obs.make_batch(batch_led=True),
                self.actions.make_batch(batch_led=True, append_last=True),
            )

            # learn Supervise content
            if self.do_supervision:
                target_actions, last_mask = self.agent_core.make_batch_actions(
                    b_content=np.reshape(content_, (1, -1))
                )
                target_actions = pad(np.expand_dims(target_actions, axis=1), len(self.rewards), pad_value=0, append_to_front=True)
                last_mask = pad(np.expand_dims(last_mask, axis=1), len(self.rewards), pad_value=0.0, append_to_front=True)
                self.supervised_trainer.train(
                    obs=self.obs[:-1].make_batch(batch_led=True),
                    actions=self.actions.make_batch(batch_led=True),
                    target_actions=target_actions,
                    masks=last_mask
                )
            
            # learn RL
            self.trainer.learn(
                self.obs[:-1].make_batch(batch_led=True), 
                self.actions.make_batch(batch_led=True), 
                self.logprobs, 
                self.rewards, 
                self.values, 
                self.next_dones, 
                np.reshape(last_value, (-1)), [next_done],
                masks=self.actions.make_mask(batch_led=True)
            )

            self.supervised_trainer.save()
            self.trainer.save()
            self.agent_core.save()
            
            self.trainer.reset(time=0.0)
            self.obs.mark(skip_last=True)
            left_over_slide = self.actions.mark()
            self.logprobs = self.logprobs[left_over_slide]
            self.rewards = self.rewards[left_over_slide]
            self.next_dones = self.next_dones[left_over_slide]
            self.values = self.values[left_over_slide]

        if game_state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = f"Game is over or not played, choosing RESET"
            self.obs.clear()
            self.actions.clear()
            self.last_position = None
            return action
    
        thought_steps = 1
        while True:
            # Choose a random action (except RESET)
            # this one return batch leading tensors (batch, 1, ...)
            packed_action, position, newlogprob, _, newvalue = self.agent_core.get_action_and_value(
                self.obs.make_batch(batch_led=True),
                self.actions.make_batch(batch_led=True, append_last=True),
                use_action=False,
                use_grad=False
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
        
            # Decide whether to execute action or think more
            if ext_flag.item() > 0.5 or thought_steps >= 2:
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