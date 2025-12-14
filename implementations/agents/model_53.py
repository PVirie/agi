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

        self.reset()


    def reset(self):
        self.trainer.reset(time=0.0)
        self.obs.clear()
        self.actions.clear()
        self.logprobs = []
        self.rewards = []
        self.next_dones = []
        self.values = []

        self.current_score = None
        self.thought_steps = None


    def choose_action(self, idles, dones, truncates, latest_frames, scores, next_available_actions):

        batch_size = len(scores)

        if self.current_score is None:
            self.current_score = [0 for _ in scores]
            self.thought_steps = [0 for _ in scores]
            self.obs.append(
                np.zeros((batch_size, 1), dtype=np.float32), 
                np.zeros((batch_size, self.agent_core.position_size), dtype=np.float32), 
                np.zeros((batch_size, self.agent_core.content_size), dtype=np.float32)
            )

        content = extract_frame(latest_frames) # content must be batch leading tensor (batch_size, ...)
        reward = np.array([score - self.current_score[i] for i, score in enumerate(scores)])
        self.current_score = [score for score in scores] # copy
        next_done = [done for done in dones] # copy
        truncated = [truncate for truncate in truncates] # copy

        for i, idle in enumerate(idles):
            if not idle:
                # replace the reward and content
                self.obs.update_last(
                    np.reshape(reward, (-1, 1)), 
                    self.obs.get_last()[:, 1:1 + self.agent_core.position_size],
                    np.reshape(content, (batch_size, -1))
                )

        # learn Supervise content
        if self.do_supervision and len(self.rewards) > 0:
            target_actions, last_mask = self.agent_core.make_batch_actions(
                b_content=np.reshape(content, (batch_size, -1))
            )
            non_idle_mask = np.array([[0.0] if idle else [1.0] for idle in idles], dtype=np.float32)
            target_actions = pad(np.expand_dims(target_actions, axis=1), len(self.rewards), pad_value=0, append_to_front=True)
            supervised_mask = pad(np.expand_dims(last_mask * non_idle_mask, axis=1), len(self.rewards), pad_value=0.0, append_to_front=True)
            self.supervised_trainer.train(
                obs=self.obs[:-1].make_batch(batch_led=True),
                actions=self.actions.make_batch(batch_led=True),
                target_actions=target_actions,
                masks=supervised_mask
            )

        if (any(next_done) or any(truncated) or any([r != 0 for r in reward])) and len(self.rewards) > 0:
            
            # compute last value from the current context (past observation) and the recent observation
            # this one return batch leading tensors (batch, 1)
            last_value = self.agent_core.get_latest_value(
                self.obs.make_batch(batch_led=True),
                self.actions.make_batch(batch_led=True, append_last=True),
            )

            # learn RL
            self.trainer.learn(
                self.obs[:-1].make_batch(batch_led=True), 
                self.actions.make_batch(batch_led=True), 
                self.logprobs, 
                self.rewards, 
                self.values, 
                self.next_dones, 
                np.reshape(last_value, (-1)), next_done,
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


        # Choose a random action
        # this one return batch leading tensors (batch, 1, ...)
        packed_action, position, newlogprob, _, newvalue = self.agent_core.get_action_and_value(
            self.obs.make_batch(batch_led=True),
            self.actions.make_batch(batch_led=True, append_last=True),
            use_action=False,
            use_grad=False
        )

        # store last states
        self.actions.append(packed_action[:, -1, ...])
        self.logprobs.append(newlogprob[:, -1, ...])
        self.rewards.append([0 for _ in range(batch_size)])
        self.next_dones.append([False for _ in range(batch_size)])
        self.values.append(newvalue[:, -1, ...])

        # extract output here
        ext_flag, a, x, y, content = self.agent_core.unpack_action(packed_action[:, -1, ...])

        position = position[:, -1, ...]
        self.obs.append(np.zeros((batch_size, 1), dtype=np.float32), position, content)
        
        # if next done, reset score and thought steps
        action = []
        for i, (d, t) in enumerate(zip(dones, truncates)):
            if d or t:
                self.current_score[i] = 0
                self.thought_steps[i] = 0
            else:
                self.thought_steps[i] += 1

            if ext_flag[i].item() == 1 or self.thought_steps[i] >= 2:
                action.append((a[i].item(), x[i].item(), y[i].item()))
                self.thought_steps[i] = 0
            else:
                action.append(None)

        return action