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
        self.values = []
        self.rewards = []
        self.next_dones = []
        self.last_truncates = []
        self.last_idles = []

        self.current_score = None
        self.thought_steps = None


    def choose_action(self, last_idles, next_dones, last_truncates, latest_frames, scores, next_available_actions, force_train=False):

        batch_size = len(scores)
        current_cl = len(self.rewards)

        if self.current_score is None:
            self.current_score = [0 for _ in scores]
            self.thought_steps = [0 for _ in scores]
            self.obs.append(
                np.zeros((batch_size, 1), dtype=np.float32), 
                np.zeros((batch_size, self.agent_core.position_size), dtype=np.float32), 
                np.zeros((batch_size, self.agent_core.content_size), dtype=np.float32)
            )
            self.last_truncates.append([True for _ in range(batch_size)])
            self.last_idles.append([False for _ in range(batch_size)])

        for i, t in enumerate(last_truncates):
            if t:
                self.current_score[i] = 0
                self.thought_steps[i] = 0

        content = extract_frame(latest_frames) # content must be batch leading tensor (batch_size, ...)
        reward = np.array([score - self.current_score[i] for i, score in enumerate(scores)])
        self.current_score = [s for s in scores] # copy
        next_done = [d for d in next_dones] # copy
        last_truncated = [t for t in last_truncates] # copy
        last_idle = [idle for idle in last_idles] # copy

        for i, (idle, t, idle) in enumerate(zip(last_idles, last_truncates, last_idles)):
            if not idle:
                # replace the reward and content
                self.obs.update_last(
                    np.reshape(reward, (-1, 1)), 
                    self.obs.get_last()[:, 1:1 + self.agent_core.position_size],
                    np.reshape(content, (batch_size, -1))
                )

            self.last_truncates[-1][i] = t
            self.last_idles[-1][i] = idle

        # learn Supervise content
        if self.do_supervision and current_cl > 0:
            target_actions, last_mask = self.agent_core.make_batch_actions(
                b_content=np.reshape(content, (batch_size, -1))
            )
            non_idle_mask = np.array([[0.0] if idle else [1.0] for idle in last_idles], dtype=np.float32)
            target_actions = pad(np.expand_dims(target_actions, axis=1), current_cl, pad_value=0, append_to_front=True)
            supervised_mask = pad(np.expand_dims(last_mask * non_idle_mask, axis=1), current_cl, pad_value=0.0, append_to_front=True)
            self.supervised_trainer.train(
                obs=self.obs[:-1].make_batch(batch_led=True),
                actions=self.actions.make_batch(batch_led=True),
                target_actions=target_actions,
                masks=supervised_mask
            )

        if (any(next_done) or any(last_truncated) or any([r != 0 for r in reward]) or force_train) and current_cl > 1:
            
            # compute last value from the current context (past observation) and the recent observation
            # this one return batch leading tensors (batch, 1)
            last_value = self.agent_core.get_latest_value(
                self.obs.make_batch(batch_led=True),
                self.actions.make_batch(batch_led=True, append_last=True),
            )

            # masks has shape (batch_size, context_length)
            masks = self.actions.make_mask(batch_led=True)
            # need to shift last_truncates by 1 to the left, because t signals whether t-1 is truncated
            masks = masks * (1.0 - np.stack(self.last_truncates[1:], axis=1, dtype=np.float32))

            # learn RL
            self.trainer.learn(
                obs=self.obs[:-1].make_batch(batch_led=True), 
                actions=self.actions.make_batch(batch_led=True), 
                logprobs=self.logprobs, 
                values=self.values, 
                rewards=self.rewards, 
                next_dones=self.next_dones, 
                last_value=np.reshape(last_value, (-1)), 
                last_done=next_done,
                masks=masks
            )

            self.supervised_trainer.save()
            self.trainer.save()
            self.agent_core.save()
            
            self.trainer.reset(time=0.0)

            left_over_slide = self.actions.mark()
            self.logprobs = self.logprobs[left_over_slide]
            self.values = self.values[left_over_slide]
            self.rewards = self.rewards[left_over_slide]
            self.next_dones = self.next_dones[left_over_slide]

            left_over_slide = self.obs.mark(skip_last=True)
            self.last_truncates = self.last_truncates[left_over_slide]
            self.last_idles = self.last_idles[left_over_slide]


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
        self.values.append(newvalue[:, -1, ...])
        self.rewards.append(reward)
        self.next_dones.append(next_done)

        # extract output here
        ext_flag, a, x, y, content = self.agent_core.unpack_action(packed_action[:, -1, ...])

        position = position[:, -1, ...]
        self.obs.append(np.zeros((batch_size, 1), dtype=np.float32), position, content)
        self.last_truncates.append([False for _ in range(batch_size)])
        self.last_idles.append([ext_flag[i].item() == 0 for i in range(batch_size)])
        
        # if next done, reset score and thought steps
        action = []
        for i, d in enumerate(next_dones):
            if d:
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