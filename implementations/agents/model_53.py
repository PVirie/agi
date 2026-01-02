import numpy as np
from interfaces.learning import PPO_Learner, Supervised_Learner
from interfaces.core import Core
from interfaces.memory import Memory, Memory_Operation_Type
from interfaces.data_structure import Context_Collector


class Model_53:
    
    def __init__(self, 
                 agent_core: Core, 
                 trainer: PPO_Learner, supervised_trainer: Supervised_Learner, 
                 context_collector: Context_Collector, action_collector: Context_Collector,
                 memory: Memory,
                 max_num_thought_steps: int = 2,
                 do_supervision: bool = False,
                 use_memory: bool = True
                 ):
        self.agent_core = agent_core
        self.trainer = trainer
        self.supervised_trainer = supervised_trainer
        self.obs = context_collector
        self.actions = action_collector
        self.memory = memory

        self.max_num_thought_steps = max_num_thought_steps
        self.do_supervision = do_supervision
        self.use_memory = use_memory

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

        self.agent_core.eval()

        self.thought_steps = None


    def choose_action(self, last_idles, next_dones, last_truncates, last_resets, latest_frames, rewards, next_available_actions, force_train=False):

        batch_size = len(latest_frames)
        current_cl = len(self.rewards)

        if self.thought_steps is None:
            self.thought_steps = [0 for _ in range(batch_size)]
            self.obs.append(
                np.zeros((batch_size, 1), dtype=np.float32), 
                np.zeros((batch_size, self.agent_core.position_size), dtype=np.float32), 
                np.zeros((batch_size, self.agent_core.content_size), dtype=np.float32)
            )
            self.last_truncates.append([True for _ in range(batch_size)])
            self.last_idles.append([False for _ in range(batch_size)])

        for i, (idle, t) in enumerate(zip(last_idles, last_truncates)):
            if t:
                self.thought_steps[i] = 0

        content = np.reshape(np.stack(latest_frames, axis=0), (batch_size, -1)) # content must be batch leading tensor (batch_size, ...)
        reward = np.array([r for r in rewards])
        next_done = [d for d in next_dones] # copy
        last_truncated = [t for t in last_truncates] # copy
        last_idle = [idle for idle in last_idles] # copy

        update_mask = np.zeros((batch_size, 1 + self.agent_core.position_size + self.agent_core.content_size), dtype=np.float32)
        memory_action = [Memory_Operation_Type.IDLE for _ in range(batch_size)]
        for i, (idle, t, r) in enumerate(zip(last_idles, last_truncates, last_resets)):
            if r:
                memory_action[i] = Memory_Operation_Type.RESET
            if not idle:
                # update only reward and content
                update_mask[i, 0] = 1.0
                update_mask[i, 1 + self.agent_core.position_size:] = 1.0
                memory_action[i] = Memory_Operation_Type.CACHE
            self.last_truncates[-1][i] = t
            self.last_idles[-1][i] = idle

        # replace the reward and content
        last_obs = self.obs.get_last()
        new_value = np.concatenate([
            np.reshape(reward, (-1, 1)), 
            np.zeros((batch_size, self.agent_core.position_size), dtype=np.float32), 
            content
        ], axis=1)
        update_value = last_obs * (1.0 - update_mask) + new_value * update_mask
        self.obs.update_last(update_value)

        # cache new observation into memory
        position = last_obs[:, 1:1 + self.agent_core.position_size]
        self.memory.operate(
            tuple_record=(
                np.reshape(reward, (-1, 1)), 
                position, 
                content
            ), 
            operation=memory_action
        )

        if (any(next_done) or any(last_truncated) or force_train) and current_cl > 1:
            
            self.agent_core.train()

            if self.do_supervision:
                # learn Supervise content

                # make action format
                recorded_actions = self.actions.make_batch(batch_led=True)
                last_actions = self.agent_core.pack_action(b_content=content)
                target_actions = np.concatenate([
                    recorded_actions[:, 1:, :],
                    np.reshape(last_actions, (batch_size, 1, -1))
                ], axis=1)
                
                # masks has shape (batch_size, context_length)
                masks = self.actions.make_mask(batch_led=True)
                masks = masks * (1.0 - np.stack(self.last_idles[1:], axis=1, dtype=np.float32))
                # make feature mask of shape (batch_size, context_length, 5)
                # and filter only content part
                masks = np.expand_dims(masks, axis=-1)
                masks = np.concatenate([
                    np.zeros((batch_size, masks.shape[1], 4), dtype=np.float32),
                    masks
                ], axis=-1)

                self.supervised_trainer.train(
                    obs=self.obs[:-1].make_batch(batch_led=True),
                    actions=recorded_actions,
                    target_actions=target_actions,
                    masks=masks
                )

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

            self.agent_core.eval()


        # Choose a random action
        # this one return batch leading tensors (batch, 1, ...)
        packed_action, position, newlogprob, _, newvalue = self.agent_core.get_action_and_value(
            self.obs.make_batch(batch_led=True),
            self.actions.make_batch(batch_led=True, append_last=True),
            use_action=False,
            use_grad=False,
            extra_params={
                "available_actions": next_available_actions
            }
        )

        # store last states
        self.actions.append(packed_action[:, -1, ...])
        self.logprobs.append(newlogprob[:, -1, ...])
        self.values.append(newvalue[:, -1, ...])
        self.rewards.append(reward)
        self.next_dones.append(next_done)

        # extract output here
        int_action, ext_action, content = self.agent_core.unpack_action(packed_action[:, -1, ...])
        position = position[:, -1, ...]

        # if next done, reset score and thought steps
        return_action = [None for _ in range(batch_size)]
        memory_action = [Memory_Operation_Type.IDLE for _ in range(batch_size)]
        memory_fetch_index = [-1 for _ in range(batch_size)]
        for i, d in enumerate(next_dones):
            flag = int_action[i].item()
            self.thought_steps[i] += 1
            if flag == 0 or self.thought_steps[i] >= self.max_num_thought_steps:
                # observe external
                return_action[i] = ext_action[i]
                self.thought_steps[i] = 0
                memory_action[i] = Memory_Operation_Type.IDLE
            else:
                if self.use_memory:
                    if flag == 1:
                        memory_action[i] = Memory_Operation_Type.IDLE
                    elif flag == 2:
                        # position based retrieve
                        memory_action[i] = Memory_Operation_Type.FETCH
                        memory_fetch_index[i] = 1
                    elif flag == 3:
                        # content based retrieve
                        memory_action[i] = Memory_Operation_Type.FETCH
                        memory_fetch_index[i] = 2
                    elif flag == 4:
                        # record node
                        memory_action[i] = Memory_Operation_Type.CACHE
                else:
                    memory_action[i] = Memory_Operation_Type.IDLE

            if d:
                self.thought_steps[i] = 0

        reward, position, content = self.memory.operate(
            tuple_record=(
                np.zeros((batch_size, 1), dtype=np.float32),
                position, 
                content
            ),
            operation=memory_action,
            index=memory_fetch_index
        )

        self.obs.append(reward, position, content)
        self.last_truncates.append([False for _ in range(batch_size)])
        self.last_idles.append([return_action[i] is None for i in range(batch_size)])
        
        return return_action