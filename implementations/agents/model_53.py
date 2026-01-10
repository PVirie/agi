import numpy as np
from interfaces.learning import RL_Learner, Supervised_Learner
from interfaces.agent import Agent
from interfaces.network import Policy_Network, Value_Network
from interfaces.memory import Memory, Memory_Operation_Type
from interfaces.data_structure import Context_Collector


def make_valid_mask(valid_actions, action_size):
    """
    valid_actions: a list (batch) of list of int
    action_size: int
    """
    valid_mask = np.zeros((len(valid_actions), action_size), dtype=np.bool)
    for i, va in enumerate(valid_actions):
        valid_mask[i, va] = True
    return valid_mask


def apply_cascading_masks(masks, *stop_conditions):
    """
    Updates masks by zeroing out elements where ANY stop_condition is True.
    
    Args:
        masks (np.array): The initial mask array.
        *stop_conditions (list): Variable number of lists/arrays to stack and filter by.
                                 (e.g., last_idles, last_truncates)
    """
    combined_stop_flags = None

    for cond in stop_conditions:
        current_flags = np.stack(cond, axis=1).astype(bool)
        if combined_stop_flags is None:
            combined_stop_flags = current_flags
        else:
            # In-place logical_or is memory efficient and fast
            np.logical_or(combined_stop_flags, current_flags, out=combined_stop_flags)

    # ~combined_stop_flags turns True (stop) into False.
    # .astype(float) turns False into 0.0, True into 1.0.
    keep_factor = (~combined_stop_flags).astype(np.float32)
    return masks * keep_factor


class Model_53(Agent):
    
    def __init__(self, 
                 policy_model: Policy_Network, 
                 value_model: Value_Network,
                 trainer: RL_Learner, supervised_trainer: Supervised_Learner, 
                 context_collector: Context_Collector, action_collector: Context_Collector, valid_action_collector: Context_Collector,
                 memory: Memory,
                 max_num_thought_steps: int = 2,
                 do_supervision: bool = False,
                 use_memory: bool = True
                 ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.trainer = trainer
        self.supervised_trainer = supervised_trainer
        self.obs = context_collector
        self.actions = action_collector
        self.valid_actions = valid_action_collector
        self.memory = memory

        self.max_num_thought_steps = max_num_thought_steps
        self.do_supervision = do_supervision
        self.use_memory = use_memory

        self.reset()


    def reset(self):
        self.trainer.reset(time=0.0)
        self.obs.clear()
        self.actions.clear()
        self.valid_actions.clear()
        self.rewards = []
        self.next_dones = []
        self.last_truncates = []
        self.last_idles = []

        self.policy_model.eval()
        self.value_model.eval()

        self.thought_steps = None


    def choose_action(self, 
                      last_idles, next_dones, last_truncates, last_resets, 
                      latest_frames, rewards, next_available_actions, 
                      force_train=False):

        batch_size = len(latest_frames)
        current_cl = len(self.rewards)

        # initialize
        if self.thought_steps is None:
            self.thought_steps = [0 for _ in range(batch_size)]
            self.obs.append(
                np.zeros((batch_size, 1), dtype=np.float32), 
                np.zeros((batch_size, self.policy_model.position_size), dtype=np.float32), 
                np.zeros((batch_size, self.policy_model.content_size), dtype=np.float32)
            )
            self.rewards.append(
                np.zeros((batch_size,), dtype=np.float32)
            )
            self.valid_actions.append(
                np.ones((batch_size, self.policy_model.flag_size), dtype=np.bool),
                np.ones((batch_size, self.policy_model.action_size), dtype=np.bool)
            )
            self.last_truncates.append([True for _ in range(batch_size)])
            self.last_idles.append([False for _ in range(batch_size)])
            self.next_dones.append([False for _ in range(batch_size)])

        for i, (idle, t) in enumerate(zip(last_idles, last_truncates)):
            if t:
                self.thought_steps[i] = 0

        content = np.reshape(np.stack(latest_frames, axis=0), (batch_size, -1)) # content must be batch leading tensor (batch_size, ...)
        reward = np.array([r for r in rewards])

        update_mask = np.zeros((batch_size, 1 + self.policy_model.position_size + self.policy_model.content_size), dtype=np.float32)
        memory_action = [Memory_Operation_Type.IDLE for _ in range(batch_size)]
        for i, (idle, d, t, r) in enumerate(zip(last_idles, next_dones, last_truncates, last_resets)):
            if r:
                memory_action[i] = Memory_Operation_Type.RESET
            if not idle:
                # update only reward and content
                update_mask[i, 0] = 1.0
                update_mask[i, 1 + self.policy_model.position_size:] = 1.0
                memory_action[i] = Memory_Operation_Type.CACHE
            self.last_truncates[-1][i] = t
            self.last_idles[-1][i] = idle
            self.next_dones[-1][i] = d

        # replace the reward and content
        last_obs = self.obs.get_last()
        new_value = np.concatenate([
            np.reshape(reward, (-1, 1)), 
            np.zeros((batch_size, self.policy_model.position_size), dtype=np.float32), 
            content
        ], axis=1)
        update_value = last_obs * (1.0 - update_mask) + new_value * update_mask
        self.obs.update_last(update_value)
        self.rewards[-1] = reward
        self.valid_actions.update_last(
            make_valid_mask([
                [0] if self.thought_steps[i] == self.max_num_thought_steps - 1 or not self.use_memory else list(range(self.policy_model.flag_size)) 
                for i in range(batch_size)
            ], self.policy_model.flag_size),
            make_valid_mask(next_available_actions, self.policy_model.action_size)
        )

        # cache new observation into memory
        position = last_obs[:, 1:1 + self.policy_model.position_size]
        self.memory.operate(
            tuple_record=(
                np.reshape(reward, (-1, 1)), 
                position, 
                content
            ), 
            operation=memory_action
        )

        if (any(next_dones) or any(last_truncates) or force_train) and current_cl > 1:
            
            self.policy_model.train()
            self.value_model.train()

            if self.do_supervision:
                # learn Supervise content

                # make action format
                recorded_obs = self.obs.make_batch(batch_led=True)[:, 1:, :]  # shape (batch_size, context_length, obs_size)
                target_actions = np.concatenate([
                    np.zeros((recorded_obs.shape[0], recorded_obs.shape[1], self.policy_model.packed_action_size - self.policy_model.content_size), dtype=np.float32),
                    recorded_obs[:, :, 1 + self.policy_model.position_size:]  # content part
                ], axis=-1)
                
                # masks has shape (batch_size, context_length)
                masks = self.actions.make_mask(batch_led=True)
                masks = apply_cascading_masks(
                    masks,
                    self.last_idles[1:],
                    self.last_truncates[1:],
                    self.next_dones[:-1]
                )

                self.supervised_trainer.train(
                    obs=self.obs[:-1].make_batch(batch_led=True),
                    actions=self.actions.make_batch(batch_led=True), 
                    target_actions=target_actions,
                    valid_actions=self.valid_actions[:-1].make_batch(batch_led=True),
                    masks=masks,
                    trained_logprob_indices=[4] # only content part
                )

            # learn RL

            # masks has shape (batch_size, context_length)
            masks = self.actions.make_mask(batch_led=True)
            # need to shift last_truncates by 1 to the left, because t signals whether t-1 is truncated
            masks = apply_cascading_masks(masks, self.last_truncates[1:])

            self.trainer.learn(
                obs=self.obs.make_batch(batch_led=True), 
                actions=self.actions.make_batch(batch_led=True), 
                rewards=self.rewards[1:],
                next_dones=self.next_dones[1:],
                masks=masks,
                valid_actions=self.valid_actions[:-1].make_batch(batch_led=True)
            )

            self.supervised_trainer.save()
            self.trainer.save()
            self.policy_model.save()
            self.value_model.save()
            
            # reset
            self.trainer.reset(time=0.0)

            self.actions.mark()

            left_over_slide = self.obs.mark(skip_last=True)
            self.valid_actions.mark(skip_last=True)
            self.rewards = self.rewards[left_over_slide]
            self.last_truncates = self.last_truncates[left_over_slide]
            self.last_idles = self.last_idles[left_over_slide]
            self.next_dones = self.next_dones[left_over_slide]

            self.policy_model.eval()
            self.value_model.eval()


        # Choose a random action
        # this one return batch leading tensors (batch, 1, ...)
        packed_action, position = self.policy_model.get_action(
            self.obs.make_batch(batch_led=True),
            self.actions.make_batch(batch_led=True, append_last=True),
            self.valid_actions.make_batch(batch_led=True)
        )

        # extract output here
        int_action, ext_action, content = self.policy_model.unpack_action(packed_action[:, -1, ...])
        position = position[:, -1, ...]

        # if next done, reset score and thought steps
        return_action = [None for _ in range(batch_size)]
        memory_action = [Memory_Operation_Type.IDLE for _ in range(batch_size)]
        memory_fetch_index = [-1 for _ in range(batch_size)]
        selected_int_action = np.zeros((batch_size,), dtype=int)
        for i, d in enumerate(next_dones):
            flag = int_action[i].item()
            self.thought_steps[i] += 1
            if flag == 0 or self.thought_steps[i] >= self.max_num_thought_steps:
                # observe external
                return_action[i] = ext_action[i]
                self.thought_steps[i] = 0
                memory_action[i] = Memory_Operation_Type.IDLE
                selected_int_action[i] = 0
            else:
                if self.use_memory:
                    if flag == 1:
                        memory_action[i] = Memory_Operation_Type.IDLE
                        selected_int_action[i] = 1
                    elif flag == 2:
                        # position based retrieve
                        memory_action[i] = Memory_Operation_Type.FETCH
                        memory_fetch_index[i] = 1
                        selected_int_action[i] = 2
                    elif flag == 3:
                        # content based retrieve
                        memory_action[i] = Memory_Operation_Type.FETCH
                        memory_fetch_index[i] = 2
                        selected_int_action[i] = 3
                    elif flag == 4:
                        # record node
                        memory_action[i] = Memory_Operation_Type.CACHE
                        selected_int_action[i] = 4
                else:
                    memory_action[i] = Memory_Operation_Type.IDLE
                    selected_int_action[i] = 1

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

        # off-policy warning: here we store the corrected action after memory fetch
        selected_actions = self.policy_model.pack_action(
            b_int=selected_int_action,
            b_ext=ext_action,
            b_content=content
        )
        
        # store last states
        self.actions.append(selected_actions)

        self.obs.append(reward, position, content)
        self.rewards.append(reward)
        self.valid_actions.append(
            np.ones((batch_size, self.policy_model.flag_size), dtype=np.bool),
            np.ones((batch_size, self.policy_model.action_size), dtype=np.bool)
        )
        self.last_truncates.append([False for _ in range(batch_size)])
        self.last_idles.append([return_action[i] is None for i in range(batch_size)])
        self.next_dones.append([False for _ in range(batch_size)])
        
        return return_action
    

if __name__ == "__main__":
    # test mask

    masks = np.ones((2, 5), dtype=np.float32)
    last_idles = [
        [False, True],
        [False, False],
        [True, False],
        [False, False],
        [False, True],
    ]
    last_truncates = [
        [False, False],
        [True, False],
        [False, False],
        [True, False],
        [False, False],
    ]
    next_dones = [
        [False, False],
        [False, True],
        [False, False],
        [True, True],
        [False, False],
    ]   
    updated_masks = apply_cascading_masks(masks, last_idles, last_truncates, next_dones)
    manual_masks = masks.copy()
    manual_masks = manual_masks * (1.0 - np.stack(last_idles, axis=1).astype(np.float32))
    manual_masks = manual_masks * (1.0 - np.stack(last_truncates, axis=1).astype(np.float32))
    manual_masks = manual_masks * (1.0 - np.stack(next_dones, axis=1).astype(np.float32))
    assert np.allclose(updated_masks, manual_masks), "Cascading mask application failed!"