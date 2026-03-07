import numpy as np
from enum import Enum
from interfaces.learning import RL_Index_Learner
from interfaces.agent import Agent
from interfaces.network import Policy_Index_Network, Value_Index_Network
from interfaces.memory import Episodic_Memory
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


class Model_68(Agent):
    
    def __init__(self, 
                 policy_model: Policy_Index_Network, value_model: Value_Index_Network,
                 trainer: RL_Index_Learner, 
                 index_collector: Context_Collector,
                 action_collector: Context_Collector, valid_action_collector: Context_Collector,
                 memory: Episodic_Memory,
                 max_num_thought_steps: int = 2,
                 minibatch_size: int = 8,
                 do_supervision: bool = False,
                 ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.trainer = trainer
        self.indices = index_collector
        self.actions = action_collector
        self.valid_actions = valid_action_collector
        self.memory = memory

        self.max_num_thought_steps = max_num_thought_steps
        self.do_supervision = do_supervision
        self.minibatch_size = minibatch_size

        self.valid_int_actions = [0]
        # set of external observation is a subset of valid_int_actions that triggers external observation override
        self.observe_external_int_actions = list(set([0]).intersection(set(self.valid_int_actions)))

        self.reset()


    def reset(self):
        self.indices.clear()
        self.actions.clear()
        self.valid_actions.clear()
        self.rewards = []
        self.last_idles = []
        self.last_dones = []
        self.last_truncates = []

        self.thought_steps = None


    def choose_action(self, 
                      last_idles, last_dones, last_truncates, last_resets, 
                      latest_frames, rewards, next_available_actions, 
                      force_train=False):

        batch_size = len(latest_frames)
        current_cl = len(self.rewards)

        # initialize
        if self.thought_steps is None:
            self.thought_steps = [0 for _ in range(batch_size)]
            self.indices.append(
                np.zeros((batch_size, self.policy_model.working_memory_size), dtype=np.int32)
            )
            self.actions.append(
                np.zeros((batch_size, self.policy_model.packed_action_size), dtype=np.float32)
            )
            self.valid_actions.append(
                np.ones((batch_size, self.policy_model.int_action_size), dtype=np.bool),
                np.ones((batch_size, self.policy_model.ext_action_size), dtype=np.bool)
            )
            self.rewards.append(
                np.zeros((batch_size,), dtype=np.float32)
            )
            self.last_idles.append([False for _ in range(batch_size)])
            self.last_dones.append([False for _ in range(batch_size)])
            self.last_truncates.append([True for _ in range(batch_size)])
        
        content = np.reshape(np.stack(latest_frames, axis=0), (batch_size, -1)) # content must be batch leading tensor (batch_size, ...)
        reward = np.array([r for r in rewards])

        # get last action's position
        index = self.indices.get_last()
        last_action = self.actions.get_last()
        int_action, ext_action, position, _ = self.policy_model.unpack_action(last_action)

        for i, (idle, d, t, r) in enumerate(zip(last_idles, last_dones, last_truncates, last_resets)):
            flag = int_action[i].item()
            if r:
                # clear all memory
                self.memory.reset(i)
            if not idle:
                # cache
                cache_pos = self.memory.cache(i, (reward[i:i+1], int_action[i:i+1], ext_action[i:i+1], position[i], content[i]))
                index[i, flag] = cache_pos
            else:
                # if idle, use content and reward from last thought
                fetch_pos = self.memory.fetch(i, (reward[i:i+1], int_action[i:i+1], ext_action[i:i+1], position[i], content[i]), pivot_index=3)
                index[i, flag] = fetch_pos
            if d or t:
                self.thought_steps[i] = 0
            self.last_idles[-1][i] = idle
            self.last_dones[-1][i] = d
            self.last_truncates[-1][i] = t


        # replace the reward and content
        self.indices.update_last(index)
        self.rewards[-1] = reward
        self.valid_actions.update_last(
            make_valid_mask([
                self.observe_external_int_actions if self.thought_steps[i] == self.max_num_thought_steps else self.valid_int_actions 
                for i in range(batch_size)
            ], self.policy_model.int_action_size),
            make_valid_mask(next_available_actions, self.policy_model.ext_action_size)
        )

        # if (any(last_dones) or any(last_truncates) or force_train) and current_cl > 1:
        if force_train and current_cl > 1:
            
            if self.do_supervision:
                # learn Supervise content
                # masks has shape (batch_size, context_length)
                svl_masks = apply_cascading_masks(
                    self.actions.make_mask(batch_led=True)[:, :-1],
                    self.last_idles[1:],
                    self.last_dones[1:],
                    self.last_truncates[1:]
                )
            else:
                svl_masks = None

            # learn RL

            # masks has shape (batch_size, context_length)
            # need to shift last_truncates by 1 to the left, because t signals whether t-1 is truncated
            masks = apply_cascading_masks(
                self.actions.make_mask(batch_led=True)[:, :-1], 
                self.last_truncates[1:]
            )

            self.trainer.learn(
                obs=self.memory.make_batch(batch_led=True), 
                indices=self.indices.make_batch(batch_led=True),
                last_actions=self.actions.make_batch(batch_led=True), 
                rewards=self.rewards[1:],
                next_dones=self.last_dones[1:],
                valid_actions=self.valid_actions[:-1].make_batch(batch_led=True),
                masks=masks,
                aux_masks=svl_masks
            )
            
            left_over_slide = self.indices.mark(skip_last=True)
            self.actions.mark(skip_last=True)
            self.valid_actions.mark(skip_last=True)
            self.rewards = self.rewards[left_over_slide]
            self.last_idles = self.last_idles[left_over_slide]
            self.last_dones = self.last_dones[left_over_slide]
            self.last_truncates = self.last_truncates[left_over_slide]


        # Choose a random action
        b_memory = self.memory.make_batch(batch_led=True) # this is use as ref, need to feed all
        b_indices = self.indices.get_last_batch(batch_led=True)
        b_valid_actions = self.valid_actions.get_last_batch(batch_led=True)
        packed_action = []
        for start in range(0, batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            mb_memory = b_memory[start:end, ...]
            mb_indices = b_indices[start:end, ...]
            mb_valid_actions = b_valid_actions[start:end, ...]

            mb_packed_action = self.policy_model.get_action(
                context=mb_memory,
                indices=mb_indices,
                valid_actions=mb_valid_actions,
            )
            packed_action.append(mb_packed_action[:, -1, ...])
        packed_action = np.concatenate(packed_action, axis=0)  # (batch_size, packed_action_size)

        # extract output here
        int_action, ext_action, position, content = self.policy_model.unpack_action(packed_action)

        # check flag for external observation override and update thought steps
        return_action = [None for _ in range(batch_size)]
        for i in range(batch_size):
            flag = int_action[i].item()
            self.thought_steps[i] += 1
            if flag in self.observe_external_int_actions:
                return_action[i] = ext_action[i]
                self.thought_steps[i] = 0

        # populate last states
        self.indices.append(
            np.zeros((batch_size, self.policy_model.working_memory_size), dtype=np.int32)
        )
        self.actions.append(packed_action)
        self.valid_actions.append(
            np.ones((batch_size, self.policy_model.int_action_size), dtype=np.bool),
            np.ones((batch_size, self.policy_model.ext_action_size), dtype=np.bool)
        )
        self.rewards.append(np.zeros((batch_size, 1), dtype=np.float32))
        self.last_idles.append([return_action[i] is None for i in range(batch_size)])
        self.last_dones.append([False for _ in range(batch_size)])
        self.last_truncates.append([False for _ in range(batch_size)])

        return return_action
    