import numpy as np
from enum import Enum
from interfaces.learning import RL_Learner
from interfaces.agent import Agent
from interfaces.network import Policy_Network, Value_Network
from interfaces.memory import Graph_Memory, Graph_Memory_Operation_Type
from interfaces.data_structure import Context_Collector


def make_valid_mask(valid_actions, action_size):
    """
    valid_actions: a list (batch) of list of int
    action_size: int
    """
    valid_mask = np.zeros((len(valid_actions), action_size), dtype=bool)
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


class Scheme(str, Enum):
    FLIPFLOP = "flipflop"
    FULL = "full"


class Model_74(Agent):
    
    def __init__(self, 
                 policy_model: Policy_Network,
                 trainer: RL_Learner, 
                 context_collector: Context_Collector, 
                 action_collector: Context_Collector, 
                 valid_action_collector: Context_Collector,
                 graph_memory: Graph_Memory,
                 max_num_thought_steps: int = 2,
                 do_supervision: bool = False,
                 scheme: Scheme = Scheme.FLIPFLOP
                 ):
        self.policy_model = policy_model
        self.trainer = trainer
        self.obs = context_collector
        self.actions = action_collector
        self.valid_actions = valid_action_collector
        self.graph_memory = graph_memory

        self.max_num_thought_steps = max_num_thought_steps
        self.do_supervision = do_supervision

        # 0 obs idle, 1 for obs write, 2 for obs create
        # 3 thought for link, 4 thought for write then move, 5 thought create
        # 6 thought for rotate edge
        if scheme == Scheme.FLIPFLOP:
            self.valid_int_actions = [0, 1, 2, 3, 4, 5]
            self.observe_external_int_actions = [0, 1, 2]
        elif scheme == Scheme.FULL:
            self.valid_int_actions = [0, 1, 2, 3, 4, 5, 6]
            self.observe_external_int_actions = [0, 1, 2]

        self.reset()


    def reset(self):
        self.obs.clear()
        self.actions.clear()
        self.valid_actions.clear()
        self.rewards = []
        self.last_idles = []
        self.last_dones = []
        self.last_truncates = []
        self.last_edge_update_times = []

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
            self.obs.append(
                np.zeros((batch_size, self.policy_model.packed_context_size), dtype=np.float32)
            )
            self.actions.append(
                np.zeros((batch_size, self.policy_model.packed_action_size), dtype=np.float32)
            )
            self.valid_actions.append(
                np.ones((batch_size, self.policy_model.int_action_size), dtype=bool),
                np.ones((batch_size, self.policy_model.ext_action_size), dtype=bool)
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
        last_action = self.actions.get_last()
        int_action, ext_action, position, last_content = self.policy_model.unpack_action(last_action)

        memory_action = [Graph_Memory_Operation_Type.IDLE for _ in range(batch_size)]
        for i, (idle, d, t, r) in enumerate(zip(last_idles, last_dones, last_truncates, last_resets)):
            flag = int_action[i].item()
            if r:
                memory_action[i] |= Graph_Memory_Operation_Type.RESET
            if not idle:
                if flag == 1:
                    memory_action[i] |= Graph_Memory_Operation_Type.WRITE
                if flag == 2:
                    memory_action[i] |= Graph_Memory_Operation_Type.CREATE
            if idle:
                # if idle, use content and reward from last thought
                content[i, :] = last_content[i, :]
                reward[i] = 0
                if flag == 3:
                    memory_action[i] |= Graph_Memory_Operation_Type.LINK
                if flag == 4:
                    memory_action[i] |= Graph_Memory_Operation_Type.WRITE
                    memory_action[i] |= Graph_Memory_Operation_Type.MOVE
                if flag == 5:
                    memory_action[i] |= Graph_Memory_Operation_Type.CREATE
                if flag == 6:
                    memory_action[i] |= Graph_Memory_Operation_Type.ROTATE
            if d or t:
                self.thought_steps[i] = 0
                # reset position
                position[i, :] = np.zeros_like(position[i, :])
            self.last_idles[-1][i] = idle
            self.last_dones[-1][i] = d
            self.last_truncates[-1][i] = t

        # update memory
        mem_op_results = self.graph_memory.execute(
            operations=memory_action,
            write_value=content,
            edge_1=position[:, 0],
            edge_2=position[:, 1]
        )
        self.last_edge_update_times.append(self.graph_memory.get_edge_update_time())

        # update reward by mem_op_results (0 for True, -1 for False)
        reward += np.array([0 if res else -1 for res in mem_op_results])

        context = self.graph_memory.get_node_context()
        context = context.reshape(batch_size, -1) # reshape to (batch_size, context_size)

        # replace the reward and content
        self.obs.update_last(
            self.policy_model.pack_context(
                b_reward=reward,
                b_int=int_action,
                b_ext=ext_action,
                b_position=position,
                b_content=context
            )
        )
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
            
            # learn RL

            # masks has shape (batch_size, context_length)
            # need to shift last_truncates by 1 to the left, because t signals whether t-1 is truncated
            masks = apply_cascading_masks(
                self.actions.make_mask(batch_led=True)[:, :-1], 
                self.last_truncates[1:]
            )

            self.trainer.learn(
                obs=self.obs.make_batch(batch_led=True), 
                last_actions=self.actions.make_batch(batch_led=True), 
                rewards=self.rewards[1:],
                next_dones=self.last_dones[1:],
                valid_actions=self.valid_actions[:-1].make_batch(batch_led=True),
                masks=masks,
                causes=self.last_truncates
            )
            
            left_over_slide = self.obs.mark(skip_last=True)
            self.actions.mark(skip_last=True)
            self.valid_actions.mark(skip_last=True)
            self.rewards = self.rewards[left_over_slide]
            self.last_idles = self.last_idles[left_over_slide]
            self.last_dones = self.last_dones[left_over_slide]
            self.last_truncates = self.last_truncates[left_over_slide]

            self.graph_memory.reset_timestamp()
            self.last_edge_update_times = []

        # Choose a random action
        packed_action = self.policy_model.get_action(
            self.obs.get_last_batch(batch_led=True),
            self.valid_actions.get_last_batch(batch_led=True)
        )
        packed_action = packed_action[:, -1, ...]

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
        self.obs.append(
            np.zeros((batch_size, self.policy_model.packed_context_size), dtype=np.float32)
        )
        self.actions.append(packed_action)
        self.valid_actions.append(
            np.ones((batch_size, self.policy_model.int_action_size), dtype=bool),
            np.ones((batch_size, self.policy_model.ext_action_size), dtype=bool)
        )
        self.rewards.append(np.zeros((batch_size, ), dtype=np.float32))
        self.last_idles.append([return_action[i] is None for i in range(batch_size)])
        self.last_dones.append([False for _ in range(batch_size)])
        self.last_truncates.append([False for _ in range(batch_size)])

        return return_action, position
    

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
    last_dones = [
        [False, False],
        [False, True],
        [False, False],
        [True, True],
        [False, False],
    ] 
    last_truncates = [
        [False, False],
        [True, False],
        [False, False],
        [True, False],
        [False, False],
    ]  
    updated_masks = apply_cascading_masks(masks, last_idles, last_dones, last_truncates)
    manual_masks = masks.copy()
    manual_masks = manual_masks * (1.0 - np.stack(last_idles, axis=1).astype(np.float32))
    manual_masks = manual_masks * (1.0 - np.stack(last_dones, axis=1).astype(np.float32))
    manual_masks = manual_masks * (1.0 - np.stack(last_truncates, axis=1).astype(np.float32))
    assert np.allclose(updated_masks, manual_masks), "Cascading mask application failed!"