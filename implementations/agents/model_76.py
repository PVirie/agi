import numpy as np
from enum import Enum
from interfaces.learning import RL_Learner
from interfaces.agent import Agent
from interfaces.network import Actor_Network, Value_Network
from interfaces.memory import Graph_Memory, Graph_Memory_Operation_Type
from interfaces.data_structure import Context_Collector
from implementations.agents.model_base import make_valid_mask, apply_cascading_masks, make_component_flags


class Scheme(str, Enum):
    FULL = "full"


# int-action flag -> graph memory op; external flags also act in the env, thought flags do not
EXTERNAL_FLAG_OPS = {
    1: Graph_Memory_Operation_Type.WRITE,
    2: Graph_Memory_Operation_Type.CREATE,
}
THOUGHT_FLAG_OPS = {
    3: Graph_Memory_Operation_Type.LINK,
    4: Graph_Memory_Operation_Type.WRITE | Graph_Memory_Operation_Type.MOVE,
    5: Graph_Memory_Operation_Type.CREATE,
    6: Graph_Memory_Operation_Type.ROTATE,
}
# flags whose op consumes the action's write value
WRITE_VALUE_FLAGS = tuple(sorted(
    flag for flag, op in {**EXTERNAL_FLAG_OPS, **THOUGHT_FLAG_OPS}.items()
    if op & (Graph_Memory_Operation_Type.WRITE | Graph_Memory_Operation_Type.CREATE)
))
# flags whose step acts in the env (consumes the ext action); flag 0 is the external no-op
EXTERNAL_FLAGS = (0, *sorted(EXTERNAL_FLAG_OPS))
# policy component index (int, ext, edge_1, edge_2, write) -> flags on which that component is used
COMPONENT_FLAGS = make_component_flags({**EXTERNAL_FLAG_OPS, **THOUGHT_FLAG_OPS}, EXTERNAL_FLAGS)


def make_non_thought_pair_masks(masks, flags, next_dones):
    """
    Mask of the transition aux loss obs_t + delta(graph)_{t -> t'} = obs_{t'}: keeps step t only if it is a non-thought
    step whose next non-thought step t' lies in the window, with every step in t .. t' unmasked and no done in t .. t'-1.
    masks: (batch, T); flags: (batch, T) int-action flags chosen at each step; next_dones: list (T) of bools of length batch
    """
    batch_size, T = flags.shape
    is_external = np.isin(flags.astype(int), EXTERNAL_FLAGS)
    index = np.where(is_external, np.arange(T)[None, :], T)
    at_or_after = np.minimum.accumulate(index[:, ::-1], axis=1)[:, ::-1]
    next_index = np.concatenate([at_or_after[:, 1:], np.full((batch_size, 1), T)], axis=1)
    has_next = next_index < T
    next_index = np.minimum(next_index, T - 1)

    # stops in [t, t'-1] (done or masked) plus a masked t'
    stops = np.stack(next_dones, axis=1).astype(bool) | (masks <= 0)
    stop_count = np.concatenate([np.zeros((batch_size, 1), dtype=int), np.cumsum(stops, axis=1)], axis=1)
    span_clear = np.take_along_axis(stop_count, next_index, axis=1) == stop_count[:, :-1]
    next_valid = np.take_along_axis(masks, next_index, axis=1) > 0

    return (masks * (is_external & has_next & span_clear & next_valid)).astype(np.float32)


class Model(Agent):
    
    def __init__(self, 
                 policy_model: Actor_Network,
                 trainer: RL_Learner, 
                 context_collector: Context_Collector, 
                 action_collector: Context_Collector, 
                 valid_action_collector: Context_Collector,
                 graph_memory: Graph_Memory,
                 max_num_thought_steps: int = 2,
                 do_supervision: bool = False,
                 scheme: Scheme = Scheme.FULL,
                 has_intrinsic_reward: bool = False
                 ):
        """
        has_intrinsic_reward: if False, each env reward passed to choose_action is a scalar (extrinsic);
                              if True, each is a pair [extrinsic, intrinsic].
                              Rewards are always stored as (batch_size, reward_size) arrays with
                              reward_size = 2 if has_intrinsic_reward else 1, and handed to the trainer
                              as a list (context_length) of such arrays.
        """
        self.policy_model = policy_model
        self.trainer = trainer
        self.obs = context_collector
        self.actions = action_collector
        self.valid_actions = valid_action_collector
        self.graph_memory = graph_memory

        self.max_num_thought_steps = max_num_thought_steps
        self.do_supervision = do_supervision
        self.has_intrinsic_reward = has_intrinsic_reward
        self.reward_size = 2 if has_intrinsic_reward else 1

        if scheme == Scheme.FULL:
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
        self.last_cause_times = []

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
                np.zeros((batch_size, self.reward_size), dtype=np.float32)
            )
            self.last_idles.append([False for _ in range(batch_size)])
            self.last_dones.append([False for _ in range(batch_size)])
            self.last_truncates.append([True for _ in range(batch_size)])
        
        content = np.reshape(np.stack(latest_frames, axis=0), (batch_size, -1)) # content must be batch leading tensor (batch_size, ...)
        reward = np.asarray(rewards, dtype=np.float32).reshape(batch_size, self.reward_size)   # (batch_size, reward_size); [:, 0] extrinsic

        # get last action's position
        last_action = self.actions.get_last()
        int_action, ext_action, position, write_content = self.policy_model.unpack_action(last_action)

        memory_action = [Graph_Memory_Operation_Type.IDLE for _ in range(batch_size)]
        for i, (idle, d, t, r) in enumerate(zip(last_idles, last_dones, last_truncates, last_resets)):
            flag = int_action[i].item()
            if r:
                memory_action[i] = Graph_Memory_Operation_Type.RESET
            elif not idle:
                memory_action[i] |= EXTERNAL_FLAG_OPS.get(flag, Graph_Memory_Operation_Type.IDLE)
            elif idle:
                reward[i] = 0
                memory_action[i] |= THOUGHT_FLAG_OPS.get(flag, Graph_Memory_Operation_Type.IDLE)
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
            write_value=write_content, # write value
            edge_1=position[:, 0],
            edge_2=position[:, 1],
            cause_time=len(self.rewards) - 2, # the pending op comes from the action chosen one step back
        )
        self.last_cause_times.append(self.graph_memory.get_cause_times())

        # update extrinsic reward by mem_op_results (0 for True, -0.001 for False)
        reward[:, 0] += np.array([0 if res else -0.001 for res in mem_op_results])

        context = self.graph_memory.get_node_context()
        context = context.reshape(batch_size, -1) # reshape to (batch_size, context_size)

        # replace the reward and content
        self.obs.update_last(
            self.policy_model.pack_context(
                b_reward=reward[:, 0],   # the policy observes the extrinsic reward only
                b_int=int_action,
                b_ext=ext_action,
                b_position=position,
                b_content=np.concatenate([context, content], axis=-1)
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
            last_actions = self.actions.make_batch(batch_led=True)
            aux_masks = make_non_thought_pair_masks(masks, last_actions[:, 1:, 0], self.last_dones[1:])

            self.trainer.learn(
                obs=self.obs.make_batch(batch_led=True), 
                last_actions=last_actions, 
                rewards=self.rewards[1:],
                next_dones=self.last_dones[1:],
                valid_actions=self.valid_actions[:-1].make_batch(batch_led=True),
                masks=masks,
                aux=aux_masks,
                causes=self.last_cause_times
            )
            
            left_over_slide = self.obs.mark(skip_last=True)
            self.actions.mark(skip_last=True)
            self.valid_actions.mark(skip_last=True)
            self.rewards = self.rewards[left_over_slide]
            self.last_idles = self.last_idles[left_over_slide]
            self.last_dones = self.last_dones[left_over_slide]
            self.last_truncates = self.last_truncates[left_over_slide]

            self.graph_memory.reset_timestamp()
            # keep one slot so causes stay index-aligned with obs; reset_timestamp just
            # invalidated the retained entry's stamps, so they must not be reused
            self.last_cause_times = [np.full_like(self.last_cause_times[-1], -1)]

        # Choose a random action
        packed_action = self.policy_model.get_action(
            self.obs.get_last_batch(batch_led=True),
            self.valid_actions.get_last_batch(batch_led=True)
        )
        packed_action = packed_action[:, -1, ...]

        # extract output here
        int_action, ext_action, position, write_content = self.policy_model.unpack_action(packed_action)

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
        self.rewards.append(np.zeros((batch_size, self.reward_size), dtype=np.float32))
        self.last_idles.append([return_action[i] is None for i in range(batch_size)])
        self.last_dones.append([False for _ in range(batch_size)])
        self.last_truncates.append([False for _ in range(batch_size)])

        # calculate metrics per batch
        metrics = []
        # 1. branching: mean excess degree sum(max(deg - 2, 0)) / N; 0 for paths/cycles, ~0.5 binary tree, ~1 star
        degrees = self.graph_memory.next_free_edge  # (B, num_nodes); unused nodes have degree 0
        metrics.append(np.maximum(degrees - 2, 0).sum(axis=1) / self.graph_memory.total_used_nodes())
        # 2. graph size
        metrics.append(self.graph_memory.total_used_nodes())

        return return_action, metrics
    

if __name__ == "__main__":
    # test make_non_thought_pair_masks; external flags are (0, 1, 2), thought flags are 3..6
    flags = np.array([
        [1, 3, 4, 0, 5, 2],  # thoughts in between; the last external step has no pair in the window
        [0, 0, 0, 0, 0, 0],  # consecutive external steps with a done at t=2
        [2, 3, 3, 1, 6, 6],  # masked thought step inside the span t=0 .. t'=3
        [1, 1, 3, 3, 3, 3],  # the pair target t'=1 is masked
    ], dtype=np.float32)
    masks = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
    ], dtype=np.float32)
    dones = np.zeros_like(flags, dtype=bool)
    dones[1, 2] = True
    next_dones = [list(column) for column in dones.T]  # list (T) of bools of length batch

    expected = np.array([
        [1, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)

    result = make_non_thought_pair_masks(masks, flags, next_dones)
    assert result.shape == expected.shape and result.dtype == np.float32, (result.shape, result.dtype)
    assert np.array_equal(result, expected), f"\nexpected:\n{expected}\ngot:\n{result}"

    # a single-step window never has a pair
    single = make_non_thought_pair_masks(np.ones((2, 1), dtype=np.float32), np.zeros((2, 1)), [[False, False]])
    assert np.array_equal(single, np.zeros((2, 1), dtype=np.float32)), single

    print("make_non_thought_pair_masks: all tests passed")