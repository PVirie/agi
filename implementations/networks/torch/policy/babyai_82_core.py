import torch
import torch.nn as nn
import numpy as np

from implementations.networks.torch.components.base import Categorical_With_Mask, init_weights
from implementations.networks.torch.components.std_lstm import Sliding_Recurrent_Cell
from implementations.networks.torch.policy.babyai_lstm import Policy_Core as LSTM_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


def balanced_weights(event, valid):
    """
    event, valid: (batch, context_size) bool / float
    returns per-step weights (batch, context_size) giving the valid event steps and the valid other steps half of the
    total weight each, with mean 1 over the valid steps; all ones when only one kind is present
    """
    event = event.float() * valid
    count = valid.sum()
    event_count = event.sum()
    other_count = count - event_count
    both = (event_count > 0) & (other_count > 0)
    weights = torch.where(event > 0, 0.5 * count / event_count.clamp(min=1), 0.5 * count / other_count.clamp(min=1))
    return torch.where(both, weights, torch.ones_like(weights))


class Policy_Core(LSTM_Policy_Core):

    def __init__(self, 
                 int_action_size, ext_action_size,
                 node_dim,
                 internal_state_size,
                 goal_size,
                 inventory_size,
                 width, height, channel,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers, 
                 memory_dim=None,
                 cnn_channels=128, cnn_blocks=2,
                 cell="lstm",
                 fusion_queries=4, fusion_layers=1,
                 component_flags=None,
                 predict_observation=False,
                 predict_reward=False,
                 obs_classes=(11, 6, 3),
                 normalize_intrinsic=True,
                 score_decay=0.99,
                 score_std_floor=0.05,
                 reward_event_threshold=0.01,
                 device=None, 
                 persistence_path=None, first_load_path=None):
        """
        babyai_lstm's encoder and recurrence, with every action head outside the recurrence (as model 76) and two value
        streams [extrinsic, intrinsic] for ppo_graph_split.
        Pipeline per step:
            graph context, instruction, inventory, obs -> encode (see babyai_lstm) -> latent
            latent, prev ext action, prev internal action (int flag, edge_1, edge_2, write value; plain inputs, no gradient),
            prev memory state -> Sliding_Recurrent_Cell -> memory state, hidden
            hidden -> ext head, int head, edge heads, write head (Beta over [0, 1]^node_dim), value head (2)
        Training signals:
            log-probs and entropies come in two parts, [..., 0] the ext action and [..., 1] the internal action
            (int flag + edges + write value), so the extrinsic return trains the ext action and the intrinsic return the
            internal actions.
        Intrinsic predictions (optional, each scores a match in [0, 1], 1 for a perfect prediction):
            from the previous step's latent (last observation), its ext action (last action) and its memory state (current state):
            predict_observation: the next observation, per cell and channel categorical over obs_classes;
                                 score = mean probability given to the observed values
            predict_reward: the step reward (the context's reward slot); score = 1 - min(|predicted - observed|, 1).
                            Rewards are sparse, so its squared-error loss gives the reward events (|reward| > reward_event_threshold)
                            and the other steps half of the weight each; otherwise "always 0" would be the optimum.
            get_intrinsic_reward turns the enabled scores into the intrinsic reward:
              normalize_intrinsic: each score is whitened by its running mean and std (EMA with score_decay per call,
                                   std floored at score_std_floor) before summing, so every prediction contributes the same
                                   variance and the rare reward events, whose std is small, stand out (~1/sqrt(event rate));
                                   the statistics are saved with the weights
              otherwise: the scores are summed as they are (in [0, 2] with both)
            The predictors are trained by the aux loss of get_log_prob_with_value_aux (cross entropy + squared error),
            which also shapes the latent and the recurrence.
            Where the recurrence restarts (an episode start) the observation is the next episode's and is not predicted;
            the reward is still predicted, as it is the reward of the episode's last transition (BabyAI's success reward).
            The previous latent rides in the action and the context so acting can score the prediction without re-encoding;
            it is only allocated when a prediction is enabled.
        component_flags: {2: edge_1 flags, 3: edge_2 flags, 4: write flags} as make_component_flags; the int flags on which
                         each internal component is used, others neither count in the log-prob nor the entropy.
                         None: every component is used on every flag.
        obs_classes: per-channel class counts of the raw grid (minigrid object, color, state); larger values are clamped
        """
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="babyai_82_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.internal_state_size = internal_state_size
        self.edge_size = internal_state_size - 1  # edge heads pick one of the head node's edge slots
        self.goal_size = goal_size
        self.inventory_size = inventory_size
        self.width = width
        self.height = height
        self.channel = channel
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.node_dim = node_dim
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.value_size = 2  # [extrinsic, intrinsic]

        self.position_size = 2
        self.content_size = internal_state_size * node_dim + goal_size + inventory_size + width * height * channel
        self.cell = cell
        self.memory_dim = memory_dim if memory_dim is not None else hidden_size
        self.memory_size = len(layers) * self.memory_dim

        self.predict_observation = predict_observation
        self.predict_reward = predict_reward
        self.obs_classes = tuple(obs_classes)
        self.latent_size = hidden_size if (predict_observation or predict_reward) else 0

        # [int flag, edge_1, edge_2, write value (node_dim)], contiguous in both the action and the context
        self.internal_output_size = 1 + self.position_size + node_dim
        # action: [ext, internal outputs, memory stack, latent]
        self.packed_action_size = 1 + self.internal_output_size + self.memory_size + self.latent_size
        # context: [reward, ext, internal outputs, memory reset, memory stack, latent, content]; all but reward and content are the previous step's
        self.packed_context_size = 1 + 1 + self.internal_output_size + 1 + self.memory_size + self.latent_size + self.content_size

        self.ext_pos = 1
        self.internal_pos = self.ext_pos + 1
        self.position_pos = self.internal_pos + 1
        self.write_pos = self.position_pos + self.position_size
        self.memory_reset_pos = self.write_pos + node_dim
        self.memory_pos = self.memory_reset_pos + 1
        self.latent_pos = self.memory_pos + self.memory_size
        self.internal_state_pos = self.latent_pos + self.latent_size
        self.goal_pos = self.internal_state_pos + self.internal_state_size * node_dim
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        # (int_action_size,) bool: on which flags each internal component is used; not saved, so changing them keeps checkpoints loadable
        def uses(flags):
            mask = torch.ones(int_action_size, dtype=torch.bool)
            if flags is not None:
                mask[:] = False
                mask[list(flags)] = True
            return mask
        component_flags = component_flags if component_flags is not None else {}
        self.register_buffer("uses_edge_1", uses(component_flags.get(2)), persistent=False)
        self.register_buffer("uses_edge_2", uses(component_flags.get(3)), persistent=False)
        self.register_buffer("uses_write", uses(component_flags.get(4)), persistent=False)

        self._build_encoder(cnn_channels, cnn_blocks, fusion_queries, fusion_layers)

        # the previous actions are plain inputs: their effects go through the environment and the graph memory
        self.recurrence = Sliding_Recurrent_Cell(
            output_dims=hidden_size * 2,
            input_dims=hidden_size + ext_action_size + int_action_size + 2 * self.edge_size + node_dim,
            hidden_dims=self.memory_dim,
            layers=[hidden_size for _ in layers],
            cell=cell,
        )

        def make_head(output_dims):
            return nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, output_dims)
            )

        self.head_ext = make_head(ext_action_size)
        self.head_int = make_head(int_action_size)
        self.head_edge_1 = make_head(self.edge_size)
        self.head_edge_2 = make_head(self.edge_size)
        self.head_write = make_head(2 * node_dim)  # Beta (alpha, beta) per dim
        self.head_value = make_head(self.value_size)

        # built only when enabled so checkpoints saved without them still load (load_state_dict is strict)
        prediction_input_size = self.latent_size + self.memory_size + ext_action_size
        if predict_observation:
            self.head_next_obs = nn.Sequential(
                nn.LayerNorm(prediction_input_size),
                nn.Linear(prediction_input_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, width * height * sum(self.obs_classes))
            )
        if predict_reward:
            self.head_step_reward = nn.Sequential(
                nn.LayerNorm(prediction_input_size),
                nn.Linear(prediction_input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            )

        # keeps the Beta log-prob finite for write values sampled at (float32) 0 or 1
        self.write_eps = 1e-6

        self.normalize_intrinsic = normalize_intrinsic
        self.score_decay = score_decay
        self.score_std_floor = score_std_floor
        self.reward_event_threshold = reward_event_threshold
        # running moments of the [observation, reward] scores: [[E obs, E reward], [E obs^2, E reward^2]];
        # saved with the weights so a resumed run keeps its reward scale
        self.register_buffer("score_moments", torch.zeros(2, 2))
        self.register_buffer("score_updates", torch.zeros(2))
        self.register_buffer("enabled_scores", torch.tensor([float(predict_observation), float(predict_reward)]), persistent=False)

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        self._reset_encoder_parameters()
        self.recurrence.reset_parameters()

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for head in (self.head_ext, self.head_int, self.head_edge_1, self.head_edge_2, self.head_write):
            head.apply(init_actor_weights)

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.head_value.apply(init_value_weights)

        if self.predict_observation:
            self.head_next_obs.apply(init_weights)
        if self.predict_reward:
            self.head_step_reward.apply(init_weights)

        self.score_moments.zero_()
        self.score_updates.zero_()


    def _write_distribution(self, params):
        # concentrations >= 1: unimodal, bounded density, and the entropy peaks at the uniform
        alpha, beta = (1.0 + nn.functional.softplus(params)).chunk(2, dim=-1)
        return torch.distributions.Beta(alpha, beta)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        prev_ext = context[:, :, self.ext_pos].long()  # (batch_size, context_size)
        prev_flag = context[:, :, self.internal_pos].long()  # (batch_size, context_size)
        prev_position = context[:, :, self.position_pos: self.write_pos].long().clamp(0, self.edge_size - 1)  # (batch_size, context_size, 2)
        prev_write = context[:, :, self.write_pos: self.memory_reset_pos].float()  # (batch_size, context_size, node_dim)
        memory_reset = context[:, :, self.memory_reset_pos] > 0.5  # (batch_size, context_size)
        prev_memory_state = context[:, :, self.memory_pos: self.latent_pos].float()  # (batch_size, context_size, memory_size)
        internal_state = context[:, :, self.internal_state_pos: self.goal_pos]
        goal = context[:, :, self.goal_pos: self.inv_pos]
        inv = context[:, :, self.inv_pos: self.obs_pos]
        obs = context[:, :, self.obs_pos:]

        latent = self.encode(internal_state, goal, inv, obs)  # (batch_size, context_size, hidden_size)

        one_hot = nn.functional.one_hot
        recurrent_input = torch.concat([
            latent,
            one_hot(prev_ext, num_classes=self.ext_action_size).float(),
            one_hot(prev_flag, num_classes=self.int_action_size).float(),
            one_hot(prev_position[..., 0], num_classes=self.edge_size).float(),
            one_hot(prev_position[..., 1], num_classes=self.edge_size).float(),
            prev_write,
        ], dim=-1)
        memory_state, hidden = self.recurrence(recurrent_input, prev_memory_state, memory_reset)  # (batch_size, context_size, memory_size), (batch_size, context_size, hidden_size * 2)

        values = self.head_value(hidden)  # (batch_size, context_size, 2)
        return (
            self.head_ext(hidden), self.head_int(hidden), self.head_edge_1(hidden), self.head_edge_2(hidden), self.head_write(hidden),
            values, memory_state, latent
        )


    def _prediction_scores(self, context, prev_latent, prev_memory_state):
        """
        Scores each step's observation and reward as predicted from the previous step:
        prev_latent (batch, context_size, hidden_size) and prev_memory_state (batch, context_size, memory_size) are the
        previous step's latent and memory state; its ext action and the targets are read from context.
        returns loss (batch, context_size): the enabled predictions' losses summed (with grad),
        scores (batch, context_size, 2): [observation, reward] in [0, 1] (no grad), 0 for a disabled or invalid prediction,
        and valid (batch, context_size, 2) bool: where each score counts. Where the recurrence restarts (an episode start)
        the observation is the next episode's, so it is not scored, but the reward still is: it is the reward of the
        previous step's transition, e.g. a success reward, which BabyAI only gives on an episode's last step
        """
        batch_size, context_size = context.shape[:2]
        obs_valid = context[:, :, self.memory_reset_pos] <= 0.5
        reward_valid = torch.ones_like(obs_valid)
        prev_ext = nn.functional.one_hot(context[:, :, self.ext_pos].long(), num_classes=self.ext_action_size).float()
        prediction_input = torch.concat([prev_latent, prev_memory_state, prev_ext], dim=-1)

        obs_loss = torch.zeros((batch_size, context_size), device=context.device)
        reward_loss = torch.zeros((batch_size, context_size), device=context.device)
        obs_score = torch.zeros((batch_size, context_size), device=context.device)
        reward_score = torch.zeros((batch_size, context_size), device=context.device)

        if self.predict_observation:
            num_cells = self.height * self.width
            obs_logits = self.head_next_obs(prediction_input).view(batch_size, context_size, num_cells, sum(self.obs_classes))
            obs_target = context[:, :, self.obs_pos:].reshape(batch_size, context_size, num_cells, self.channel).long()
            for c, logits in enumerate(obs_logits.split(list(self.obs_classes), dim=-1)):
                target = obs_target[..., c].clamp(0, self.obs_classes[c] - 1)
                # flat (N, C) cross entropy: the spatial nll variant has no deterministic CUDA kernel
                cross_entropy = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none"
                ).view(target.shape)  # (batch, context_size, num_cells)
                obs_loss = obs_loss + cross_entropy.mean(dim=-1)
                # probability given to the observed value, averaged over cells and channels
                obs_score = obs_score + torch.exp(-cross_entropy.detach()).mean(dim=-1) / self.channel

        if self.predict_reward:
            reward = context[:, :, 0].float()
            reward_error = self.head_step_reward(prediction_input).squeeze(-1) - reward
            event = reward.abs() > self.reward_event_threshold
            reward_loss = balanced_weights(event, reward_valid.float()) * reward_error ** 2
            reward_score = (1.0 - reward_error.detach().abs()).clamp(0.0, 1.0)

        valid = torch.stack([obs_valid, reward_valid], dim=-1)
        scores = torch.stack([obs_score, reward_score], dim=-1) * valid
        loss = obs_loss * obs_valid + reward_loss * reward_valid
        return loss, scores, valid


    def _intrinsic_reward(self, scores, valid, update_statistics):
        # scores (batch, context_size, 2) [observation, reward]; valid (batch, context_size, 2) bool, see _prediction_scores
        # returns (batch, context_size): the enabled valid scores summed, each whitened first if normalize_intrinsic
        valid = valid.float() * self.enabled_scores
        if not self.normalize_intrinsic:
            return (scores * valid).sum(dim=-1)
        if update_statistics:
            count = valid.sum(dim=(0, 1))  # (2,)
            batch_moments = torch.stack([
                (scores * valid).sum(dim=(0, 1)),
                (scores ** 2 * valid).sum(dim=(0, 1)),
            ]) / count.clamp(min=1)
            # per score: the first update sets the moments, later ones blend them in; no valid step keeps them
            decay = torch.where(self.score_updates > 0, torch.full_like(count, self.score_decay), torch.zeros_like(count))
            decay = torch.where(count > 0, decay, torch.ones_like(count))
            self.score_moments.copy_(decay * self.score_moments + (1.0 - decay) * batch_moments)
            self.score_updates += (count > 0).float()
        mean = self.score_moments[0]
        std = (self.score_moments[1] - mean ** 2).clamp(min=0.0).sqrt().clamp(min=self.score_std_floor)
        return ((scores - mean) / std * valid).sum(dim=-1)


    def get_intrinsic_reward(self, context, prev_memory_state=None, update_statistics=True):
        """
        context: (batch, context_size, packed_context_size), each step holding its observation and reward
        prev_memory_state: (batch, context_size, memory_size), the memory state the previous steps ended with, or None for
                           the context's memory slot; differs from it where the recurrence restarts (zeroed there), which
                           the reward prediction of an episode's last transition needs
        update_statistics: fold these steps' scores into the running statistics used by normalize_intrinsic
        returns numpy (reward (batch, context_size), scores (batch, context_size, 2)):
        the intrinsic reward of each step (see normalize_intrinsic) and its raw [observation, reward] scores in [0, 1],
        0 where not scored (see _prediction_scores)
        """
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(prev_memory_state, np.ndarray):
            prev_memory_state = torch.tensor(prev_memory_state, dtype=torch.float32).to(self.device)
        batch_size, context_size = context.shape[:2]
        if self.latent_size == 0:
            return np.zeros((batch_size, context_size), dtype=np.float32), np.zeros((batch_size, context_size, 2), dtype=np.float32)
        if prev_memory_state is None:
            prev_memory_state = context[:, :, self.memory_pos: self.latent_pos]
        with torch.no_grad():
            _, scores, valid = self._prediction_scores(
                context, context[:, :, self.latent_pos: self.internal_state_pos], prev_memory_state
            )
            reward = self._intrinsic_reward(scores, valid, update_statistics)
        return reward.cpu().numpy(), scores.cpu().numpy()


    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            # float: node values are continuous and would be truncated by a long cast
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        with torch.no_grad():
            ext_logits, int_logits, edge_1_logits, edge_2_logits, write_params, _, memory_state, latent = self.compute(context)
            ext_action = Categorical_With_Mask(logits=ext_logits, mask=available_actions).sample()
            int_action = Categorical_With_Mask(logits=int_logits, mask=available_flags).sample()
            edge_1 = Categorical_With_Mask(logits=edge_1_logits).sample()
            edge_2 = Categorical_With_Mask(logits=edge_2_logits).sample()
            write_value = self._write_distribution(write_params).sample().clamp(self.write_eps, 1 - self.write_eps)

        action = torch.concat([
            ext_action.unsqueeze(-1).float(),
            int_action.unsqueeze(-1).float(),
            edge_1.unsqueeze(-1).float(),
            edge_2.unsqueeze(-1).float(),
            write_value,
            memory_state,
            latent[..., :self.latent_size],
        ], dim=-1)
        return action.cpu().numpy().astype(float)


    def get_log_prob_with_value_aux(self, context, selected_action, valid_actions=None):
        return self._evaluate(context, selected_action, valid_actions)


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        return self._evaluate(context, selected_action, valid_actions)[:3]


    def _evaluate(self, context, selected_action, valid_actions=None):
        # context: (batch, context_size + 1, packed_context_size); selected_action: (batch, context_size, packed_action_size)
        # returns log-probs and entropies (batch, context_size, 2) [ext action, internal action], values (batch, context_size + 1, 2),
        # and the prediction loss (batch, context_size) of the observation and reward each action led to (None if no prediction)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(selected_action, np.ndarray):
            selected_action = torch.tensor(selected_action, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        ext_logits, int_logits, edge_1_logits, edge_2_logits, write_params, values, memory_state, latent = self.compute(context)

        # values keep all contexts: the last one is the bootstrap value
        # drop the last context for everything else: it is the observation after the last action
        probs_ext = Categorical_With_Mask(logits=ext_logits[:, :-1], mask=available_actions)
        probs_int = Categorical_With_Mask(logits=int_logits[:, :-1], mask=available_flags)
        probs_edge_1 = Categorical_With_Mask(logits=edge_1_logits[:, :-1])
        probs_edge_2 = Categorical_With_Mask(logits=edge_2_logits[:, :-1])
        probs_write = self._write_distribution(write_params[:, :-1])

        flag = selected_action[:, :, 1].long()
        uses_edge_1 = self.uses_edge_1[flag]
        uses_edge_2 = self.uses_edge_2[flag]
        uses_write = self.uses_write[flag]
        write_value = selected_action[:, :, 2 + self.position_size: 2 + self.position_size + self.node_dim].clamp(self.write_eps, 1 - self.write_eps)

        zeros = torch.zeros_like(flag, dtype=values.dtype)
        log_prob_internal = (
            probs_int.log_prob(flag)
            + torch.where(uses_edge_1, probs_edge_1.log_prob(selected_action[:, :, 2]), zeros)
            + torch.where(uses_edge_2, probs_edge_2.log_prob(selected_action[:, :, 3]), zeros)
            + torch.where(uses_write, probs_write.log_prob(write_value).sum(dim=-1), zeros)
        )
        # the write value's entropy is averaged over its node_dim dims, so it weighs as one head
        entropy_internal = (
            probs_int.entropy()
            + torch.where(uses_edge_1, probs_edge_1.entropy(), zeros)
            + torch.where(uses_edge_2, probs_edge_2.entropy(), zeros)
            + torch.where(uses_write, probs_write.entropy().mean(dim=-1), zeros)
        )

        log_probs = torch.stack([probs_ext.log_prob(selected_action[:, :, 0]), log_prob_internal], dim=-1)
        entropy = torch.stack([probs_ext.entropy(), entropy_internal], dim=-1)

        prediction_loss = None
        if self.latent_size > 0:
            # step t + 1 is predicted from step t; step 0's previous step is outside the window, so its stored latent and state stand in
            prev_latent = torch.concat([context[:, :1, self.latent_pos: self.internal_state_pos], latent[:, :-1]], dim=1)
            prev_memory_state = torch.concat([context[:, :1, self.memory_pos: self.latent_pos], memory_state[:, :-1]], dim=1)
            prediction_loss = self._prediction_scores(context, prev_latent, prev_memory_state)[0][:, 1:]

        return log_probs, entropy, values, prediction_loss


    def unpack_memory(self, packed_action):
        # (batch, memory_size): every recurrent layer's state, bottom layer first
        start = 1 + self.internal_output_size
        return packed_action[:, start: start + self.memory_size].astype(float)


    def unpack_latent(self, packed_action):
        # (batch, latent_size): the step's latent, the "last observation" its successor's predictions are made from
        return packed_action[:, 1 + self.internal_output_size + self.memory_size:].astype(float)


    def pack_context(self, b_reward=None, b_int=None, b_ext=None, b_position=None, b_write=None, b_memory_reset=None, b_memory=None, b_latent=None, b_content=None):
        # see babyai_lstm.pack_context; b_latent: previous latent (batch, latent_size), see unpack_latent
        batch_size = next(b.shape[0] for b in (b_reward, b_int, b_ext, b_position, b_write, b_memory_reset, b_memory, b_latent, b_content) if b is not None)
        if b_reward is None:
            b_reward = np.zeros((batch_size,), dtype=float)
        if b_int is None:
            b_int = np.zeros((batch_size,), dtype=int)
        if b_ext is None:
            b_ext = np.zeros((batch_size, 1), dtype=int)
        if b_position is None:
            b_position = np.zeros((batch_size, self.position_size), dtype=float)
        if b_content is None:
            b_content = np.zeros((batch_size, self.content_size), dtype=float)
        if b_write is None:
            b_write = np.zeros((batch_size, self.node_dim), dtype=float)
        if b_memory is None:
            b_memory = np.zeros((batch_size, self.memory_size), dtype=float)
            b_memory_reset = np.ones((batch_size,), dtype=bool)
        if b_memory_reset is None:
            b_memory_reset = np.zeros((batch_size,), dtype=bool)
        if b_latent is None:
            b_latent = np.zeros((batch_size, self.latent_size), dtype=float)
        return np.concatenate([
            np.reshape(b_reward, (batch_size, 1)),
            np.reshape(b_ext, (batch_size, 1)),
            np.reshape(b_int, (batch_size, 1)),
            np.reshape(b_position, (batch_size, self.position_size)),
            np.reshape(b_write, (batch_size, self.node_dim)),
            np.reshape(b_memory_reset, (batch_size, 1)),
            np.reshape(b_memory, (batch_size, self.memory_size)),
            np.reshape(b_latent, (batch_size, self.latent_size)),
            np.reshape(b_content, (batch_size, self.content_size))
        ], axis=-1).astype(float)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    C, B, T, D, F = 3, 4, 5, 12, 6  # node_dim D != embedding_dim (projected)
    # flag 0 uses no internal component, flag 1 edge_1 only, flag 2 the write value only, the rest all of them
    component_flags = {2: (1, 3, 4, 5), 3: (3, 4, 5), 4: (2, 3, 4, 5)}

    def make_core(**kwargs):
        torch.manual_seed(0)
        return Policy_Core(
            int_action_size=F, ext_action_size=8,
            node_dim=D, internal_state_size=1 + C,
            goal_size=5, inventory_size=3,
            width=7, height=7, channel=3,
            dict_size=32, embedding_dim=8, pad_token_id=0,
            hidden_size=32, layers=[1, 1], memory_dim=48,
            cnn_channels=16, cnn_blocks=1,
            cell="gru",
            component_flags=component_flags,
            device=torch.device("cpu"),
            **kwargs,
        )

    core = make_core(predict_observation=True, predict_reward=True)

    def random_content():
        return np.concatenate([
            np.random.uniform(0, 1, (B, (1 + C) * D)),
            np.random.randint(0, 32, (B, 5 + 3)),
            np.stack([np.random.randint(0, n, (B, 7 * 7)) for n in (11, 6, 3)], axis=-1).reshape(B, -1),
        ], axis=-1)

    valid = np.random.rand(B, T + 1, F + 8) > 0.3
    valid[:, :, 0] = True
    valid[:, :, F] = True

    # act one step at a time as model_82 does: the context carries the executed action, the memory and the latent
    restart = np.zeros((B, T + 1), dtype=bool)
    restart[:, 0] = True
    restart[0, 3] = True  # an episode boundary inside the window
    contexts, actions, acted_scores = [], [], []
    last_action = np.zeros((B, core.packed_action_size))
    for t in range(T + 1):
        last_int, last_ext, last_position, last_write = core.unpack_action(last_action)
        prev_memory = core.unpack_memory(last_action)
        memory = prev_memory.copy()
        memory[restart[:, t]] = 0
        context = core.pack_context(
            b_reward=np.random.choice([0.0, 0.5, 1.0], B),
            b_int=last_int,
            b_ext=last_ext,
            b_position=last_position,
            b_write=last_write,
            b_memory_reset=restart[:, t],
            b_memory=memory,
            b_latent=core.unpack_latent(last_action),
            b_content=random_content(),
        )
        acted_scores.append(core.get_intrinsic_reward(context[:, None], prev_memory[:, None])[1][:, 0])
        last_action = core.get_action(context[:, None], valid[:, t:t + 1])[:, -1]
        contexts.append(context)
        actions.append(last_action)
    contexts = torch.tensor(np.stack(contexts, axis=1), dtype=torch.float32)
    actions = torch.tensor(np.stack(actions, axis=1), dtype=torch.float32)
    acted_scores = np.stack(acted_scores, axis=1)  # (B, T + 1, 2)
    valid = torch.tensor(valid)
    assert (valid[..., :F].gather(-1, actions[..., 1:2].long()).all() and valid[..., F:].gather(-1, actions[..., 0:1].long()).all()), "sampled actions must be valid"
    write_value = actions[..., 4: 4 + D]
    assert ((write_value > 0) & (write_value < 1)).all() and write_value.unique().numel() > 1, "the write value is sampled in (0, 1)^node_dim"
    assert core.packed_action_size == 1 + 3 + D + 2 * 48 + 32

    # learning slides through time and reproduces acting: memory, latent and the intrinsic reward
    ext_logits, int_logits, _, _, _, values, memory_state, latent = core.compute(contexts)
    assert torch.allclose(memory_state, actions[:, :, 4 + D: 4 + D + 2 * 48], atol=1e-5), "unrolled memory differs from the acted memory"
    assert torch.allclose(latent, actions[:, :, 4 + D + 2 * 48:], atol=1e-5), "unrolled latent differs from the acted latent"
    assert values.shape == (B, T + 1, 2)
    with torch.no_grad():
        prev_latent = torch.concat([contexts[:, :1, core.latent_pos: core.internal_state_pos], latent[:, :-1]], dim=1)
        prev_memory = torch.concat([contexts[:, :1, core.memory_pos: core.latent_pos], memory_state[:, :-1]], dim=1)
        learned_loss, learned_scores, learned_valid = core._prediction_scores(contexts, prev_latent, prev_memory)
    assert np.allclose(learned_scores.numpy(), acted_scores, atol=1e-4), "acting and learning must score the same predictions"
    assert ((acted_scores >= 0) & (acted_scores <= 1)).all() and (acted_scores[~restart][:, 0] > 0).all()
    assert (acted_scores[restart][:, 0] == 0).all(), "no observation prediction across a restart"
    assert learned_valid[0, 3].tolist() == [False, True], "the reward of an episode's last transition is still predicted"
    with torch.no_grad():
        # from the episode's last memory state, not the restart's zeroed one
        step = contexts[:, 3:4]
        step_latent = step[:, :, core.latent_pos: core.internal_state_pos]
        zeroed_loss = core._prediction_scores(step, step_latent, step[:, :, core.memory_pos: core.latent_pos])[0]
        true_loss, true_scores, _ = core._prediction_scores(step, step_latent, actions[:, 2:3, 4 + D: 4 + D + 2 * 48])
    assert not torch.isclose(zeroed_loss[0], true_loss[0]).all() and torch.allclose(zeroed_loss[1:], true_loss[1:], atol=1e-5)
    assert torch.allclose(true_scores[:, 0], learned_scores[:, 3], atol=1e-5)

    log_probs, entropy, values, aux = core.get_log_prob_with_value_aux(contexts, actions[:, :T], valid[:, :T])
    assert log_probs.shape == (B, T, 2) and entropy.shape == (B, T, 2) and values.shape == (B, T + 1, 2) and aux.shape == (B, T)
    assert torch.isfinite(log_probs).all() and torch.isfinite(aux).all() and (aux >= 0).all()
    assert torch.equal(core.get_log_probability_with_value(contexts, actions[:, :T], valid[:, :T])[0], log_probs)

    # the extrinsic part trains the ext head only, the intrinsic part the internal heads only
    def head_grads(loss):
        core.zero_grad()
        loss.backward()
        return {name: sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
                for name, head in [("ext", core.head_ext), ("int", core.head_int), ("edge_1", core.head_edge_1),
                                   ("edge_2", core.head_edge_2), ("write", core.head_write)]}

    forced = actions[:, :T].clone()
    forced[:, :, 1] = 3  # a flag using every internal component
    grads = head_grads(core.get_log_probability_with_value(contexts, forced, None)[0][..., 0].sum())
    assert grads["ext"] > 0 and all(grads[k] == 0 for k in ("int", "edge_1", "edge_2", "write")), grads
    grads = head_grads(core.get_log_probability_with_value(contexts, forced, None)[0][..., 1].sum())
    assert grads["ext"] == 0 and all(grads[k] > 0 for k in ("int", "edge_1", "edge_2", "write")), grads

    # an internal component counts only on the flags that use it
    with torch.no_grad():
        for flag, used in [(0, ()), (1, (2,)), (2, (4,)), (4, (2, 3, 4))]:
            forced = actions[:, :T].clone()
            forced[:, :, 1] = flag
            base = core.get_log_probability_with_value(contexts, forced, None)[0][..., 1]
            for index in (2, 3, 4):
                changed = forced.clone()
                if index < 4:
                    changed[:, :, index] = (changed[:, :, index] + 1) % C
                else:
                    changed[:, :, 4: 4 + D] = 1 - changed[:, :, 4: 4 + D]
                differs = not torch.allclose(core.get_log_probability_with_value(contexts, changed, None)[0][..., 1], base)
                assert differs == (index in used), f"flag {flag}: component {index} counted={differs}"

    # the aux loss trains the predictors: the prediction match rises
    optimizer = torch.optim.Adam(core.parameters(), lr=3e-3)
    for _ in range(60):
        loss = core.get_log_prob_with_value_aux(contexts, actions[:, :T], valid[:, :T])[3].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, _, _, _, _, _, memory_state, latent = core.compute(contexts)
        prev_latent = torch.concat([contexts[:, :1, core.latent_pos: core.internal_state_pos], latent[:, :-1]], dim=1)
        prev_memory = torch.concat([contexts[:, :1, core.memory_pos: core.latent_pos], memory_state[:, :-1]], dim=1)
        trained_scores = core._prediction_scores(contexts, prev_latent, prev_memory)[1]
    assert trained_scores[:, 1:].sum(-1).mean() > learned_scores[:, 1:].sum(-1).mean() + 0.3, (trained_scores.mean(), learned_scores.mean())

    # the reward loss gives the rare reward events as much weight as all the other steps together
    event = torch.zeros(4, 25, dtype=torch.bool)
    event[0, 3] = True
    event[2, 7] = True
    step_valid = torch.ones(4, 25)
    step_valid[1, :5] = 0
    weights = balanced_weights(event, step_valid)
    assert torch.isclose((weights * step_valid).sum() / step_valid.sum(), torch.tensor(1.0))
    assert torch.isclose((weights * event * step_valid).sum(), (weights * ~event * step_valid).sum())
    assert torch.equal(balanced_weights(torch.zeros_like(event), step_valid), torch.ones(4, 25)), "no event: uniform weights"

    # whitening: each score gets zero mean and unit std, so a rare reward miss stands out above the denser observation noise
    whitener = make_core(predict_observation=True, predict_reward=True)
    always = torch.ones(512, 1, 2, dtype=torch.bool)
    for _ in range(300):
        obs_scores = 0.7 + 0.05 * torch.randn(512, 1)
        reward_scores = torch.where(torch.rand(512, 1) < 0.01, 0.3, 0.999)
        rewards = whitener._intrinsic_reward(torch.stack([obs_scores, reward_scores], -1), always, update_statistics=True)
    probe = torch.tensor([[[0.7, 0.999]], [[0.75, 0.999]], [[0.7, 0.3]]])  # typical, a better observation guess, a missed reward
    typical, better_obs, missed = whitener._intrinsic_reward(probe, torch.ones(3, 1, 2, dtype=torch.bool), update_statistics=False)[:, 0]
    assert abs(typical) < 0.5 and 0.5 < better_obs - typical < 1.5 and typical - missed > 5, (typical, better_obs, missed)
    assert whitener._intrinsic_reward(probe, torch.zeros(3, 1, 2, dtype=torch.bool), update_statistics=False).abs().sum() == 0, "unscored steps add nothing"
    # a batch without any valid observation score leaves its statistics as they are
    moments = whitener.score_moments.clone()
    whitener._intrinsic_reward(probe, torch.tensor([[[False, True]]] * 3), update_statistics=True)
    assert torch.equal(whitener.score_moments[:, 0], moments[:, 0]) and not torch.equal(whitener.score_moments[:, 1], moments[:, 1])
    moments = whitener.score_moments.clone()
    reloaded = make_core(predict_observation=True, predict_reward=True)
    reloaded.load_state_dict(whitener.state_dict())
    assert torch.equal(reloaded.score_moments, moments), "the statistics are saved with the weights"

    # unnormalized, each prediction adds its score in [0, 1]; a disabled one scores 0 and adds nothing either way
    obs_only = make_core(predict_observation=True, normalize_intrinsic=False)
    reward_only = make_core(predict_reward=True)
    none = make_core()
    assert none.latent_size == 0 and none.packed_action_size == 1 + 3 + D + 2 * 48
    context = obs_only.pack_context(b_reward=np.random.rand(B), b_memory=np.zeros((B, obs_only.memory_size)), b_content=random_content())
    reward, scores = obs_only.get_intrinsic_reward(context[:, None])
    assert ((reward > 0) & (reward <= 1)).all() and np.allclose(reward, scores[..., 0]) and (scores[..., 1] == 0).all()
    context = reward_only.pack_context(b_reward=np.random.rand(B), b_memory=np.zeros((B, reward_only.memory_size)), b_content=random_content())
    reward, scores = reward_only.get_intrinsic_reward(context[:, None])
    assert (scores[..., 0] == 0).all() and np.isfinite(reward).all()
    context = none.pack_context(b_content=random_content(), b_memory=np.zeros((B, none.memory_size)))
    assert all((x == 0).all() for x in none.get_intrinsic_reward(context[:, None]))
    pair = np.stack([context, context], axis=1)
    pair_action = none.get_action(pair)[:, :1]
    assert none.get_log_prob_with_value_aux(pair, pair_action)[3] is None

    print("All tests passed!")
