import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Beta, Independent

from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import FiLMMiniGridCNN, FiLMUNet
from implementations.networks.torch.components.wavenet import BasicWavenet
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.policy.minigrid import Policy_Core as Minigrid_Policy_Core
from implementations.networks.torch.policy.babyai import Projector  # noqa: F401 -- re-exported for the task
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Minigrid_Policy_Core):

    def __init__(self, 
                 action_size,
                 node_dim,
                 internal_state_size,
                 goal_size,
                 inventory_size,
                 width, height, channel,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers, 
                 cnn_channels=128, cnn_blocks=2,
                 value_size=1,
                 location_size=64,
                 navigate_flag=2,
                 empty_value=-1.0,
                 use_aux=False,
                 obs_classes=(11, 6, 3),
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="babyai_77_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.action_size = action_size
        self.node_dim = node_dim
        # graph memory's value for unused edge slots (must match its empty_value); such node tokens are masked out
        self.empty_value = empty_value
        # keeps Beta log-probs finite for samples that round to the support boundary
        self.write_eps = 1e-4

        self.internal_state_size = internal_state_size
        self.goal_size = goal_size
        self.inventory_size = inventory_size
        self.width = width
        self.height = height
        self.channel = channel
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        # 1 for single-stream PPO (values squeezed to (batch, context)); 2 for ppo_graph_dual [extrinsic, intrinsic]
        self.value_size = value_size

        # navigate: each step emits a query/key "location vector"; q_t . k_j picks a previous step j to link to (edge_1),
        # and head_bridge decides between creating a new node (edge_2=0) and linking to step j (edge_2=1)
        self.location_size = location_size
        # action flag mapped to NAVIGATE in model_77's action_map; edge_1/edge_2 only count on these steps
        self.navigate_flag = navigate_flag
        self._bridge_key_cache = None  # rollout-only cache of past keys, see get_action

        self.position_size = 2
        self.content_size = internal_state_size * node_dim + goal_size + inventory_size + width * height * channel
        # [flag, edge_1 (window step), edge_2 (0: create, 1: link), write_value (node_dim)]
        self.packed_action_size = 1 + self.position_size + node_dim
        # [reward, flag, position, content]
        self.packed_context_size = 1 + 1 + self.position_size + self.content_size

        self.internal_state_pos = 1 + 1 + self.position_size
        self.goal_pos = self.internal_state_pos + self.internal_state_size * node_dim
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        self.pad_token_id = pad_token_id

        self.internal_state_embedding = nn.Linear(node_dim, embedding_dim)  # continuous node value -> token
        self.goal_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for goal tokens
        self.inv_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for inventory tokens

        # self.goal_wavenet = BasicWavenet(d_model=embedding_dim, num_layers=2)

        self.internal_state_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8,
            num_layers=4,
            dropout=0.01,
            max_len=internal_state_size + goal_size + inventory_size,
            use_pe=True,  # mission semantics depend on word order (attribute binding, "then"/"after")
            pooling="cls",  # mean pooling dilutes order-dependent structure ("A then B" vs "B then A")
        )

        # vision conditioned on the instruction/internal state features via FiLM
        self.conv_layers = FiLMMiniGridCNN(
            output_dims=hidden_size, 
            cond_dims=hidden_size,
            input_channels=channel, 
            width=width, height=height,
            vocab_size=256, embedding_dim=8,
            channels=cnn_channels, num_blocks=cnn_blocks
        )

        # internal_state, goal, inv, obs -> hidden
        vec_dim = action_size + hidden_size + hidden_size 
        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=vec_dim, 
            hidden_dims=hidden_size, 
            layers=[hidden_size for _ in layers]
        )

        self.head_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_size)   # action_size classes
        )
        self.head_bridge_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, location_size)
        )
        self.head_bridge_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, location_size)
        )
        self.head_bridge = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        self.head_write_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2 * node_dim)   # Beta (alpha, beta) per node dimension
        )

        self.head_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, value_size)
        )

        # auxiliary transition model (see _aux_loss): U-Net(obs_t | delta(graph)_t) -> obs_{t+1}
        # only built when enabled so checkpoints saved without it still load (load_state_dict is strict)
        self.use_aux = use_aux
        self.obs_classes = tuple(obs_classes)  # per-channel class counts of the raw grid (minigrid object, color, state)
        self.aux_write_tol = 1e-4  # a write landed if the next head node equals the sampled write value
        if use_aux:
            self.aux_node_embedding = nn.Linear(node_dim, hidden_size)
            self.aux_delta_net = nn.Sequential(
                nn.Linear(2 * hidden_size + 1, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.aux_unet = FiLMUNet(
                out_channels=sum(self.obs_classes),
                cond_dims=hidden_size,
                input_channels=channel,
                width=width, height=height,
                vocab_size=256, embedding_dim=8,
                channels=cnn_channels, num_blocks=1
            )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.internal_state_embedding.reset_parameters()
        self.goal_embedding.reset_parameters()
        self.inv_embedding.reset_parameters()

        # self.goal_wavenet.reset_parameters()

        self.internal_state_feature_extraction.reset_parameters()

        self.conv_layers.reset_parameters()

        self.backbone.reset_parameters()

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.head_action.apply(init_actor_weights)
        self.head_bridge_k.apply(init_actor_weights)
        self.head_bridge_q.apply(init_actor_weights)
        self.head_bridge.apply(init_actor_weights)
        self.head_write_value.apply(init_actor_weights)

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.head_value.apply(init_value_weights)

        if self.use_aux:
            self.aux_node_embedding.apply(init_value_weights)
            self.aux_delta_net.apply(init_value_weights)
            self.aux_unet.reset_parameters()


    def _encode(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.action_size).float()
        internal_state = context[:, :, self.internal_state_pos: self.goal_pos]  # (batch_size, context_size, internal_state_size * node_dim)
        internal_state = internal_state.reshape(batch_size, context_size, self.internal_state_size, self.node_dim)  # (batch_size, context_size, internal_state_size, node_dim)
        goal = context[:, :, self.goal_pos: (self.goal_pos + self.goal_size)]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: (self.inv_pos + self.inventory_size)]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos: ]  # (batch_size, context_size, obs_size)

        # a node token is padding when every dimension holds the empty value (unused edge slot)
        internal_state_mask = (internal_state != self.empty_value).any(dim=-1).float()  # (batch_size, context_size, internal_state_size)
        goal_inv_mask = get_padding_mask(torch.concat([goal, inv], dim=-1), self.pad_token_id)  # (batch_size, context_size, goal_size + inventory_size)
        padding_mask = torch.concat([internal_state_mask, goal_inv_mask], dim=-1)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size)
        
        internal_state_embedded = self.internal_state_embedding(internal_state.float())  # (batch_size, context_size, internal_state_size, embedding_dim)
        goal_embedded = self.goal_embedding(goal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        inv_embedded = self.inv_embedding(inv.long())  # (batch_size, context_size, inventory_size, embedding_dim)

        # # run wavenet on goal to get a better representation of the goal sequence
        # goal_embedded = goal_embedded.view(batch_size * context_size, self.goal_size, self.embedding_dim)  # (batch_size * context_size, goal_size, embedding_dim)
        # goal_embedded = self.goal_wavenet(goal_embedded)  # (batch_size * context_size, goal_size, embedding_dim)
        # goal_embedded = goal_embedded.view(batch_size, context_size, self.goal_size, self.embedding_dim)  # (batch_size, context_size, goal_size, embedding_dim)

        internal_state_goal_embedded = torch.concat([internal_state_embedded, goal_embedded, inv_embedded], dim=2)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size, embedding_dim)
        internal_state_features = self.internal_state_feature_extraction(internal_state_goal_embedded, padding_mask)  # (batch_size, context_size, hidden_size)

        obs_features = torch.reshape(obs, (batch_size * context_size, self.height, self.width, self.channel))  # (batch_size * context_size, height, width, channel)
        cond = internal_state_features.reshape(batch_size * context_size, self.hidden_size)
        obs_features = self.conv_layers(obs_features, cond)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size) # (batch_size, context_size, hidden_size)

        base_input = torch.concat([flag_onehot, internal_state_features, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        base_output = self.backbone(base_input)  # (batch_size, context_size, hidden_size)
        return base_output


    def compute(self, context):
        base_output = self._encode(context)  # (batch_size, context_size, hidden_size)

        action_logits = self.head_action(base_output)  # (batch_size, context_size, action_size)
        bridge_q = self.head_bridge_q(base_output)  # (batch_size, context_size, location_size)
        bridge_k = self.head_bridge_k(base_output)  # (batch_size, context_size, location_size)
        link_logits = self.head_bridge(base_output).squeeze(-1)  # (batch_size, context_size)
        write_value_params = self.head_write_value(base_output)  # (batch_size, context_size, 2 * node_dim)

        values = self.head_value(base_output)  # (batch_size, context_size, value_size)
        if self.value_size == 1:
            values = values.squeeze(-1)  # (batch_size, context_size)

        return action_logits, bridge_q, bridge_k, link_logits, write_value_params, values


    def _bridge_logits(self, q, k, offset=0):
        # q: (B, Tq, d) queries for window steps offset .. offset + Tq - 1; k: (B, Tk, d) keys for window steps 0 .. Tk - 1
        # returns logits (B, Tq, Tk) over which window step to bridge to, and the causal mask (step i sees keys j <= i)
        logits = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(self.location_size)
        q_pos = (torch.arange(q.size(1), device=q.device)[:, None] + offset).clamp(min=0)
        k_pos = torch.arange(k.size(1), device=q.device)[None, :]
        mask = (k_pos <= q_pos).unsqueeze(0).expand(q.size(0), -1, -1)
        return logits, mask


    def reset_bridge_cache(self):
        self._bridge_key_cache = None


    def train(self, mode=True):
        # entering train mode means learn() is about to change the parameters and the window restarts afterwards
        # (graph memory reset_timestamp), so cached keys are stale either way
        if mode:
            self.reset_bridge_cache()
        return super().train(mode)


    def _valid_mask(self, valid_actions):
        if valid_actions is None:
            return None
        if isinstance(valid_actions, np.ndarray):
            valid_actions = torch.tensor(valid_actions, dtype=torch.bool)
        return valid_actions.to(self.device)


    def _write_value_concentrations(self, write_value_params):
        alpha_raw, beta_raw = write_value_params.chunk(2, dim=-1)
        # concentrations >= 1 keep each Beta unimodal with bounded density
        alpha = 1.0 + nn.functional.softplus(alpha_raw)
        beta = 1.0 + nn.functional.softplus(beta_raw)
        return alpha, beta


    def _write_value_distribution(self, write_value_params):
        alpha, beta = self._write_value_concentrations(write_value_params)
        return Independent(Beta(alpha, beta), 1)  # joint over node_dim


    def _graph_summary(self, graph, valid):
        # permutation-invariant over neighbor slots: [head token, masked-mean neighbor token, occupancy fraction]
        tokens = self.aux_node_embedding(graph)  # (..., internal_state_size, hidden_size)
        neighbor_valid = valid[..., 1:, :]
        count = neighbor_valid.sum(dim=-2)  # (..., 1)
        neighbors = (tokens[..., 1:, :] * neighbor_valid).sum(dim=-2) / count.clamp(min=1.0)
        occupancy = count / max(self.internal_state_size - 1, 1)
        return self.aux_delta_net(torch.cat([tokens[..., 0, :], neighbors, occupancy], dim=-1))


    def _graph_delta(self, state, next_state):
        # state, next_state: (..., internal_state_size, node_dim) graph contexts before/after action t -> (..., hidden_size)
        valid = (state != self.empty_value).any(dim=-1, keepdim=True).float()
        next_valid = (next_state != self.empty_value).any(dim=-1, keepdim=True).float()
        return self._graph_summary(next_state, next_valid) - self._graph_summary(state, valid)


    def _aux_loss(self, context, write_value_params, selected_action):
        """
        Consistency rule: obs_t + delta(graph)_t = obs_{t+1}.
        A FiLM U-Net convolutionally transforms obs_t into obs_{t+1} logits, conditioned only on delta(graph)_t
        (see _graph_delta), so a_t can only reach the prediction through the graph.
        The graph memory is not differentiable; where the write landed on the next head, the head slot carries a
        straight-through gradient into E[write_value_t] (forward value unchanged).

        context: (batch, context_size + 1, packed_context_size); write_value_params: (batch, context_size, 2 * node_dim)
        selected_action: (batch, context_size, packed_action_size)
        returns (batch, context_size)
        """
        context = context.float()
        batch_size, context_size = write_value_params.shape[:2]
        graph = context[:, :, self.internal_state_pos:self.goal_pos].reshape(
            batch_size, context_size + 1, self.internal_state_size, self.node_dim
        )
        state, next_state = graph[:, :-1], graph[:, 1:]

        alpha, beta = self._write_value_concentrations(write_value_params)
        write_mean = alpha / (alpha + beta)
        written = selected_action[:, :, 3:3 + self.node_dim].float()
        next_head = next_state[:, :, 0]
        landed = ((next_head - written).abs() <= self.aux_write_tol).all(dim=-1, keepdim=True)
        next_head = next_head + torch.where(landed, write_mean - write_mean.detach(), torch.zeros_like(write_mean))
        next_state = torch.cat([next_head.unsqueeze(2), next_state[:, :, 1:]], dim=2)

        cond = self._graph_delta(state, next_state).reshape(batch_size * context_size, self.hidden_size)
        grid_shape = (batch_size * context_size, self.height, self.width, self.channel)
        obs = context[:, :-1, self.obs_pos:].reshape(grid_shape)
        obs_logits = self.aux_unet(obs, cond)  # (batch * context_size, height, width, sum(obs_classes))
        obs_target = context[:, 1:, self.obs_pos:].reshape(grid_shape).long()

        # flat (N, C) cross entropy: the spatial nll variant has no deterministic CUDA kernel
        obs_loss = 0.0
        for c, logits in enumerate(obs_logits.split(list(self.obs_classes), dim=-1)):
            target = obs_target[..., c].clamp(0, self.obs_classes[c] - 1)
            obs_loss = obs_loss + nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none"
            ).view(target.shape).mean(dim=(-2, -1))  # mean over cells
        return obs_loss.view(batch_size, context_size)


    def _distributions(self, action_logits, bridge_q, bridge_k, link_logits, write_value_params, valid_actions):
        bridge_logits, bridge_mask = self._bridge_logits(bridge_q, bridge_k)
        return (
            Categorical_With_Mask(logits=action_logits, mask=self._valid_mask(valid_actions)),
            Categorical_With_Mask(logits=bridge_logits, mask=bridge_mask),
            Bernoulli(logits=link_logits),
            self._write_value_distribution(write_value_params),
        )


    def _log_prob_and_entropy(self, dists, selected_action):
        probs_flag, probs_bridge, probs_link, probs_write_value = dists

        action_flag = selected_action[:, :, 0]
        action_bridge_step = selected_action[:, :, 1]
        action_link = selected_action[:, :, 2]
        action_write_value = selected_action[:, :, 3:3 + self.node_dim].clamp(self.write_eps, 1.0 - self.write_eps)

        # create-vs-link only matters on NAVIGATE; the link target only when NAVIGATE chose to link
        is_navigate = action_flag.long() == self.navigate_flag
        is_link = is_navigate & (action_link > 0.5)

        log_prob_flag = probs_flag.log_prob(action_flag)
        zeros = torch.zeros_like(log_prob_flag)
        log_prob_bridge = torch.where(is_link, probs_bridge.log_prob(action_bridge_step), zeros)
        log_prob_link = torch.where(is_navigate, probs_link.log_prob(action_link), zeros)
        log_prob_write_value = probs_write_value.log_prob(action_write_value)

        entropy_flag = probs_flag.entropy()
        entropy_bridge = torch.where(is_link, probs_bridge.entropy(), zeros)
        entropy_link = torch.where(is_navigate, probs_link.entropy(), zeros)
        # mean, not sum: keeps the regularizer scale independent of node_dim (log-prob stays joint)
        entropy_write_value = probs_write_value.base_dist.entropy().mean(dim=-1)

        # layout [flag, edge_1, edge_2, write] matches the packed action
        return torch.stack([
            log_prob_flag, log_prob_bridge, log_prob_link, log_prob_write_value
        ], dim=-1), torch.stack([
            entropy_flag, entropy_bridge, entropy_link, entropy_write_value
        ], dim=-1)


    @torch.no_grad()
    def get_action(self, context, valid_actions=None):
        """
        context: (batch, T, packed_context_size); context[:, -1] is the new step.
        Each call is one new window step: its key is appended to a rollout cache, and the bridge candidates are
        every cached step of the current window (the window restarts when the core enters train mode, i.e. at learn()).
        The cached keys equal those recomputed at training time since parameters only change at learn().

        returns action (batch, T, packed_action_size) = [flag, edge_1 (window step), edge_2 (0: create, 1: link), write_value (node_dim)];
        the graph memory resolves the window step to a node through its own head track.
        """
        if isinstance(context, np.ndarray):
            # float: node values are continuous and would be truncated by a long cast
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        batch_size = context.size(0)
        context_size = context.size(1)

        action_logits, bridge_q, bridge_k, link_logits, write_value_params, _ = self.compute(context)

        cache = self._bridge_key_cache
        if cache is not None and cache.size(0) != batch_size:
            raise RuntimeError(f"bridge key cache batch {cache.size(0)} does not match context batch {batch_size}; call reset_bridge_cache()")
        new_key = bridge_k[:, -1:]
        bridge_k = new_key if cache is None else torch.cat([cache, new_key], dim=1)  # (batch, window_size, location_size)
        self._bridge_key_cache = bridge_k

        bridge_logits, bridge_mask = self._bridge_logits(bridge_q, bridge_k, offset=bridge_k.size(1) - context_size)

        action_flag = Categorical_With_Mask(logits=action_logits, mask=self._valid_mask(valid_actions)).sample()
        bridge_step = Categorical_With_Mask(logits=bridge_logits, mask=bridge_mask).sample()  # (batch, T) window step
        link = Bernoulli(logits=link_logits).sample()
        write_value = self._write_value_distribution(write_value_params).sample().clamp(self.write_eps, 1.0 - self.write_eps)  # (batch, T, node_dim)

        action = np.concatenate([
            action_flag.unsqueeze(-1).cpu().numpy(),
            bridge_step.unsqueeze(-1).cpu().numpy(),
            link.unsqueeze(-1).cpu().numpy(),
            write_value.cpu().numpy()
        ], axis=-1)

        return action.astype(float)


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, packed_action_size); flag stays (batch, 1) so flag[i][0] is the env action
        flag = packed_action[:, 0:1].astype(int)
        position = packed_action[:, 1:1 + self.position_size].astype(float)
        write_value = packed_action[:, 1 + self.position_size:].astype(float)
        return flag, position, write_value


    def pack_context(self, b_reward, b_flag, b_position, b_content):
        batch_size = b_reward.shape[0]
        return np.concatenate([
            np.reshape(b_reward, (batch_size, 1)),
            np.reshape(b_flag, (batch_size, 1)),
            b_position,
            b_content
        ], axis=-1).astype(float)


    def _log_prob_with_value(self, context, selected_action, valid_actions):
        # context: (batch, context_size + 1, packed_context_size) tensor; also returns the write head params of the
        # first context_size steps so the aux loss can reuse this forward pass
        action_logits, bridge_q, bridge_k, link_logits, write_value_params, values = self.compute(context)

        # drop the last step's outputs: it is the observation after the last action
        # (step i only attends to keys j <= i, so dropping the last key changes nothing for the remaining rows)
        write_value_params = write_value_params[:, :-1]
        dists = self._distributions(
            action_logits[:, :-1], 
            bridge_q[:, :-1], bridge_k[:, :-1],
            link_logits[:, :-1],
            write_value_params,
            valid_actions,
        )
        log_probs, entropy = self._log_prob_and_entropy(dists, selected_action)
        # but maintain values for all contexts, since we need the bootstrap value for the last context
        return log_probs, entropy, values, write_value_params


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        # Return log prob will only have context_size, but value will have context_size + 1
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        log_probs, entropy, values, _ = self._log_prob_with_value(context, selected_action, valid_actions)
        return log_probs, entropy, values


    def get_log_prob_with_value_aux(self, context, selected_action, valid_actions=None):
        # get_log_probability_with_value plus the (batch, context_size) aux loss (None unless use_aux), see _aux_loss
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(selected_action, np.ndarray):
            selected_action = torch.tensor(selected_action, dtype=torch.float32).to(self.device)

        log_probs, entropy, values, write_value_params = self._log_prob_with_value(context, selected_action, valid_actions)
        aux_loss = self._aux_loss(context, write_value_params, selected_action) if self.use_aux else None
        return log_probs, entropy, values, aux_loss
