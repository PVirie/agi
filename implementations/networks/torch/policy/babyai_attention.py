import torch
import torch.nn as nn
import numpy as np

from interfaces.network import Actor_Network, Policy_Value_Network
from implementations.networks.torch.components.base import Categorical_With_Mask, init_weights
from implementations.networks.torch.components.std_conv import FiLMMiniGridCNN
from implementations.networks.torch.components.rope import RoPEDecoderOnly
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core, Actor_Network, Policy_Value_Network):

    def __init__(self, 
                 ext_action_size,
                 goal_size, inventory_size,
                 width, height, channel,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers, 
                 history_steps=32,
                 cnn_channels=128, cnn_blocks=2,
                 value_size=1,
                 device=None, persistence_path=None, first_load_path=None):
        """
        Benchmark policy for agents/model_base.py: no memory operations, it decides from its recent step history.
        Pipeline per step:
            instruction, inventory -> embedding -> transformer -> CLS --+
            obs -> FiLM CNN (conditioned on the CLS) ------------------+-> linear (+ prev ext action, prev reward) -> token
        then over steps:
            tokens -> causal transformer -> hidden
            hidden -> ext head, value head
        Each attention layer looks ceil(history_steps / num_layers) steps back, so a step's output depends on exactly
        its last self.history_steps (>= history_steps) steps. Acting on the last self.history_steps + 1 contexts
        (collectors with max_history=self.history_steps) therefore reproduces what learning computes over the whole rollout.
        Attention never crosses an episode start (b_episode_start in pack_context): a step sees only its own episode's steps.
        Trained by ppo_graph (value_size=1) or ppo_graph_dual (value_size=2).
        """
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="babyai_attention_core", device=device, persistence_path=persistence_path)
        self.device = device

        layer_history_steps = -(-history_steps // len(layers))
        # receptive field of the stacked layers
        self.history_steps = layer_history_steps * len(layers)

        self.ext_action_size = ext_action_size
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

        self.content_size = goal_size + inventory_size + width * height * channel
        # action: [ext]; context: [reward, ext, episode start, content]
        self.packed_action_size = 1
        self.packed_context_size = 1 + 1 + 1 + self.content_size

        self.reward_pos = 0
        self.ext_pos = 1
        self.start_pos = 2
        self.goal_pos = 3
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        self.goal_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for goal tokens
        self.inv_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for inventory tokens

        self.goal_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8,
            num_layers=4,
            dropout=0.01,
            max_len=goal_size + inventory_size,
            use_pe=True,  # mission semantics depend on word order (attribute binding, "then"/"after")
            pooling="cls",  # mean pooling dilutes order-dependent structure ("A then B" vs "B then A")
        )

        # vision conditioned via FiLM on the instruction features
        self.conv_layers = FiLMMiniGridCNN(
            output_dims=hidden_size, 
            cond_dims=hidden_size,
            input_channels=channel, 
            width=width, height=height,
            vocab_size=256, embedding_dim=8,
            channels=cnn_channels, num_blocks=cnn_blocks
        )

        # previous ext action (one-hot) and previous reward ride along with the step's features
        self.latent_projection = nn.Sequential(
            nn.Linear(2 * hidden_size + ext_action_size + 1, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # one causal attention layer per entry of layers; no dropout so acting (eval) and learning (train) give the same log-probs
        self.attention_core = RoPEDecoderOnly(
            d_model=hidden_size,
            num_heads=4,
            num_layers=len(layers),
            d_ff=2 * hidden_size,
            dropout=0.0,
            history_steps=layer_history_steps,
        )

        self.head_ext = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, ext_action_size)
        )

        self.head_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, value_size)
        )

        # acting only: the last window's contexts and their step tokens, see _cached_tokens
        self._token_cache = None

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.goal_embedding.reset_parameters()
        self.inv_embedding.reset_parameters()

        self.goal_feature_extraction.reset_parameters()
        self.conv_layers.reset_parameters()
        self.latent_projection.apply(init_weights)
        self.attention_core.reset_parameters()

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.head_ext.apply(init_actor_weights)

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.head_value.apply(init_value_weights)


    def encode(self, prev_reward, prev_ext, goal, inv, obs):
        # prev_reward: (batch, context_size); prev_ext: (batch, context_size) ids; goal, inv: token ids; obs: flattened grid
        # returns the per-step token (batch, context_size, hidden_size); every step is independent, so time is batched
        batch_size, context_size = goal.shape[:2]
        n = batch_size * context_size

        tokens = torch.concat([
            self.goal_embedding(goal.long()),
            self.inv_embedding(inv.long())
        ], dim=2)  # (batch_size, context_size, goal_size + inventory_size, embedding_dim)
        padding_mask = get_padding_mask(torch.concat([goal, inv], dim=-1), self.pad_token_id)
        instruction_summary = self.goal_feature_extraction(tokens, padding_mask)  # (batch_size, context_size, hidden_size)

        obs = torch.reshape(obs, (n, self.height, self.width, self.channel))
        obs_summary = self.conv_layers(obs, instruction_summary.reshape(n, self.hidden_size))  # (n, hidden_size)

        prev_ext_one_hot = nn.functional.one_hot(prev_ext.reshape(n), num_classes=self.ext_action_size).float()
        latent = self.latent_projection(torch.concat([
            instruction_summary.reshape(n, self.hidden_size), obs_summary, prev_ext_one_hot, prev_reward.reshape(n, 1)
        ], dim=-1))
        return latent.view(batch_size, context_size, self.hidden_size)


    def reset_token_cache(self):
        self._token_cache = None


    def train(self, mode=True):
        # entering train mode means learn() is about to change the parameters, so cached step tokens are stale
        if mode:
            self.reset_token_cache()
        return super().train(mode)


    def _to_context(self, context):
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        return context


    def _encode_context(self, context):
        # (batch, context_size, packed_context_size) -> step tokens (batch, context_size, hidden_size)
        return self.encode(
            context[:, :, self.reward_pos],
            context[:, :, self.ext_pos].long(),
            context[:, :, self.goal_pos: self.inv_pos],
            context[:, :, self.inv_pos: self.obs_pos],
            context[:, :, self.obs_pos:],
        )


    def _cached_tokens(self, context):
        # acting slides its window one step per call, and step tokens depend on their own step only,
        # so when the cached window's last steps equal this window's past steps only the newest step is encoded.
        # Any mismatch (first call, episode reset of the agent, a different caller) re-encodes the whole window
        cache, window = self._token_cache, context.shape[1]
        if cache is not None and window > 1:
            cached_context, cached_tokens = cache
            past = window - 1
            if (cached_context.shape[0] == context.shape[0] and cached_context.shape[1] >= past
                    and torch.equal(cached_context[:, -past:], context[:, :-1])):
                tokens = torch.concat([cached_tokens[:, -past:], self._encode_context(context[:, -1:])], dim=1)
                self._token_cache = (context, tokens)
                return tokens
        tokens = self._encode_context(context)
        self._token_cache = (context, tokens)
        return tokens


    def compute(self, context, tokens=None):
        # context has shape (batch, context_size, self.packed_context_size); tokens: precomputed step tokens or None
        # returns ext logits (batch, context_size, ext_action_size) and values (batch, context_size[, value_size])
        context = self._to_context(context)
        if tokens is None:
            tokens = self._encode_context(context)  # (batch_size, context_size, hidden_size)

        # steps attend within their own episode only; numbering episodes from the context's start keeps "same episode" unchanged
        # when acting cuts the context to its last window. A step is always in its own episode, so no attention row is fully masked
        episode = torch.cumsum((context[:, :, self.start_pos] > 0.5).long(), dim=1)  # (batch_size, context_size)
        same_episode = episode[:, :, None] == episode[:, None, :]  # (batch_size, context_size, context_size)
        episode_mask = torch.zeros(same_episode.shape, dtype=context.dtype, device=context.device).masked_fill(~same_episode, float('-inf'))

        hidden = self.attention_core(tokens, pad_mask=episode_mask[:, None])  # (batch_size, context_size, hidden_size)

        ext_logits = self.head_ext(hidden)  # (batch_size, context_size, ext_action_size)
        values = self.head_value(hidden)  # (batch_size, context_size, value_size)
        if self.value_size == 1:
            values = values.squeeze(-1)  # (batch_size, context_size)
        return ext_logits, values


    def _to_mask(self, valid_actions):
        if valid_actions is None:
            return None
        if isinstance(valid_actions, np.ndarray):
            valid_actions = torch.tensor(valid_actions, dtype=torch.bool)
        # valid_actions has shape (batch, context_size, ext_action_size)
        return valid_actions.to(self.device)


    def get_action(self, context, valid_actions=None):
        with torch.no_grad():
            context = self._to_context(context)
            # the cache is only valid while the parameters are fixed, i.e. in eval mode
            tokens = None if self.training else self._cached_tokens(context)
            ext_logits, _ = self.compute(context, tokens)
            ext_action = Categorical_With_Mask(logits=ext_logits, mask=self._to_mask(valid_actions)).sample()
        return ext_action.unsqueeze(-1).cpu().numpy().astype(float)  # (batch, context_size, packed_action_size)


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        # returns the ext action's log-prob and entropy (batch, context_size), values (batch, context_size + 1[, value_size])
        if isinstance(selected_action, np.ndarray):
            selected_action = torch.tensor(selected_action, dtype=torch.float32).to(self.device)
        ext_logits, values = self.compute(context)
        # values keep all contexts: the last one is the bootstrap value
        # drop the last context for everything else: it is the observation after the last action
        probs_ext = Categorical_With_Mask(logits=ext_logits[:, :-1, :], mask=self._to_mask(valid_actions))
        return probs_ext.log_prob(selected_action[:, :, 0]), probs_ext.entropy(), values


    def get_log_prob_with_value_aux(self, context, selected_action, valid_actions=None):
        log_probs, entropy, values = self.get_log_probability_with_value(context, selected_action, valid_actions)
        return log_probs, entropy, values, None  # no auxiliary loss


    def unpack_action(self, packed_action):
        # packed_action (batch, packed_action_size): [ext]
        # returns ext action (batch, 1) and the action content (batch, 0): this policy emits none
        return packed_action[:, 0:1].astype(int), packed_action[:, 1:].astype(float)


    def pack_context(self, b_reward=None, b_ext=None, b_episode_start=None, b_content=None):
        # b_episode_start: (batch,) bool, True where the context holds a new episode's first observation
        batch_size = next(b.shape[0] for b in (b_reward, b_ext, b_episode_start, b_content) if b is not None)
        if b_reward is None:
            b_reward = np.zeros((batch_size,), dtype=float)
        if b_ext is None:
            b_ext = np.zeros((batch_size, 1), dtype=int)
        if b_episode_start is None:
            b_episode_start = np.zeros((batch_size,), dtype=bool)
        if b_content is None:
            b_content = np.zeros((batch_size, self.content_size), dtype=float)
        return np.concatenate([
            np.reshape(b_reward, (batch_size, 1)),
            np.reshape(b_ext, (batch_size, 1)),
            np.reshape(b_episode_start, (batch_size, 1)),
            np.reshape(b_content, (batch_size, self.content_size))
        ], axis=-1).astype(float)


if __name__ == "__main__":
    from implementations.collectors.states import State_Sequence

    torch.manual_seed(0)
    np.random.seed(0)

    B, T, A = 4, 12, 7
    core = Policy_Core(
        ext_action_size=A,
        goal_size=5, inventory_size=3,
        width=7, height=7, channel=3,
        dict_size=32, embedding_dim=8, pad_token_id=0,
        hidden_size=32, layers=[1, 1],
        history_steps=3,
        cnn_channels=16, cnn_blocks=1,
        device=torch.device("cpu"),
    )
    H = core.history_steps
    assert H == 4, "3 steps over 2 layers rounds up to 2 steps per layer"

    def random_content():
        return np.concatenate([
            np.random.randint(0, 32, (B, 5 + 3)),
            np.random.randint(0, 10, (B, 7 * 7 * 3)),
        ], axis=-1)

    valid = np.random.rand(B, T + 1, A) > 0.3
    valid[:, :, 0] = True

    # act one step at a time as model_base does, through a collector holding the last H + 1 contexts
    collector = State_Sequence(max_history=H)
    episode_start = np.zeros((B, T + 1), dtype=bool)
    episode_start[:, 0] = True
    episode_start[0, 8] = True  # batch 0 starts a new episode mid-sequence
    last_action = np.zeros((B, core.packed_action_size))
    actions = []
    for t in range(T + 1):
        ext_action, content = core.unpack_action(last_action)
        assert ext_action.shape == (B, 1) and content.shape == (B, 0)
        collector.append(core.pack_context(b_reward=np.random.rand(B), b_ext=ext_action, b_episode_start=episode_start[:, t], b_content=random_content()))
        packed = core.get_action(collector.get_last_batch(), valid[:, max(0, t - H): t + 1])
        assert packed.shape == (B, min(t, H) + 1, core.packed_action_size)
        last_action = packed[:, -1]
        actions.append(last_action)
    contexts = torch.tensor(collector.make_batch(), dtype=torch.float32)
    actions = np.stack(actions, axis=1)  # actions[:, t] was taken after contexts[:, t]
    assert valid[np.arange(B)[:, None], np.arange(T + 1), actions[..., 0].astype(int)].all(), "sampled ext actions must be valid"

    # learning over the whole sequence reproduces acting on each step's last H + 1 contexts
    with torch.no_grad():
        full_logits, full_values = core.compute(contexts)
        for t in range(T + 1):
            window_logits, window_values = core.compute(contexts[:, max(0, t - H): t + 1])
            assert torch.allclose(full_logits[:, t], window_logits[:, -1], atol=1e-5), f"windowed logits differ from acting at step {t}"
            assert torch.allclose(full_values[:, t], window_values[:, -1], atol=1e-5)

        # step t sees exactly steps t - H .. t
        perturbed = contexts.clone()
        perturbed[:, 2, core.obs_pos:] = (perturbed[:, 2, core.obs_pos:] + 1) % 10
        changed = core.compute(perturbed)[0]
        assert torch.allclose(changed[:, :2], full_logits[:, :2], atol=1e-6), "a step must not see later steps"
        assert not torch.allclose(changed[:, 2 + H], full_logits[:, 2 + H], atol=1e-6), "the oldest step in the window must matter"
        assert torch.allclose(changed[:, 2 + H + 1:], full_logits[:, 2 + H + 1:], atol=1e-6), "a step beyond the window must not matter"

        # attention never crosses an episode start, even within the window
        perturbed = contexts.clone()
        perturbed[:, 7, core.obs_pos:] = (perturbed[:, 7, core.obs_pos:] + 1) % 10
        changed = core.compute(perturbed)[0]
        assert torch.allclose(changed[0, 8:], full_logits[0, 8:], atol=1e-6), "a new episode must not see the previous one"
        assert not torch.allclose(changed[1, 8], full_logits[1, 8], atol=1e-6), "without a start, the previous step must matter"

    # ppo_graph: context_size + 1 contexts, values keep the bootstrap step, no aux loss
    log_probs, entropy, values, aux = core.get_log_prob_with_value_aux(contexts, actions[:, :T], valid[:, :T])
    assert log_probs.shape == (B, T) and entropy.shape == (B, T) and values.shape == (B, T + 1) and aux is None
    assert torch.isfinite(log_probs).all(), "acted ext actions must have finite log-probs"

    core.train()
    log_probs, _, values = core.get_log_probability_with_value(contexts, actions[:, :T], valid[:, :T])
    (log_probs.sum() + values.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.attention_core.parameters()), "attention core must train"

    print("All tests passed!")
