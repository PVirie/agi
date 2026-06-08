import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import ImpalaCNN
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core):

    def __init__(self, 
                 int_action_size, ext_action_size, 
                 goal_size, inventory_size,
                 dict_size, embedding_dim, pad_token_id,
                 width, height, channel,
                 state_size,
                 hidden_size, layers, 
                 history_steps=0, max_temporal_len=32, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="base_token_image_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.width = width
        self.height = height
        self.channel = channel

        self.goal_size = goal_size
        self.inventory_size = inventory_size
        self.obs_size = width * height * channel
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.position_size = state_size
        self.content_size = goal_size + inventory_size + self.obs_size
        self.packed_action_size = 1 + 1 + self.position_size + self.content_size
        self.packed_context_size = 1 + 1 + 1 + self.position_size + self.content_size

        self.goal_pos = 1 + 1 + 1 + state_size
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        self.pad_token_id = pad_token_id
        self.goal_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for goal tokens
        self.goal_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8, 
            num_layers=2, 
            max_len=goal_size
        )

        self.image_embedding = nn.Embedding(256, 4)  # for image pixels, shared across channels
        self.feature_channel = self.channel * 4
        self.conv_layers = ImpalaCNN(
            output_dims=hidden_size, 
            input_channels=self.feature_channel, 
            width=width, height=height,
            depths=layers
        )

        # goal, inv, obs -> hidden
        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=hidden_size + inventory_size * embedding_dim + hidden_size, 
            hidden_dims=hidden_size, 
            layers=[hidden_size for _ in layers]
        )

        self.head_int = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int_action_size)   # int_action_size classes
        )
        self.head_ext = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, ext_action_size)   # ext_action_size classes
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.goal_embedding.reset_parameters()
        self.image_embedding.reset_parameters()
        self.conv_layers.reset_parameters()
        self.goal_feature_extraction.reset_parameters()

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

        self.head_int.apply(init_actor_weights)
        self.head_ext.apply(init_actor_weights)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.ext_action_size).float()
        goal = context[:, :, self.goal_pos: (self.goal_pos + self.goal_size)]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: (self.inv_pos + self.inventory_size)]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos: ]  # (batch_size, context_size, obs_size)

        goal_padding_mask = get_padding_mask(goal, self.pad_token_id)  # (batch_size, context_size, goal_size)
        goal_embedded = self.goal_embedding(goal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        goal_features = self.goal_feature_extraction(goal_embedded, goal_padding_mask)  # (batch_size, context_size, hidden_size)
        
        inv_embedded = self.goal_embedding(inv.long())  # (batch_size, context_size, inventory_size, inv_size * embedding_dim)
        inv_embedded = inv_embedded.view(batch_size, context_size, -1)  # (batch_size, context_size, inventory_size * embedding_dim)
        
        obs_embedded = self.image_embedding(obs.long())  # (batch_size, context_size, obs_size, embedding_dim)
        obs_features = torch.reshape(obs_embedded, (batch_size * context_size, self.height, self.width, self.feature_channel))  # (batch_size * context_size, height, width, channel * embedding_dim)
        obs_features = obs_features.permute(0, 3, 1, 2)  # (batch_size * context_size, channel * embedding_dim, height, width)
        obs_features = self.conv_layers(obs_features)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size) # (batch_size, context_size, hidden_size)

        # Base flow

        base_input = torch.concat([goal_features, inv_embedded, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        base_output = self.backbone(base_input)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(base_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(base_output)  # (batch_size, context_size, ext_action_size)

        return int_logits, ext_logits
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)

        batch_size = context.size(0)
        context_size = context.size(1)

        action_int = probs_int.sample()
        action_ext = probs_ext.sample()
        action_position = np.zeros((batch_size, context_size, self.position_size), dtype=np.float32)
        action_content = np.zeros((batch_size, context_size, self.content_size), dtype=np.float32)

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            action_position,
            action_content
        ], axis=-1)

        return action.astype(float)
    

    def get_log_probability(self, context, selected_action, valid_actions=None):
        # context has shape (batch, context_size, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext
        ], dim=-1)


    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        context_full = context
        context = context_full[:, :-1, :]  # remove last context for computing logprob

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext
        ], dim=-1), None  # no aux loss for now


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return int_action, ext_action, position

        int_part = packed_action[:, 0].astype(int)
        ext_part = packed_action[:, 1:2].astype(int)
        position = packed_action[:, 2:2 + self.position_size].astype(float)
        content = packed_action[:, 2 + self.position_size:].astype(float)

        return int_part, ext_part, position, content
    

    def pack_context(self, b_reward=None, b_int=None, b_ext=None, b_position=None, b_content=None):
        # b_xxx has shape (batch, ...)
        # return packed_action_seq of shape (batch, self.packed_action_size) of type int
        # replace none with zeros

        batch_size = None
        if b_reward is not None:
            batch_size = b_reward.shape[0]
        elif b_int is not None:
            batch_size = b_int.shape[0]
        elif b_ext is not None:
            batch_size = b_ext.shape[0]
        elif b_position is not None:
            batch_size = b_position.shape[0]
        elif b_content is not None:
            batch_size = b_content.shape[0]
        else:
            raise ValueError("At least one of b_reward, b_content must be provided")
        
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

        packed_context = np.concatenate([
            np.reshape(b_reward, (batch_size, 1)),
            np.reshape(b_int, (batch_size, 1)),
            b_ext,
            b_position,
            b_content
        ], axis=-1).astype(float)

        return packed_context

