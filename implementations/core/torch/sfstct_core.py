import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np

from interfaces.core import Core
from .base import Multilayer_Relu, init_weights
from .sfstct import SpatialEncoder, TemporalEncoder


class SF_STCT_Core(Core, nn.Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, heads, layers):
        super().__init__()
        # content_size = channels x height x width

        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height

        self.width = width
        self.height = height
        self.channel = channel

        self.hidden_size = hidden_size

        # Configuration matches the report specification
        self.spatial_encoder = SpatialEncoder(
            img_size=width, patch_size=4, in_chans=channel,
            vector_dim=1 + position_size, 
            embed_dim=hidden_size, depth=layers
        )
        
        self.temporal_encoder = TemporalEncoder(
            embed_dim=hidden_size, depth=layers
        )

        # Prediction Heads [10]
        # Decoupling MLP before projection is best practice
        self.head_x = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, width) # width classes
        )
        self.head_y = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, height) # height classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_size)   # action_size classes
        )
        self.head_content = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.content_size), # content_size classes
            nn.Sigmoid()
        )
        self.head_flag = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 2)   # 2 classes
        )
        self.head_value = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1)   # Regression output
        )

        self.value_logstd = nn.Parameter(torch.zeros(1, 1))

        self.position_step = Multilayer_Relu(position_size + action_size + width + height, position_size, hidden_size, 2)

        self.reset_parameters()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.spatial_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.head_x.apply(init_weights)
        self.head_y.apply(init_weights)
        self.head_action.apply(init_weights)
        self.head_content.apply(init_weights)
        self.head_flag.apply(init_weights)
        self.head_value.apply(init_weights)

        # make sure value_logstd is initialized to 0
        nn.init.constant_(self.value_logstd, 0.0)

        self.position_step.reset_parameters()


    def __compute(self, context):
        # x has shape (batch, context_size, 1 + position_size + content_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, 1 + self.position_size:]
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        non_image_part = context[:, :, :1 + self.position_size]

        # 1. Spatial Processing (Independent per frame)
        spatial_feats = self.spatial_encoder(image_part, non_image_part) # Output: (B, T, hidden_size)
        
        # 2. Temporal Processing (Across frames with causal mask)
        temporal_feat = self.temporal_encoder(spatial_feats) # Output: (B, T, hidden_size)
        
        return temporal_feat
    

    def get_latest_value(self, context):
        temporal_feat = self.__compute(context)
        logits_value = self.head_value(temporal_feat)    # (B, T, 1)
        return logits_value[:, -1]
    

    def get_action_and_value(self, context, action=None):
        batch_size = context.size(0)
        temporal_feat = self.__compute(context)

        logits_x = self.head_x(temporal_feat)      # (B, T, 64)
        logits_y = self.head_y(temporal_feat)      # (B, T, 64)
        logits_action = self.head_action(temporal_feat) # (B, T, 6)
        pprobs_content = self.head_content(temporal_feat) # (B, T, content_size)
        logits_flag = self.head_flag(temporal_feat)    # (B, T, 2)
        value = self.head_value(temporal_feat)    # (B, T, 1)

        props_x = Categorical(logits=logits_x)
        props_y = Categorical(logits=logits_y)
        props_action = Categorical(logits=logits_action)
        props_content = Bernoulli(probs=pprobs_content)
        props_flag = Categorical(logits=logits_flag)

        if action is None:
            action_flag = props_flag.sample()
            action_action = props_action.sample()
            action_x = props_x.sample()
            action_y = props_y.sample()
            action_content = props_content.sample()

            action = torch.cat([
                action_flag.unsqueeze(-1),
                action_action.unsqueeze(-1),
                action_x.unsqueeze(-1),
                action_y.unsqueeze(-1),
                action_content
            ], dim=-1)
        else:
            action_flag = action[:, :, 0]
            action_action = action[:, :, 1]
            action_x = action[:, :, 2]
            action_y = action[:, :, 3]
            action_content = action[:, :, 4:]

        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = props_y.log_prob(action_y)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_content = props_content.log_prob(action_content).sum(-1)
        log_prob_flag = props_flag.log_prob(action_flag)
        batch_log_prob = log_prob_x + log_prob_y + log_prob_action + log_prob_content + log_prob_flag

        entropy_x = props_x.entropy()
        entropy_y = props_y.entropy()
        entropy_action = props_action.entropy()
        entropy_content = props_content.entropy().sum(-1)
        entropy_flag = props_flag.entropy()
        batch_entropy = entropy_x + entropy_y + entropy_action + entropy_content + entropy_flag

        # collapse last dimension
        batch_log_prob = torch.reshape(batch_log_prob, (batch_size, -1))
        batch_entropy = torch.reshape(batch_entropy, (batch_size, -1))
        batch_value = torch.reshape(value, (batch_size, -1))

        return action, batch_log_prob, batch_entropy, batch_value


    def unpack_action(self, packed_action, context):
        # packed_action has shape (batch, 1 + 3 + content_size)
        # context has shape (batch, context_size, 1 + position_size + content_size)
        # return ext_flag, action_data, position, content

        ext_part = packed_action[:, 0]
        action = packed_action[:, 1]
        x = packed_action[:, 2]
        y = packed_action[:, 3]
        content = packed_action[:, 4:]

        last_position = context[:, -1, 1:1 + self.position_size]
        # make one hot encoding for action, x, y
        action_onehot = torch.nn.functional.one_hot(action.long(), num_classes=self.action_size).float()
        x_onehot = torch.nn.functional.one_hot(x.long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(y.long(), num_classes=self.height).float()
        position = self.position_step(torch.concat([last_position, action_onehot, x_onehot, y_onehot], dim=-1))

        return ext_part, action, x, y, position, content