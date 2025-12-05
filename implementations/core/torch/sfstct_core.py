import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.core import Core
from .base import Multilayer_Relu, init_weights
from .sfstct import SpatialEncoder, TemporalEncoder


class SF_STCT_Core(Core, nn.Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, heads, layers, device=None, persistence_path=None):
        super().__init__()
        # content_size = channels x height x width

        self.device = device

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

        self.persistence_path = persistence_path
        if self.persistence_path is not None:
            self.load()


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


    def load(self):
        if self.persistence_path is not None:
            try:
                checkpoint = torch.load(f"{self.persistence_path}/core_checkpoint.pth", map_location=self.device)
                self.load_state_dict(checkpoint["model_state_dict"])
                logging.info(f"Core: Loaded checkpoint from {self.persistence_path}/core_checkpoint.pth")
            except FileNotFoundError:
                logging.info(f"Core: No checkpoint found at {self.persistence_path}/core_checkpoint.pth")


    def save(self):
        if self.persistence_path is not None:
            torch.save({
                "model_state_dict": self.state_dict(),
            }, f"{self.persistence_path}/core_checkpoint.pth")
            print("Core: Saved checkpoint to", f"{self.persistence_path}/core_checkpoint.pth")


    def __compute_position(self, last_position, action):
        # last_position has shape (batch, context_size, position_size)
        # action has shape (batch, context_size, 1 + 3 + content_size)

        # make one hot encoding for action, x, y
        action_onehot = torch.nn.functional.one_hot(action[:, :, 1].long(), num_classes=self.action_size).float()
        x_onehot = torch.nn.functional.one_hot(action[:, :, 2].long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(action[:, :, 3].long(), num_classes=self.height).float()
        true_positions = self.position_step(torch.concat([last_position, action_onehot, x_onehot, y_onehot], dim=-1))

        # shift position by one step
        true_positions = torch.cat([last_position[:, :1, :], true_positions[:, :-1, :]], dim=1)

        return true_positions


    def __compute(self, context, action):
        # context has shape (batch, context_size, 1 + position_size + content_size)
        # action has shape (batch, context_size, 1 + 3 + content_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # true_positions has shape (batch, context_size, position_size)
        true_positions = self.__compute_position(context[:, :, 1:1 + self.position_size], action)

        # first slice the image content
        image_content = context[:, :, 1 + self.position_size:]
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        non_image_part = torch.concatenate([context[:, :, :1], true_positions], dim=-1)

        # 1. Spatial Processing (Independent per frame)
        spatial_feats = self.spatial_encoder(image_part, non_image_part) # Output: (B, T, hidden_size)
        
        # 2. Temporal Processing (Across frames with causal mask)
        temporal_feat = self.temporal_encoder(spatial_feats) # Output: (B, T, hidden_size)
        
        return temporal_feat
    

    def get_latest_value(self, context, action):
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
        elif action is None:
            action = torch.zeros((context.size(0), context.size(1), 1 + 3 + self.content_size), dtype=torch.int64).to(self.device)

        with torch.no_grad():
            temporal_feat = self.__compute(context, action)
            logits_value = self.head_value(temporal_feat)    # (B, T, 1)
            
        return logits_value[:, -1, ...].cpu().numpy()
    

    def get_action_and_value(self, context, action, use_action=False, use_grad=True):
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
        elif action is None:
            action = torch.zeros((context.size(0), context.size(1), 1 + 3 + self.content_size), dtype=torch.float32).to(self.device)

        batch_size = context.size(0)
        temporal_feat = self.__compute(context, action)

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

        if use_action:
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

        with torch.no_grad():
            last_position = context[:, :, 1:1 + self.position_size]
            # make one hot encoding for action, x, y
            action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
            x_onehot = torch.nn.functional.one_hot(action_x.long(), num_classes=self.width).float()
            y_onehot = torch.nn.functional.one_hot(action_y.long(), num_classes=self.height).float()
            position = self.position_step(torch.concat([last_position, action_onehot, x_onehot, y_onehot], dim=-1))

        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = props_y.log_prob(action_y)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_content = props_content.log_prob(action_content).sum(-1)
        log_prob_flag = props_flag.log_prob(action_flag)
        batch_log_prob = log_prob_x + log_prob_y + log_prob_action + log_prob_content + log_prob_flag
        
        # clamp batch_log_prob to avoid too large negatives
        batch_log_prob = torch.clamp(batch_log_prob, min=-10, max=0)

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

        action = action.cpu().numpy().astype(int)
        position = position.cpu().numpy().astype(float)

        if not use_grad:
            batch_log_prob = batch_log_prob.detach().cpu().numpy()
            batch_entropy = batch_entropy.detach().cpu().numpy()
            batch_value = batch_value.detach().cpu().numpy()

        return action, position, batch_log_prob, batch_entropy, batch_value


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, 1 + 3 + content_size)
        # return ext_flag, action_data... , content

        ext_part = packed_action[:, 0].astype(float)
        action = packed_action[:, 1].astype(int)
        x = packed_action[:, 2].astype(int)
        y = packed_action[:, 3].astype(int)
        content = packed_action[:, 4:].astype(float)

        return ext_part, action, x, y, content