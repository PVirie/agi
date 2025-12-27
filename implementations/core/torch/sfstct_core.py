import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.core import Core
from .base import init_weights
from .sfstct import SpatialEncoder, TemporalEncoder


class SF_STCT_Core(Core, nn.Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, layers, device=None, persistence_path=None):
        super().__init__()
        # content_size = channels x height x width

        self.device = device

        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height

        self.packed_action_size = 1 + 3 + self.content_size  # ext_flag + action + x + y + content

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
        self.head_flag = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 5)   # 5 classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_size)   # action_size classes
        )
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
        self.head_content = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.content_size), # content_size classes
            nn.Sigmoid()
        )
        self.head_value = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1)   # Regression output
        )

        self.value_logstd = nn.Parameter(torch.zeros(1, 1))

        # self.position_step = Multilayer_Relu(position_size + action_size + width + height, position_size, hidden_size, 2)
        self.position_step = nn.Sequential(
            nn.Linear(position_size + action_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size),
            nn.Sigmoid()
        )

        self.reset_parameters()

        self.persistence_path = persistence_path
        if self.persistence_path is not None:
            self.load()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.spatial_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()

        self.head_flag.apply(init_weights)
        self.head_action.apply(init_weights)
        self.head_x.apply(init_weights)
        self.head_y.apply(init_weights)
        self.head_content.apply(init_weights)
        self.head_value.apply(init_weights)

        self.position_step.apply(init_weights)

        # make sure value_logstd is initialized to 0
        nn.init.constant_(self.value_logstd, 0.0)


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


    def __compute(self, context, action):
        # context has shape (batch, context_size, 1 + position_size + content_size)
        # action has shape (batch, context_size, self.packed_action_size)
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
    

    def get_latest_value(self, context, action):
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
        elif action is None:
            action = torch.zeros((context.size(0), context.size(1), self.packed_action_size), dtype=torch.int64).to(self.device)

        with torch.no_grad():
            temporal_feat = self.__compute(context, action)
            logits_value = self.head_value(temporal_feat)    # (B, T, 1)
            
        return logits_value[:, -1, ...].cpu().numpy()
    

    def get_action_and_value(self, context, action, use_action=False, use_grad=True):
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        if action is None:
            action = torch.zeros((context.size(0), context.size(1), self.packed_action_size), dtype=torch.float32).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)

        batch_size = context.size(0)
        temporal_feat = self.__compute(context, action)

        logits_flag = self.head_flag(temporal_feat)    # (B, T, 5)
        logits_action = self.head_action(temporal_feat) # (B, T, 6)
        logits_x = self.head_x(temporal_feat)      # (B, T, 64)
        logits_y = self.head_y(temporal_feat)      # (B, T, 64)
        pprobs_content = self.head_content(temporal_feat) # (B, T, content_size)
        value = self.head_value(temporal_feat)    # (B, T, 1)

        props_flag = Categorical(logits=logits_flag)
        props_action = Categorical(logits=logits_action)
        props_x = Categorical(logits=logits_x)
        props_y = Categorical(logits=logits_y)
        props_content = Bernoulli(probs=pprobs_content)

        if use_action:
            action_flag = action[:, :, 0]
            action_action = action[:, :, 1]
            action_x = action[:, :, 2]
            action_y = action[:, :, 3]
            action_content = action[:, :, 4:]
        else:
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


        # compute position
        last_position = context[:, :, 1:1 + self.position_size]
        # make one hot encoding for action, x, y
        action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
        # x_onehot = torch.nn.functional.one_hot(action_x.long(), num_classes=self.width).float()
        # y_onehot = torch.nn.functional.one_hot(action_y.long(), num_classes=self.height).float()
        # position = self.position_step(torch.concat([last_position, action_onehot, x_onehot, y_onehot], dim=-1))
        logits_position = self.position_step(torch.concat([last_position, action_onehot], dim=-1))
        props_position = Bernoulli(probs=logits_position)
        position = props_position.sample()


        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = props_y.log_prob(action_y)
        log_prob_content = props_content.log_prob(action_content).sum(-1)
        log_prob_position = props_position.log_prob(position).sum(-1)
        batch_log_prob = log_prob_flag + log_prob_action + log_prob_x + log_prob_y + log_prob_content + log_prob_position
        
        # clamp batch_log_prob to avoid too large negatives
        batch_log_prob = torch.clamp(batch_log_prob, min=-100, max=0)

        entropy_flag = props_flag.entropy()
        entropy_action = props_action.entropy()
        entropy_x = props_x.entropy()
        entropy_y = props_y.entropy()
        entropy_content = props_content.entropy().sum(-1)
        entropy_position = props_position.entropy().sum(-1)
        batch_entropy = entropy_flag + entropy_action + entropy_x + entropy_y + entropy_content + entropy_position

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


    def get_log_probability(self, context, action, target_action=None, f_mask=None):
        """
        context has shape (batch, context_size, ...)
        action has shape (batch, context_size, self.packed_action_size)
        target_action has shape (batch, context_size, self.packed_action_size)
        f_mask has shape (batch, context_size, 5)
        """

        batch_size = context.size(0)
        context_size = context.size(1)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        if action is None:
            action = torch.zeros((batch_size, context_size, 1 + 3 + self.content_size), dtype=torch.float32).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)

        if f_mask is None:
            f_mask = torch.ones((batch_size, context_size, 5), dtype=torch.float32).to(self.device)
        elif isinstance(f_mask, np.ndarray):
            f_mask = torch.tensor(f_mask, dtype=torch.float32).to(self.device)

        temporal_feat = self.__compute(context, action)

        logits_flag = self.head_flag(temporal_feat)    # (B, T, 5)
        logits_action = self.head_action(temporal_feat) # (B, T, 6)
        logits_x = self.head_x(temporal_feat)      # (B, T, 64)
        logits_y = self.head_y(temporal_feat)      # (B, T, 64)
        pprobs_content = self.head_content(temporal_feat) # (B, T, content_size)

        props_flag = Categorical(logits=logits_flag)
        props_action = Categorical(logits=logits_action)
        props_x = Categorical(logits=logits_x)
        props_y = Categorical(logits=logits_y)
        props_content = Bernoulli(probs=pprobs_content)

        if target_action is None:
            target_action = action

        action_flag = target_action[:, :, 0]
        action_action = target_action[:, :, 1]
        action_x = target_action[:, :, 2]
        action_y = target_action[:, :, 3]
        action_content = target_action[:, :, 4:]

        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = props_y.log_prob(action_y)
        log_prob_content = props_content.log_prob(action_content).sum(-1)

        log_prob = torch.stack([log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_content], dim=-1)
        masked_log_prob = log_prob * f_mask
        sum_log_prob = torch.sum(masked_log_prob, dim=-1)

        return sum_log_prob


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return ext_flag, action_data... , content

        ext_part = packed_action[:, 0].astype(int)
        action = packed_action[:, 1].astype(int)
        x = packed_action[:, 2].astype(int)
        y = packed_action[:, 3].astype(int)
        content = packed_action[:, 4:].astype(float)

        return ext_part, action, x, y, content
    

    def pack_action(self, b_ext=None, b_action=None, b_x=None, b_y=None, b_content=None):
        # b_xxx has shape (batch, ...)
        # return packed_action_seq of shape (batch, self.packed_action_size) of type int
        # replace none with zeros

        batch_size = None
        if b_ext is not None:
            batch_size = b_ext.shape[0]
        elif b_action is not None:
            batch_size = b_action.shape[0]
        elif b_x is not None:
            batch_size = b_x.shape[0]
        elif b_y is not None:
            batch_size = b_y.shape[0]
        elif b_content is not None:
            batch_size = b_content.shape[0]
        else:
            raise ValueError("At least one of b_ext, b_action, b_x, b_y, b_content must be provided")
        
        if b_ext is None:
            b_ext = np.zeros((batch_size,), dtype=int)
        if b_action is None:
            b_action = np.zeros((batch_size,), dtype=int)
        if b_x is None:
            b_x = np.zeros((batch_size,), dtype=int)
        if b_y is None:
            b_y = np.zeros((batch_size,), dtype=int)
        if b_content is None:
            b_content = np.zeros((batch_size, self.content_size), dtype=float)

        packed_action = np.concatenate([
            b_ext.reshape((batch_size, 1)),
            b_action.reshape((batch_size, 1)),
            b_x.reshape((batch_size, 1)),
            b_y.reshape((batch_size, 1)),
            b_content
        ], axis=-1).astype(int)

        return packed_action
    
