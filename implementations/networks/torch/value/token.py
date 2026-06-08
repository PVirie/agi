import torch
import torch.nn as nn
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from interfaces.network import Value_Network
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import ImpalaCNN
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from utilities.safe_torch_module import Safe_nn_Module


class Value_Core(Value_Network, nn.Module, Safe_nn_Module):

    def __init__(self, 
                 int_action_size, ext_action_size,
                 position_size,
                 output_dims,
                 token_part_size,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers,
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="token_image_value_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.token_part_size = token_part_size
        self.embedding_dim = embedding_dim

        self.int_action_size = int_action_size  # num classes for flag
        self.action_size = ext_action_size
        self.position_size = position_size
        self.content_size = token_part_size
        self.packed_action_size = 1 + output_dims + position_size + self.content_size  # int_flag + action + ... + position + content
        self.packed_context_size = 1 + 1 + output_dims + position_size + self.content_size  # reward + packed_action_size

        self.hidden_size = hidden_size
        self.output_dims = output_dims

        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for tokens
        self.token_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size, 
            nhead=8, 
            num_layers=2, 
            max_len=token_part_size
        )

        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=hidden_size, 
            hidden_dims=hidden_size, 
            layers=[hidden_size for _ in layers]
        )

        self.read_out_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.token_feature_extraction.reset_parameters()
        self.backbone.reset_parameters()

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.read_out_layers.apply(init_value_weights)


    def compute(self, context):
        # context has shape (batch, context_size, packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        contents = context[:, :, (1 + 1 + self.output_dims + self.position_size):]  # (batch_size, context_size, token_part_size)

        token_mask = get_padding_mask(contents, self.pad_token_id)  # (batch_size, context_size, token_part_size)
        embedded = self.embedding(contents.long())  # (batch_size, context_size, packed_context_size, embedding_dim)
        token_features = self.token_feature_extraction(embedded, token_mask)  # (batch_size, context_size, hidden_size)
        features = self.backbone(token_features)  # (batch_size, context_size, hidden_size)
        values = self.read_out_layers(features)  # (batch_size, context_size, 1)
        return values
    

    def get_value(self, context):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.int64).to(self.device)

        batch_size = context.size(0)
        context_size = context.size(1)
        value = self.compute(context)

        # collapse last dimension
        batch_value = torch.reshape(value, (batch_size, context_size))

        return batch_value

