import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging
import copy

from interfaces.network import Q_Network
from ..components.base import init_weights, Categorical_With_Mask
from ..components.conv_resnet import ResNet, Bottleneck, Block
from utilities.safe_torch_module import Safe_nn_Module



class ParametricResNetQNetwork(nn.Module):
    def __init__(self, ResBlock, layer_list, action_feature_dim, num_channels=3, hidden_dim=512):
        super(ParametricResNetQNetwork, self).__init__()
        
        # --- 1. State Encoder (Image) ---
        # We reuse your ResNet to process the image part
        self.state_encoder = ResNet(ResBlock, layer_list, num_classes=1, num_channels=num_channels)
        self.state_feature_dim = 512 * ResBlock.expansion
        self.state_encoder.fc = nn.Identity() # Remove classification head
        
        # Project ResNet output to a shared hidden dimension
        self.state_projector = nn.Sequential(
            nn.Linear(self.state_feature_dim, hidden_dim),
            nn.ReLU()
        )

        # --- 2. Action Encoder (Feature Vector) ---
        # Processes the action features: (Batch, Seq, Num_Choices, Features)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- 3. Interaction / Scoring Head ---
        # Takes concatenated State + Action embeddings and outputs a Q-value
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Scalar Q-value output
        )
        
        self.reset_parameters()


    def reset_parameters(self):
        self.state_encoder.reset_parameters()
        self.state_projector.apply(init_weights)
        self.action_encoder.apply(init_weights)
        self.scorer.apply(init_weights)


    def forward(self, state, actions):
        """
        state:   (Batch, Seq, Channels, Height, Width)
        actions: (Batch, Seq, Num_choices, Action_Features)
        
        Returns:
        q_values: (Batch, Seq, Num_choices)
        """
        B, S, C, H, W = state.shape
        _, _, N, F = actions.shape
        
        # -----------------------------------------------------------
        # 1. Process States
        # -----------------------------------------------------------
        # Flatten Batch and Seq: (B*S, C, H, W)
        flat_state = state.view(B * S, C, H, W)
        
        # Encode: (B*S, state_feature_dim)
        state_feats = self.state_encoder(flat_state)
        
        # Project: (B*S, hidden_dim)
        state_emb = self.state_projector(state_feats)
        
        # -----------------------------------------------------------
        # 2. Process Actions
        # -----------------------------------------------------------
        # Flatten Batch, Seq, and Choices to feed into Linear layer
        # Input: (B * S * N, F)
        flat_actions = actions.view(B * S * N, F)
        
        # Encode: (B * S * N, hidden_dim)
        action_emb = self.action_encoder(flat_actions)
        
        # Reshape to separate the 'N' choices: (B*S, N, hidden_dim)
        action_emb = action_emb.view(B * S, N, -1)
        
        # -----------------------------------------------------------
        # 3. Combine and Score
        # -----------------------------------------------------------
        # We need to pair the state with *each* of the N actions.
        # Unsqueeze state to broadcast: (B*S, 1, hidden_dim)
        state_emb_expanded = state_emb.unsqueeze(1)
        
        # Expand state to match N choices: (B*S, N, hidden_dim)
        state_emb_expanded = state_emb_expanded.expand(-1, N, -1)
        
        # Concatenate: (B*S, N, hidden_dim * 2)
        cat_features = torch.cat([state_emb_expanded, action_emb], dim=2)
        
        # Calculate Q values: (B*S, N, 1)
        q_flat = self.scorer(cat_features)
        
        # -----------------------------------------------------------
        # 4. Final Reshape
        # -----------------------------------------------------------
        # Reshape back to (B, S, N)
        q_values = q_flat.view(B, S, N)
        
        return q_values


class Q_Core(Q_Network, nn.Module, Safe_nn_Module):
    def __init__(self, ResBlock, layer_list, action_feature_dim, num_channels=3, hidden_dim=512):
        super(Q_Core, self).__init__()

        # --- Initialize Online Networks ---
        # Q1: The first critic
        self.q1 = ParametricResNetQNetwork(
            ResBlock, layer_list, action_feature_dim, num_channels, hidden_dim
        )
        # Q2: The second critic (for Min-Q trick)
        self.q2 = ParametricResNetQNetwork(
            ResBlock, layer_list, action_feature_dim, num_channels, hidden_dim
        )

        # --- Initialize Target Networks ---
        # Create exact copies of the online networks
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Freeze Target Networks
        # We do not update them via gradients, only via soft_update (polyak averaging)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False


    def get_q1_values(self, context, action):
        """
        Forward pass for Online Q1
        Returns: (Batch, Seq, Num_choices)
        """
        return self.q1(context, action)


    def get_q2_values(self, context, action):
        """
        Forward pass for Online Q2
        Returns: (Batch, Seq, Num_choices)
        """
        return self.q2(context, action)


    def get_q1_target_values(self, context, action):
        """
        Forward pass for Target Q1 (No Grad)
        Returns: (Batch, Seq, Num_choices)
        """
        with torch.no_grad():
            return self.q1_target(context, action)


    def get_q2_target_values(self, context, action):
        """
        Forward pass for Target Q2 (No Grad)
        Returns: (Batch, Seq, Num_choices)
        """
        with torch.no_grad():
            return self.q2_target(context, action)


    def soft_update(self, tau=0.005):
        """
        Polyak averaging for target networks:
        theta_target = tau * theta_online + (1 - tau) * theta_target
        """
        # Update Q1 Target
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update Q2 Target
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



if __name__ == "__main__":
    # Test
    # Batch=2, Seq=5, Choices=4, Act_Feats=10, Channels=3, 128x128 Img
    model = ParametricResNetQNetwork(Block, [2, 2, 2, 2], action_feature_dim=10)
    
    s = torch.randn(2, 5, 3, 128, 128)
    a = torch.randn(2, 5, 4, 10)
    
    q = model(s, a)
    
    print(f"State shape: {s.shape}")
    print(f"Action shape: {a.shape}")
    print(f"Output shape: {q.shape}")
    
    assert q.shape == (2, 5, 4)
    print("Test Passed!")


    # 1. Initialize the SAC Double Q wrapper
    sac_critic = Q_Core(
        ResBlock=Bottleneck, 
        layer_list=[2, 2, 2, 2], # ResNet18 config
        action_feature_dim=10
    )
    
    # 2. Dummy Data
    state = torch.randn(2, 5, 3, 128, 128) # (Batch, Seq, C, H, W)
    action_candidates = torch.randn(2, 5, 4, 10) # (Batch, Seq, N, Features)

    # 3. Get Q-Values
    q1_vals = sac_critic.get_q1_values(state, action_candidates)
    q2_vals = sac_critic.get_q2_values(state, action_candidates)
    
    # 4. Get Target Values
    q1_t_vals = sac_critic.get_q1_target_values(state, action_candidates)

    print(f"Q1 Shape: {q1_vals.shape}") # Should be (2, 5, 4)
    
    # 5. Perform Soft Update
    sac_critic.soft_update(tau=0.01)
    print("Soft update completed.")