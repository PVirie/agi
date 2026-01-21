import torch
import torch.nn as nn
import copy

from interfaces.network import Q_Network
from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.std_conv import ImpalaCNN
from utilities.safe_torch_module import Safe_nn_Module


class ParametricImpalaQNetwork(nn.Module):
    def __init__(self, input_channels, width, height, action_feature_dim, hidden_dim, layers):
        super(ParametricImpalaQNetwork, self).__init__()
        
        # A. State Encoder (IMPALA)
        self.state_encoder = ImpalaCNN(output_dims=hidden_dim, input_channels=input_channels, width=width, height=height, depths=layers)
        
        self.state_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # B. Action Encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # C. Scorer (Interaction Head)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Scalar Q-value
        )
        
        self.reset_parameters()

        self.eval()


    def reset_parameters(self):
        self.state_projector.apply(init_weights)
        self.action_encoder.apply(init_weights)
        self.scorer.apply(init_weights)


    def forward(self, state, actions):
        """
        state:   (Batch, Seq, Channels, Height, Width)
        actions: (Batch, Seq, Num_choices, Action_Features)
        """
        B, S, C, H, W = state.shape
        _, _, N, F = actions.shape
        
        # 1. Encode States
        flat_state = state.reshape(B * S, C, H, W)
        state_feats = self.state_encoder(flat_state)    # (B*S, hidden_dim)
        state_emb = self.state_projector(state_feats)   # (B*S, hidden_dim)
        
        # 2. Encode Actions
        flat_actions = actions.reshape(B * S * N, F)
        action_emb = self.action_encoder(flat_actions)  # (B*S*N, hidden_dim)
        action_emb = action_emb.reshape(B * S, N, -1)      # (B*S, N, hidden_dim)
        
        # 3. Combine
        # Expand state to match N choices per sequence step
        state_emb_expanded = state_emb.unsqueeze(1).expand(-1, N, -1)
        cat_features = torch.cat([state_emb_expanded, action_emb], dim=2)
        
        # 4. Score
        q_flat = self.scorer(cat_features)
        q_values = q_flat.reshape(B, S, N)

        
        return q_values
    

class Q_Core(Q_Network, nn.Module, Safe_nn_Module):
    def __init__(self, input_channels, width, height, action_feature_dim, hidden_dim, layers):
        super(Q_Core, self).__init__()

        # Online Networks
        self.q1 = ParametricImpalaQNetwork(input_channels, width, height, action_feature_dim, hidden_dim, layers)
        self.q2 = ParametricImpalaQNetwork(input_channels, width, height, action_feature_dim, hidden_dim, layers)

        # Target Networks
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Freeze Targets
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False


    def get_q1_values(self, context, action):
        return self.q1(context, action)


    def get_q2_values(self, context, action):
        return self.q2(context, action)


    def get_q1_target_values(self, context, action):
        with torch.no_grad():
            return self.q1_target(context, action)


    def get_q2_target_values(self, context, action):
        with torch.no_grad():
            return self.q2_target(context, action)


    def soft_update(self, tau=0.005):
        """
        Updates target parameters: theta_target = tau*theta_online + (1-tau)*theta_target
        """
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



if __name__ == "__main__":
    # Test
    model = ParametricImpalaQNetwork(input_channels=3, width=128, height=128, action_feature_dim=10, hidden_dim=512, layers=[16,32,32])
    
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
        input_channels=3,
        width=128,
        height=128,
        action_feature_dim=10,
        hidden_dim=512,
        layers=[16, 32, 32]
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