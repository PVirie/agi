import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Multilayer_Relu, apply_transformer, causal_mask, reset_transformer_decoder


class Model(nn.Module):
    def __init__(self, 
                 input_size, hidden_size, 
                 context_head, context_layers,
                 action_head, action_layers,
                 state_head, state_layers,
                 device):
        super(Model, self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        self.embedding    = nn.Embedding(input_size, hidden_size, device=device)
        # state encoder

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=context_head, device=device)
        self.context_decoder = nn.TransformerDecoder(decoder_layer, num_layers=context_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=action_head, device=device)
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=action_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=state_head, device=device)
        self.state_decoder = nn.TransformerDecoder(decoder_layer, num_layers=state_layers)

        self.critic = Multilayer_Relu(hidden_size, 1, hidden_size, 2, device=device)
        self.actor = Multilayer_Relu(hidden_size, hidden_size, hidden_size, 2, device=device)

        self.pe = nn.Embedding(1024, hidden_size, device=device)

        self.reset_parameters()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.embedding.reset_parameters()
        reset_transformer_decoder(self.context_decoder)
        reset_transformer_decoder(self.action_decoder)
        reset_transformer_decoder(self.state_decoder)
        self.critic.reset_parameters()
        self.actor.reset_parameters()
        self.pe.reset_parameters()


    def apply_positional_encoding(self, x, context_size):
        pe = self.pe(torch.arange(0, context_size, device=self.device))
        return x + pe
    

    def forward(self, objectives, observations, actions, pivot_positions):
        # objectives has shape batch x objective_context_size
        # observations has shape batch x n_contexts x context_size
        # actions has shape batch x n_pivots x n_actions x action_size
        # pivot_positions has shape batch x n_pivots
        batch = objectives.size(0)
        objective_context_size = objectives.size(1)
        n_contexts = observations.size(1)
        context_size = observations.size(2)
        n_pivots = pivot_positions.size(1)
        n_actions = actions.size(2)
        action_size = actions.size(3)

        objective_embedding = self.embedding(objectives) # batch x objective_context_size x hidden
        objective_embedding = self.apply_positional_encoding(objective_embedding, objective_context_size)
        
        obs_embedding = self.embedding(observations) # batch x n_contexts x context_size x hidden
        obs_embedding = self.apply_positional_encoding(obs_embedding, context_size)
        obs_embedding = apply_transformer(self.context_decoder, torch.reshape(obs_embedding, (-1, context_size, self.hidden_size)))
        obs_embedding = torch.reshape(obs_embedding[:, 0, :], (-1, n_contexts, self.hidden_size)) # batch x n_contexts x hidden

        obs_embedding = self.apply_positional_encoding(obs_embedding, n_contexts)
        state_internal = apply_transformer(self.state_decoder, obs_embedding, memory=objective_embedding, tgt_mask=causal_mask(n_contexts, self.device), tgt_is_causal=True) # batch x n_contexts x hidden
        
        pivot_state_internal = torch.gather(state_internal, 1, pivot_positions.unsqueeze(-1).expand(-1, -1, self.hidden_size)) # batch x n_pivots x hidden
        
        action_embedding = self.embedding(actions) # batch x n_pivots x n_actions x action_size x hidden
        action_embedding = self.apply_positional_encoding(action_embedding, action_size)
        action_embedding = apply_transformer(self.action_decoder, torch.reshape(action_embedding, (-1, action_size, self.hidden_size)))
        action_embedding = torch.reshape(action_embedding[:, 0, :], (batch, n_pivots, n_actions, self.hidden_size)) # batch x n_pivots x n_actions x hidden

        # now cross product between the state_internal and the action_embedding
        pre_actions = torch.reshape(self.actor(pivot_state_internal), (batch, n_pivots, self.hidden_size, 1)) # batch x n_pivots x hidden x 1
        scores = torch.matmul(action_embedding, pre_actions) # batch x n_pivots x n_actions x 1

        state_values = self.critic(pivot_state_internal) # batch x n_pivots x 1

        # return scores has shape batch x n_pivots x n_actions; is the scores of individual actions along the context length
        # state_values has shape batch x n_pivots; is the values of the best action from the current state
        return scores.squeeze(-1), state_values.squeeze(-1)
    