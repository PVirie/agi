import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import apply_transformer, causal_mask, reset_transformer_decoder


def access(Vs, Ss, x, scores, num_slots, dims, value_access):

    Vs = torch.reshape(Vs, (-1, num_slots, dims))
    Ss = torch.reshape(Ss, (-1, num_slots))

    # Access
    if value_access:
        x = torch.reshape(x, (-1, dims))
        Vl = torch.unsqueeze(x, 2)
        denom = torch.norm(Vs, dim=2, keepdims=True) * torch.norm(Vl, dim=1, keepdims=True)
        # prevent divide by zero
        denom = torch.maximum(1e-6, denom)
        dot_scores = torch.matmul(Vs, Vl) / denom
        # force dot_scores shape [batch, memory_size]
        dot_scores = torch.reshape(dot_scores, (-1, num_slots))
        max_indices = torch.argmax(dot_scores, dim=1, keepdims=True)    
    else:
        # score has range [0, 1], quantize to int slots, must be handled from out side
        scores = torch.reshape(scores, (-1, 1))
        max_indices = torch.round(scores * (num_slots - 1)).to(torch.int64)
        max_indices = torch.clamp(max_indices, min=0, max=num_slots - 1)

    v = torch.gather(Vs, torch.unsqueeze(max_indices, dim=-1).expand(-1, -1, dims), axis=1)
    s = torch.gather(Ss, 1, max_indices)
    
    return v, s


class Model(nn.Module):
    def __init__(self, 
                 input_size, hidden_size,
                 context_head, context_layers,
                 objective_head, objective_layers,
                 action_head, action_layers,
                 value_head, value_layers,
                 device):
        super(Model, self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        self.embedding    = nn.Embedding(input_size, hidden_size, device=device)
        # state encoder

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=context_head, device=device)
        self.context_decoder = nn.TransformerDecoder(decoder_layer, num_layers=context_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=objective_head, device=device)
        self.objective_decoder = nn.TransformerDecoder(decoder_layer, num_layers=objective_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=action_head, device=device)
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=action_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=value_head, device=device)
        self.value_decoder = nn.TransformerDecoder(decoder_layer, num_layers=value_layers)

        self.pe = nn.Embedding(1024, hidden_size, device=device)

        self.reset_parameters()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.embedding.reset_parameters()
        reset_transformer_decoder(self.context_decoder)
        reset_transformer_decoder(self.objective_decoder)
        reset_transformer_decoder(self.action_decoder)
        reset_transformer_decoder(self.value_decoder)
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
        objective_embedding = apply_transformer(self.objective_decoder, objective_embedding)
        
        obs_embedding = self.embedding(observations) # batch x n_contexts x context_size x hidden
        obs_embedding = self.apply_positional_encoding(obs_embedding, context_size)
        obs_embedding = apply_transformer(self.context_decoder, torch.reshape(obs_embedding, (-1, context_size, self.hidden_size)))
        obs_embedding = torch.reshape(obs_embedding[:, 0, :], (-1, n_contexts, self.hidden_size)) # batch x n_contexts x hidden

        obs_embedding = self.apply_positional_encoding(obs_embedding, n_contexts)
        value_internal = apply_transformer(self.value_decoder, obs_embedding, memory=objective_embedding, tgt_mask=causal_mask(n_contexts, self.device), tgt_is_causal=True) # batch x n_contexts x hidden
        pivot_value_internal = torch.gather(value_internal, 1, pivot_positions.unsqueeze(-1).expand(-1, -1, self.hidden_size)) # batch x n_pivots x hidden
        
        action_embedding = self.embedding(actions) # batch x n_pivots x n_actions x action_size x hidden
        action_embedding = self.apply_positional_encoding(action_embedding, action_size)
        action_embedding = apply_transformer(self.action_decoder, torch.reshape(action_embedding, (-1, action_size, self.hidden_size)))
        action_embedding = torch.reshape(action_embedding[:, 0, :], (batch, n_pivots, n_actions, self.hidden_size)) # batch x n_pivots x n_actions x hidden

        # now cross product between the state_internal and the action_embedding
        pre_actions = torch.reshape(pivot_value_internal, (batch, n_pivots, self.hidden_size, 1)) # batch x n_pivots x hidden x 1
        qs = torch.matmul(action_embedding, pre_actions) # batch x n_pivots x n_actions x 1
        qs = qs[:, :, :, 0] # batch x n_pivots x n_actions

        # state_values = torch.max(qs, dim=2, keepdim=False)[0] # batch x n_pivots
        # state_values = torch.mean(qs, dim=2, keepdim=False) # batch x n_pivots; use means stabilize training
        # use mean of top k instead
        # top_k = min(4, n_actions)
        # state_values, _ = torch.topk(qs, top_k, dim=2, largest=True, sorted=False)
        # state_values = torch.mean(state_values, dim=2, keepdim=False)
        # use softmax
        with torch.no_grad():
            sfm = torch.nn.functional.softmax(qs, dim=2)
        state_values = torch.sum(sfm * qs, dim=2, keepdim=False)

        return qs, state_values
