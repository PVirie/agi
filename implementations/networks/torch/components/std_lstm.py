import torch
import torch.nn as nn

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.std_resnet import ResNet


class LSTM(nn.Module):
    """
    LSTM whose initial cell state is supplied by the caller instead of being carried internally,
    so the recurrence can be stored outside the network (e.g. in the context or a graph memory).
    Only the cell state c is carried; gates see c_{t-1} directly (peephole style) in place of h_{t-1}.

    forward(x (batch, time, input_dims), prev_state (batch, time, hidden_dims), reset (batch, time) bool or None)
        -> (output (batch, time, output_dims), state (batch, time, hidden_dims))
    Slides through time: step t takes the state computed at step t - 1 (so gradients flow back through time),
    except at t = 0 and where reset is True, which take prev_state[:, t] (e.g. a stored state or zeros at an episode start).
    """

    def __init__(self, output_dims, input_dims, hidden_dims, layers):
        super().__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.layers = layers

        self.trunk = ResNet(
            output_dims=output_dims,
            input_dims=input_dims + hidden_dims,
            hidden_dims=output_dims,
            layers=layers
        )
        self.gate_layer = nn.Linear(output_dims, 4 * hidden_dims)  # input, forget, cell candidate, output
        self.output_layer = nn.Linear(output_dims + hidden_dims, output_dims)


    def reset_parameters(self):
        self.trunk.reset_parameters()
        self.gate_layer.apply(init_weights)
        self.output_layer.apply(init_weights)
        with torch.no_grad():
            # forget-gate bias 1: retain the carried state by default (Jozefowicz et al., 2015)
            self.gate_layer.bias[self.hidden_dims:2 * self.hidden_dims].fill_(1.0)


    def step(self, x, prev_state):
        features = self.trunk(torch.concat([x, prev_state], dim=-1))
        i, f, g, o = self.gate_layer(nn.functional.gelu(features)).chunk(4, dim=-1)
        state = torch.sigmoid(f) * prev_state + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.sigmoid(o) * torch.tanh(state)
        output = self.output_layer(torch.concat([features, hidden], dim=-1))
        return output, state


    def forward(self, x, prev_state, reset=None):
        outputs = []
        states = []
        state = prev_state[:, 0]
        for t in range(x.size(1)):
            if t > 0 and reset is not None:
                state = torch.where(reset[:, t, None], prev_state[:, t], state)
            output, state = self.step(x[:, t], state)
            outputs.append(output)
            states.append(state)
        return torch.stack(outputs, dim=1), torch.stack(states, dim=1)
