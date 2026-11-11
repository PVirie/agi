import torch
import torch.nn as nn

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.std_resnet import ResNet


class LSTM_Cell(nn.Module):
    """
    One LSTM layer over a caller-supplied cell state.
    Only the cell state c is carried; gates see c_{t-1} directly (peephole style) in place of h_{t-1}.
    """

    def __init__(self, output_dims, input_dims, hidden_dims, layers):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.trunk = ResNet(
            output_dims=output_dims,
            input_dims=input_dims + hidden_dims,
            hidden_dims=output_dims,
            layers=layers
        )
        # keeps gate pre-activations unsaturated: stacked cells otherwise grow their outputs layer over layer
        self.norm = nn.LayerNorm(output_dims)
        self.gate_layer = nn.Linear(output_dims, 4 * hidden_dims)  # input, forget, cell candidate, output
        self.output_layer = nn.Linear(output_dims + hidden_dims, output_dims)


    def reset_parameters(self):
        self.trunk.reset_parameters()
        self.norm.reset_parameters()
        self.gate_layer.apply(init_weights)
        self.output_layer.apply(init_weights)
        with torch.no_grad():
            # forget-gate bias 1: retain the carried state by default (Jozefowicz et al., 2015)
            self.gate_layer.bias[self.hidden_dims:2 * self.hidden_dims].fill_(1.0)


    def forward(self, x, prev_state):
        features = self.norm(self.trunk(torch.concat([x, prev_state], dim=-1)))
        i, f, g, o = self.gate_layer(nn.functional.gelu(features)).chunk(4, dim=-1)
        state = torch.sigmoid(f) * prev_state + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.sigmoid(o) * torch.tanh(state)
        output = self.output_layer(torch.concat([features, hidden], dim=-1))
        return output, state


class GRU_Cell(nn.Module):
    """
    One GRU layer over a caller-supplied hidden state h, with a res-block input encoder:
    z, r = sigmoid(W [f, h]), n = tanh(W_n f + r * (U_n h)), h' = (1 - z) * n + z * h, where f = trunk(x).
    h stays in [-1, 1] when it starts there (e.g. zeros).
    """

    def __init__(self, output_dims, input_dims, hidden_dims, layers):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.trunk = ResNet(
            output_dims=output_dims,
            input_dims=input_dims,
            hidden_dims=output_dims,
            layers=layers
        )
        # keeps gate pre-activations unsaturated: stacked cells otherwise grow their outputs layer over layer
        self.norm = nn.LayerNorm(output_dims)
        self.gate_layer = nn.Linear(output_dims + hidden_dims, 2 * hidden_dims)  # update, reset
        self.candidate_input_layer = nn.Linear(output_dims, hidden_dims)
        self.candidate_state_layer = nn.Linear(hidden_dims, hidden_dims)
        self.output_layer = nn.Linear(output_dims + hidden_dims, output_dims)


    def reset_parameters(self):
        self.trunk.reset_parameters()
        self.norm.reset_parameters()
        self.gate_layer.apply(init_weights)
        self.candidate_input_layer.apply(init_weights)
        self.candidate_state_layer.apply(init_weights)
        self.output_layer.apply(init_weights)
        with torch.no_grad():
            # update-gate bias 1: retain the carried state by default, as the LSTM forget-gate bias
            self.gate_layer.bias[:self.hidden_dims].fill_(1.0)


    def forward(self, x, prev_state):
        features = self.norm(self.trunk(x))
        activated = nn.functional.gelu(features)
        z, r = torch.sigmoid(self.gate_layer(torch.concat([activated, prev_state], dim=-1))).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate_input_layer(activated) + r * self.candidate_state_layer(prev_state))
        state = (1 - z) * candidate + z * prev_state
        output = self.output_layer(torch.concat([features, state], dim=-1))
        return output, state


class Recurrent_Stack(nn.Module):
    """
    Multi-layer recurrent network whose initial states are supplied by the caller instead of being carried internally,
    so the recurrence can be stored outside the network (e.g. in the context or a graph memory).
    One cell_class per entry of layers (a single res block of that width each); layer l reads layer l - 1's output.
    The state is the concatenation of every layer's state, bottom layer first, so the top layer is state[..., -hidden_dims:].

    forward(x (batch, time, input_dims), prev_state (batch, time, len(layers) * hidden_dims), reset (batch, time) bool or None)
        -> (output (batch, time, output_dims), state (batch, time, len(layers) * hidden_dims))
    Slides through time: step t takes the state computed at step t - 1 (so gradients flow back through time),
    except at t = 0 and where reset is True, which take prev_state[:, t] (e.g. a stored state or zeros at an episode start).
    """

    cell_class = None

    def __init__(self, output_dims, input_dims, hidden_dims, layers):
        super().__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.layers = layers
        self.num_layers = len(layers)
        self.state_dims = self.num_layers * hidden_dims

        self.cells = nn.ModuleList([
            self.cell_class(
                output_dims=output_dims,
                input_dims=input_dims if l == 0 else output_dims,
                hidden_dims=hidden_dims,
                layers=[layer_dims]
            )
            for l, layer_dims in enumerate(layers)
        ])


    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()


    def step(self, x, prev_state):
        states = []
        for cell, cell_state in zip(self.cells, prev_state.chunk(self.num_layers, dim=-1)):
            x, cell_state = cell(x, cell_state)
            states.append(cell_state)
        return x, torch.concat(states, dim=-1)


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


class LSTM(Recurrent_Stack):
    """Recurrent_Stack of LSTM_Cell; the state is every layer's cell state c."""
    cell_class = LSTM_Cell


class GRU(Recurrent_Stack):
    """Recurrent_Stack of GRU_Cell; the state is every layer's hidden state h."""
    cell_class = GRU_Cell
