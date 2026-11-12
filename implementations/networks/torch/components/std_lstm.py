import torch
import torch.nn as nn

from implementations.networks.torch.components.base import init_weights, Categorical_With_Mask, gumbel_rao_straight_through
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


def _step_mask(mask, t):
    # masks may cover fewer steps than the input (learning passes none for the bootstrap step)
    if mask is None or t >= mask.size(1):
        return None
    return mask[:, t]


class Gumbel_Recurrent_Cell(nn.Module):
    """
    A Recurrent_Stack step that owns categorical output heads trained by backpropagation instead of RL:
        (x_t, state_{t-1}, head outputs_{t-1}) -> (head logits_t, state_t, output_t)
    Each head's previous output is fed back as a one-hot. Its forward value is always the category that was actually
    taken (supplied by the caller, e.g. replayed from a rollout), so acting and learning see identical inputs.
    Where the recurrence slides on (t > 0, no reset) its gradient is the straight-through Gumbel-Rao relaxation of the
    step t - 1 logits, so losses on later steps backpropagate through the discrete choice into the heads;
    where it restarts the feedback is a constant one-hot.

    forward(x (batch, time, input_dims), prev_outputs (batch, time, num_heads) integer, prev_state (batch, time, state_dims),
            reset (batch, time) bool or None, masks list (num_heads) of (batch, time or fewer, size) bool or None, sample)
        -> (logits list (num_heads) of (batch, time, size), state (batch, time, state_dims), output (batch, time, output_dims),
            samples (batch, time, num_heads) long if sample else None)
    """

    def __init__(self, output_dims, input_dims, hidden_dims, layers, head_sizes, cell="lstm", gumbel_tau=1.0, gumbel_samples=8):
        super().__init__()
        self.head_sizes = list(head_sizes)
        self.gumbel_tau = gumbel_tau
        self.gumbel_samples = gumbel_samples

        recurrent_classes = {"lstm": LSTM, "gru": GRU}
        if cell not in recurrent_classes:
            raise ValueError(f"cell must be one of {list(recurrent_classes)}, got {cell!r}")
        self.recurrence = recurrent_classes[cell](
            output_dims=output_dims,
            input_dims=input_dims + sum(self.head_sizes),
            hidden_dims=hidden_dims,
            layers=layers
        )
        self.state_dims = self.recurrence.state_dims

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dims, output_dims),
                nn.GELU(),
                nn.Linear(output_dims, size)
            )
            for size in self.head_sizes
        ])


    def reset_parameters(self):
        self.recurrence.reset_parameters()
        for head in self.heads:
            head.apply(init_weights)


    def forward(self, x, prev_outputs, prev_state, reset=None, masks=None, sample=False):
        masks = masks if masks is not None else [None] * len(self.heads)
        prev_outputs = prev_outputs.long()
        prev_one_hot = torch.concat([
            nn.functional.one_hot(prev_outputs[..., h], num_classes=size).float()
            for h, size in enumerate(self.head_sizes)
        ], dim=-1)
        relax = torch.is_grad_enabled()  # the relaxation only changes gradients, never the forward value

        step_logits, states, outputs, samples = [], [], [], []
        state = prev_state[:, 0]
        logits = None
        for t in range(x.size(1)):
            feedback = prev_one_hot[:, t]
            if t > 0:
                restart = reset[:, t, None] if reset is not None else torch.zeros_like(feedback[:, :1], dtype=torch.bool)
                state = torch.where(restart, prev_state[:, t], state)
                if relax:
                    relaxed = torch.concat([
                        gumbel_rao_straight_through(
                            l, prev_outputs[:, t, h], tau=self.gumbel_tau, num_samples=self.gumbel_samples, mask=_step_mask(m, t - 1)
                        )
                        for h, (l, m) in enumerate(zip(logits, masks))
                    ], dim=-1)
                    feedback = torch.where(restart, feedback, relaxed)

            output, state = self.recurrence.step(torch.concat([x[:, t], feedback], dim=-1), state)
            logits = [head(output) for head in self.heads]

            if sample:
                samples.append(torch.stack([
                    Categorical_With_Mask(logits=l, mask=_step_mask(m, t)).sample() for l, m in zip(logits, masks)
                ], dim=-1))

            step_logits.append(logits)
            states.append(state)
            outputs.append(output)

        all_logits = [torch.stack(head_logits, dim=1) for head_logits in zip(*step_logits)]
        return all_logits, torch.stack(states, dim=1), torch.stack(outputs, dim=1), (torch.stack(samples, dim=1) if sample else None)
