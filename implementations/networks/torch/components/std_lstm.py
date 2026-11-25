from enum import Enum

import torch
import torch.nn as nn

from implementations.networks.torch.components.base import init_weights, Categorical_With_Mask
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


class Residual_Cell(nn.Module):
    """
    One residual recurrent layer over a caller-supplied state s, with no gates or squashing on the carried path:
    f = trunk([x, LN(s)]), s' = s + U gelu(f).
    ds'/ds = I + (a learned term), so gradients always have an untouched path back through time.
    s is unbounded; it is only read through a LayerNorm, so its scale never saturates anything downstream.
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
        # eps 1: near-linear while s is small (incl. the zero initial state, where eps 1e-5 would amplify gradients ~300x),
        # normalizing once s has grown
        self.state_norm = nn.LayerNorm(hidden_dims, eps=1.0)
        self.norm = nn.LayerNorm(output_dims)
        self.update_layer = nn.Linear(output_dims, hidden_dims)
        self.output_layer = nn.Linear(output_dims + hidden_dims, output_dims)


    def reset_parameters(self):
        self.trunk.reset_parameters()
        self.state_norm.reset_parameters()
        self.norm.reset_parameters()
        self.output_layer.apply(init_weights)
        # small updates: start near the identity recurrence (carry the state), as the LSTM forget-gate bias
        nn.init.orthogonal_(self.update_layer.weight, gain=0.1)
        nn.init.zeros_(self.update_layer.bias)


    def forward(self, x, prev_state):
        features = self.norm(self.trunk(torch.concat([x, self.state_norm(prev_state)], dim=-1)))
        state = prev_state + self.update_layer(nn.functional.gelu(features))
        output = self.output_layer(torch.concat([features, self.state_norm(state)], dim=-1))
        return output, state


class Recurrent_Stack(nn.Module):
    """
    Multi-layer recurrent network whose initial states are supplied by the caller instead of being carried internally,
    so the recurrence can be stored outside the network (e.g. in the context or a graph memory).
    One cell_class per entry of layers (a single res block of that width each); layer l reads layer l - 1's output.
    The state is the concatenation of every layer's state, bottom layer first, so the top layer is state[..., -hidden_dims:].

    step(x (batch, input_dims), prev_state (batch, len(layers) * hidden_dims))
        -> (output (batch, output_dims), state (batch, len(layers) * hidden_dims))
    One time step; the caller slides it through time (see Straight_Through_Recurrent_Cell).
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


class LSTM(Recurrent_Stack):
    """Recurrent_Stack of LSTM_Cell; the state is every layer's cell state c."""
    cell_class = LSTM_Cell


class GRU(Recurrent_Stack):
    """Recurrent_Stack of GRU_Cell; the state is every layer's hidden state h."""
    cell_class = GRU_Cell


class Residual_RNN(Recurrent_Stack):
    """Recurrent_Stack of Residual_Cell; the state is every layer's (unbounded) residual state s."""
    cell_class = Residual_Cell


RECURRENT_CLASSES = {"lstm": LSTM, "gru": GRU, "residual": Residual_RNN}


def make_recurrent_stack(cell, **kwargs):
    if cell not in RECURRENT_CLASSES:
        raise ValueError(f"cell must be one of {list(RECURRENT_CLASSES)}, got {cell!r}")
    return RECURRENT_CLASSES[cell](**kwargs)


class Sliding_Recurrent_Cell(nn.Module):
    """
    A Recurrent_Stack slid through time over caller-supplied states, with no output heads:
        (x_t, state_{t-1}) -> (state_t, output_t)
    At t = 0 and wherever reset is set the step starts from prev_state[:, t] (e.g. replayed from a rollout, or zeros
    at an episode start); elsewhere it carries its own state from step t - 1, so gradients flow back through time.

    forward(x (batch, time, input_dims), prev_state (batch, time, state_dims), reset (batch, time) bool or None)
        -> (state (batch, time, state_dims), output (batch, time, output_dims))
    """

    def __init__(self, output_dims, input_dims, hidden_dims, layers, cell="lstm"):
        super().__init__()
        self.recurrence = make_recurrent_stack(
            cell, output_dims=output_dims, input_dims=input_dims, hidden_dims=hidden_dims, layers=layers
        )
        self.state_dims = self.recurrence.state_dims


    def reset_parameters(self):
        self.recurrence.reset_parameters()


    def forward(self, x, prev_state, reset=None):
        states, outputs = [], []
        state = prev_state[:, 0]
        for t in range(x.size(1)):
            if t > 0 and reset is not None:
                state = torch.where(reset[:, t, None], prev_state[:, t], state)
            output, state = self.recurrence.step(x[:, t], state)
            states.append(state)
            outputs.append(output)
        return torch.stack(states, dim=1), torch.stack(outputs, dim=1)


def _step_mask(mask, t):
    # masks may cover fewer steps than the input (learning passes none for the bootstrap step)
    if mask is None or t >= mask.size(1):
        return None
    return mask[:, t]


class Sample_Type(Enum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class Straight_Through_Recurrent_Cell(nn.Module):
    """
    A Recurrent_Stack step that owns output heads whose sampled outputs are fed back into the next step:
        (x_t, state_{t-1}, head outputs_{t-1}) -> (head params_t, state_t, output_t)
    heads: list of (Sample_Type, size), one per head:
        CATEGORICAL: a category among size (params: size logits); output 1 slot (the index), fed back as a one-hot
        CONTINUOUS: a point in [0, 1]^size, independent Beta(1 + softplus(a), 1 + softplus(b)) per dim
                    (params: 2 * size); output size slots, fed back as is
    Each head's previous output is fed back with its forward value always the output that was actually taken (supplied by
    the caller, e.g. replayed from a rollout), so acting and learning see identical inputs.
    Where the recurrence slides on (t > 0, no reset) its gradient is straight-through the step t - 1 expected output
    (taken + E - E.detach(); E = the masked softmax or the Beta mean), so losses on later steps backpropagate,
    evaluated at the taken output, into the heads; where it restarts the feedback is a constant.

    forward(x (batch, time, input_dims), prev_outputs (batch, time, output_size), prev_state (batch, time, state_dims),
            reset (batch, time) bool or None, masks list (num_heads) of (batch, time or fewer, size) bool or None, sample,
            read callable or None)
        -> (params list (num_heads) of (batch, time, param_size), state (batch, time, state_dims), output (batch, time, output_dims),
            samples (batch, time, output_size) if sample else None)
    output_size is the heads' output slots concatenated in order; masks apply to categorical heads only.
    read(t, state_{t-1}) -> (batch, read_dims): an input computed per step from the state the step starts from (after any
    restart), appended to x[:, t]; input_dims counts x and the read together. It lets the state steer what the step reads.
    """

    def __init__(self, output_dims, input_dims, hidden_dims, layers, heads, cell="lstm"):
        super().__init__()
        for sample_type, _ in heads:
            if not isinstance(sample_type, Sample_Type):
                raise ValueError(f"head sample type must be a Sample_Type, got {sample_type!r}")
        self.head_specs = [(sample_type, int(size)) for sample_type, size in heads]
        self.num_heads = len(self.head_specs)
        self.output_widths = [1 if t == Sample_Type.CATEGORICAL else size for t, size in self.head_specs]
        self.output_size = sum(self.output_widths)
        self.output_slices = []
        offset = 0
        for width in self.output_widths:
            self.output_slices.append(slice(offset, offset + width))
            offset += width

        self.recurrence = make_recurrent_stack(
            cell,
            output_dims=output_dims,
            input_dims=input_dims + sum(size for _, size in self.head_specs),
            hidden_dims=hidden_dims,
            layers=layers
        )
        self.state_dims = self.recurrence.state_dims

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dims, output_dims),
                nn.GELU(),
                nn.Linear(output_dims, size if sample_type == Sample_Type.CATEGORICAL else 2 * size)
            )
            for sample_type, size in self.head_specs
        ])


    def reset_parameters(self):
        self.recurrence.reset_parameters()
        for head in self.heads:
            head.apply(init_weights)


    def distribution(self, h, params, mask=None):
        sample_type, _ = self.head_specs[h]
        if sample_type == Sample_Type.CATEGORICAL:
            return Categorical_With_Mask(logits=params, mask=mask)
        if mask is not None:
            raise ValueError(f"head {h} is continuous and cannot be masked")
        # concentrations >= 1: unimodal, bounded density, and the entropy peaks at the uniform
        alpha, beta = (1.0 + nn.functional.softplus(params)).chunk(2, dim=-1)
        return torch.distributions.Beta(alpha, beta)


    def entropy(self, params, masks=None):
        # (...,) summed over heads; a continuous head's differential entropy is averaged over its dims,
        # so every head weighs as a single variable
        masks = masks if masks is not None else [None] * self.num_heads
        total = 0
        for h, (p, m) in enumerate(zip(params, masks)):
            entropy = self.distribution(h, p, m).entropy()
            total = total + (entropy if self.head_specs[h][0] == Sample_Type.CATEGORICAL else entropy.mean(dim=-1))
        return total


    def _feedback(self, prev_outputs):
        # forward value of every head's previous output as the recurrence reads it: one-hot index or the value itself
        parts = []
        for (sample_type, size), output_slice in zip(self.head_specs, self.output_slices):
            taken = prev_outputs[..., output_slice]
            if sample_type == Sample_Type.CATEGORICAL:
                taken = nn.functional.one_hot(taken[..., 0].long(), num_classes=size).float()
            parts.append(taken.float())
        return torch.concat(parts, dim=-1)


    def forward(self, x, prev_outputs, prev_state, reset=None, masks=None, sample=False, read=None):
        masks = masks if masks is not None else [None] * self.num_heads
        prev_feedback = self._feedback(prev_outputs)

        step_params, states, outputs, samples = [], [], [], []
        state = prev_state[:, 0]
        expected = None
        for t in range(x.size(1)):
            feedback = prev_feedback[:, t]
            if t > 0:
                restart = reset[:, t, None] if reset is not None else torch.zeros_like(feedback[:, :1], dtype=torch.bool)
                state = torch.where(restart, prev_state[:, t], state)
                # E - E.detach() is exactly 0 in the forward pass, so the input stays the taken output bit for bit
                feedback = torch.where(restart, feedback, feedback + (expected - expected.detach()))

            x_t = x[:, t] if read is None else torch.concat([x[:, t], read(t, state)], dim=-1)
            output, state = self.recurrence.step(torch.concat([x_t, feedback], dim=-1), state)
            params = [head(output) for head in self.heads]
            distributions = [self.distribution(h, p, _step_mask(m, t)) for h, (p, m) in enumerate(zip(params, masks))]
            expected = torch.concat([
                d.probs if sample_type == Sample_Type.CATEGORICAL else d.mean
                for d, (sample_type, _) in zip(distributions, self.head_specs)
            ], dim=-1)

            if sample:
                samples.append(torch.concat([
                    d.sample().unsqueeze(-1).float() if sample_type == Sample_Type.CATEGORICAL else d.sample()
                    for d, (sample_type, _) in zip(distributions, self.head_specs)
                ], dim=-1))

            step_params.append(params)
            states.append(state)
            outputs.append(output)

        all_params = [torch.stack(head_params, dim=1) for head_params in zip(*step_params)]
        return all_params, torch.stack(states, dim=1), torch.stack(outputs, dim=1), (torch.stack(samples, dim=1) if sample else None)
