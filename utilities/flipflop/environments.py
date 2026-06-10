import datasets
import numpy as np

_MAPPING = {'w': 1, 'r': 2, 'i': 3, '0': 4, '1': 5}

dataset = datasets.load_dataset('synthseq/flipflop', cache_dir="/app/cache/datasets")

def tokenize_batch(batch):
    tokenized_batch = [[_MAPPING[char] for char in s] for s in batch['text']]
    return {
        'text': batch['text'],
        'tokens': np.array(tokenized_batch, dtype=np.int64)
    }

dataset.set_transform(tokenize_batch)

TOKEN_PAD = 0
TOKEN_W = 1
TOKEN_R = 2
TOKEN_I = 3
TOKEN_0 = 4
TOKEN_1 = 5
NUM_TOKENS = 6

_sequence_cache = {}

def _load_sequences(category):
    """Precompute and cache tokenized sequences for a dataset split."""
    if category not in _sequence_cache:
        # Column access returns all raw texts (transform preserves 'text' key)
        texts = dataset[category]['text']
        sequences = [
            np.array([_MAPPING[c] for c in text], dtype=np.int64)
            for text in texts
        ]
        np.random.shuffle(sequences)
        _sequence_cache[category] = sequences
    return _sequence_cache[category]


class FlipFlop_Environment:
    """
    Farama-style batched RL environment over the synthseq/flipflop dataset.

    Observation: np.array([token_id], dtype=int64), token_id in 0..4
                 (w=0, r=1, i=2, 0=3, 1=4)
    Action:      integer token_id in 0..4
    Reward:      +0.1 if prev obs was TOKEN_R and action == current obs
                 -0.1 if prev obs was TOKEN_R and action != current obs
                  0.0 otherwise
    Termination: True at the final token of each sequence.
    Auto-reset:  On termination, each env advances to
                 (current_index + batch_size) % n_samples,
                 looping back to the front when the end is reached.
    """

    def __init__(self, batch_size, category='train', loop=True):
        self.batch_size = batch_size
        self._sequences = _load_sequences(category)
        self.n_samples = len(self._sequences)

        self._available_actions = [list(range(NUM_TOKENS))] * batch_size

        # Per-env mutable state
        self._dataset_indices = list(range(batch_size))
        self._next_index = batch_size  # Round-robin pointer: next sequence to assign
        self._positions = [0] * batch_size
        self._prev_tokens = [None] * batch_size
        self._episode_returns = [0.0] * batch_size
        self._episode_eval_count = [0.0] * batch_size
        self._episode_correct_count = [0.0] * batch_size

        self.return_obs = [None] * batch_size
        self.return_rewards = [np.float32(0.0)] * batch_size
        self.return_terminations = [False] * batch_size
        self.return_truncations = [False] * batch_size
        self.return_infos = [{} for _ in range(batch_size)]

        self.loop = loop
        self.valid = [True] * batch_size  # Used to indicate if env is still valid (not done when loop=False)

    def reset(self, seed=None):
        self._dataset_indices = list(range(self.batch_size))
        self._next_index = self.batch_size
        self._positions = [0] * self.batch_size
        self._prev_tokens = [None] * self.batch_size
        self._episode_returns = [0.0] * self.batch_size
        self._episode_eval_count = [0.0] * self.batch_size
        self._episode_correct_count = [0.0] * self.batch_size

        for i in range(self.batch_size):
            tokens = self._sequences[self._dataset_indices[i]]
            self.return_obs[i] = np.array([int(tokens[0])], dtype=np.int64)
            self.return_infos[i] = {}

        self.valid = [True] * self.batch_size

        return list(self.return_obs), list(self.return_infos)

    def step(self, actions):
        for i in range(self.batch_size):
            if actions[i] is None:
                self.return_rewards[i] = np.float32(0.0)
                self.return_terminations[i] = False
                self.return_truncations[i] = False
                continue

            tokens = self._sequences[self._dataset_indices[i]]
            pos = self._positions[i]
            prev_token = self._prev_tokens[i]
            current_token = int(tokens[pos])

            # Reward: agent predicted after seeing TOKEN_R; current token is ground truth
            if prev_token == TOKEN_R:
                self._episode_eval_count[i] += 1.0
                if int(actions[i]) == current_token:
                    self._episode_correct_count[i] += 1
                    reward = np.float32(0.1)
                else:
                    reward = np.float32(-0.1)
            else:
                reward = np.float32(0.0)

            self._episode_returns[i] += float(reward)
            self.return_rewards[i] = reward
            self._prev_tokens[i] = current_token

            if pos == len(tokens) - 1:
                # Sequence finished
                self.return_terminations[i] = True
                self.return_truncations[i] = False
                self.return_infos[i] = {
                    "episode": {
                        "r": self._episode_returns[i],
                        "ec": self._episode_eval_count[i] if self.valid[i] else 0.0,
                        "cc": self._episode_correct_count[i] if self.valid[i] else 0.0,
                    }
                }

                if not self.loop and self._next_index >= self.n_samples:
                    # If not looping and we've exhausted all sequences, mark this env as invalid
                    self.valid[i] = False

                # Round-robin: assign next available sequence
                self._dataset_indices[i] = self._next_index % self.n_samples
                self._next_index += 1
                self._positions[i] = 0
                self._prev_tokens[i] = None
                self._episode_returns[i] = 0.0
                self._episode_eval_count[i] = 0.0
                self._episode_correct_count[i] = 0.0

                # Return first token of the new sequence
                new_tokens = self._sequences[self._dataset_indices[i]]
                self.return_obs[i] = np.array([int(new_tokens[0])], dtype=np.int64)
            else:
                self.return_terminations[i] = False
                self.return_truncations[i] = False
                self.return_infos[i] = {}

                self._positions[i] = pos + 1
                self.return_obs[i] = np.array([int(tokens[pos + 1])], dtype=np.int64)

        return (
            list(self.return_obs),
            list(self.return_rewards),
            list(self.return_terminations),
            list(self.return_truncations),
            list(self.return_infos),
        )

    def get_available_actions(self):
        return self._available_actions

    def close(self):
        pass

    def has_more(self):
        return any(self.valid)