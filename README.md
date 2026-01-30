# AGI

An attempt to solve AGI.

| Generation | Model Name            | Description                                                    |
| ---------- | --------------------- | -------------------------------------------------------------- |
| LIII       | Cognitive algebra     | A learnable computationally universal model.                   |
| LIX        | Explicit algebra core | An improved version of Model 53 with an explicit algebra core. |

## Setup

1. Setup environment in `secrets.env` file, place it in the root directory with the following content:

```env
    export SCHEME="https"
    export HOST="three.arcprize.org"
    export PORT="443"

    export ARC_API_KEY="your_arc_api_key_here"
```

## Run

There are two approaches to run the AGI agent:

1. Use vscode debugger with the provided launch configuration.
2. Run using the shell script:

```bash
./run_manual.sh {torch-rocm} tasks/{arcagi3}.py {flags}
```

For example, to run the ARC AGI task with supervised auxiliary loss for 360 seconds:

```bash
./run_manual.sh torch-rocm tasks/arcagi3.py -svl -hr 0.1
```

## To do

- [x] Implement core AGI algorithms
- [x] Save and load model checkpoints
- [x] Asynchronous external action handle
- [x] External memory module
- [x] Using available options to improve exploration
- [x] Penalize action without any effect
- [x] Memory store reward
- [x] Atari environment
- [x] RL baseline (With simple models, too complex models tend to have high variance and very slow learning.)
- [x] Supervised learning auxiliary loss
- [x] Model 53: Full cognitive ability
    - [x] Use RL to select two observation modes
        - [x] Override content at the predict position
        - [x] Jump to observed position
- [ ] Algebra core
    - [ ] Exchange information between heads
    - [ ] Integrate content module (full Cognitive map or Tolman-Eichenbaum Machine)
    - [ ] Transfer value network from successful tasks to new tasks

## RL Environment Observation Handling: Standard vs. Vectorized

In Reinforcement Learning, handling the transition between episodes correctly is critical for preserving the mathematical validity of the Markov Decision Process (MDP). There is a significant divergence in behavior between **Standard (Single) Environments** and **Vectorized (Parallel) Environments**.

### 1. The Core Distinction

#### Standard Environments (e.g., `gymnasium.Env`)

- **Mechanism:** "Stop and Wait."
- **Behavior:** When `terminated` or `truncated` returns `True`, the environment pauses.
- **Observation:** The `next_obs` returned is the **actual final state** of the episode (e.g., the frame where the agent died).
- **Action:** The user must explicitly call `reset()` to begin the next episode.

#### Vectorized Environments (e.g., `gymnasium.vector.VectorEnv`, `stable_baselines3.VecEnv`)

- **Mechanism:** "Auto-Reset."
- **Behavior:** To maintain a constant batch size for GPU efficiency, sub-environments cannot pause. The wrapper automatically resets any finished environment immediately within the same `step()`.
- **Observation:** The `next_obs` returned is the **initial state (t=0)** of the _new_ episode.
- **Action:** The user does not call reset. The **actual final state** of the old episode is moved to the `info` dictionary to prevent it from being overwritten.

### 2. Comparison Matrix

| Feature                             | Standard Env                                      | Vectorized Env                                           |
| :---------------------------------- | :------------------------------------------------ | :------------------------------------------------------- |
| **Reset Responsibility**            | Manual (User calls `reset()`)                     | Automatic (Wrapper calls `reset()`)                      |
| **Returned `next_obs`**             | **Terminal State** (Old Episode)                  | **Initial State** (New Episode)                          |
| **Correct "Next State" for Buffer** | `next_obs`                                        | `info["final_observation"]`                              |
| **Logic Flow**                      | Step $\rightarrow$ Check Done $\rightarrow$ Reset | Step $\rightarrow$ Check Done $\rightarrow$ Extract Info |

### 3. Implementation Pattern: The "Mask Out" Strategy

Use this pattern **only** when your vectorized environment does **not** provide the true final observation (e.g., in `info['final_observation']`) after a Time Limit (Truncation).

**The Logic:**
Instead of guessing the future value or using the wrong observation (the start of the next episode), we simply **ignore** the specific transition where the time limit occurred. This avoids introducing mathematical bias into the value function.

**Implementation (PyTorch Pseudo-code):**

```python
import torch.nn.functional as F

# Assume batch size of N
# obs: Current observations
# next_obs: The observations returned by step() (Start of NEW episode)
# terminated: [True, False, ...] (Game Over)
# truncated:  [False, True, ...] (Time Limit, missing final obs)

# 1. Calculate Target
with torch.no_grad():
    next_values = agent.get_value(next_obs)

    # Standard Terminated Logic: mask future value if died (gamma=0)
    # We do NOT treat truncated as terminated for the target calculation yet.
    # We assume standard bootstrapping for now, but we will discard it later.
    bootstrap_mask = ~terminated
    target = rewards + gamma * next_values * bootstrap_mask

# 2. Calculate Raw Loss
current_values = agent.get_value(obs)
# Must use reduction='none' to get a loss per sample
loss_elements = F.mse_loss(current_values, target, reduction='none')

# 3. Apply the "Mask Out"
# We want to keep the gradient ONLY if it is NOT truncated.
# If truncated is True, valid_mask becomes 0.
valid_mask = ~truncated

# Zero out the loss for the truncated steps
masked_loss = loss_elements * valid_mask

# 4. Final Reduction
# Divide by the number of valid samples to keep the scale correct.
# Clamp to 1.0 to avoid division by zero if entire batch is truncated.
final_loss = masked_loss.sum() / valid_mask.sum().clamp(min=1.0)

# 5. Backprop
final_loss.backward()
```

### References

- [Basic gym interface](https://gymnasium.farama.org/introduction/basic_usage/)
- [New standard Vector Environments](https://ale.farama.org/vector-environment/)
- [Basic Vector Environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)

## Normalization Techniques in Reinforcement Learning

Most RL models predict not only action values but also their distributions. To encourage sampling, it's common to use normalization layers to prevent the model from collapsing to a sharp peak.
