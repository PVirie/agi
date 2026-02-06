# AGI

An attempt to solve AGI.

| Generation | Model Name        | Description                                                      |
| ---------- | ----------------- | ---------------------------------------------------------------- |
| LIII       | Cognitive algebra | A learnable computationally universal model.                     |
| LXI        | Fast and slow     | Using fast learnable module to capture fast position assignment. |
| LXII       | Rapid papagation  | Extend fast position assignment to next step                     |

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
    - [x] Multiple games
- [x] RL baseline (With simple models, too complex models tend to have high variance and very slow learning.)
- [x] Supervised learning auxiliary loss
- [x] Model 53: Full cognitive ability
    - [x] Use RL to select two observation modes
        - [x] Override content at the predict position
        - [x] Jump to observed position
- [ ] Algebra core
    - [x] Feature similarity position inference
    - [ ] Extract position mechanics (full Cognitive map or Tolman-Eichenbaum Machine)
- [ ] Experiments
    - [ ] Algebra core alone performance
    - [ ] Algebra core transfer training performance
    - [ ] Game mechanic similarity analysis
