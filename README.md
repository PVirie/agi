# AGI

An attempt to solve AGI.

## Setup

1. Setup environment in `secrets.env` file, place it in the root directory with the following content:

```env
    export DEBUG="False"
    export RECORDINGS_DIR="/app/artifacts/experiments/recordings"

    export SCHEME="https"
    export HOST="three.arcprize.org"
    export PORT="443"
    export WDM_LOG="0"

    export OPENAI_API_KEY="your_openai_api_key_here"
    export ARC_API_KEY="your_arc_api_key_here"
    export AGENTOPS_API_KEY="your_agentops_api_key_here"
```

## To do

-   [x] Implement core AGI algorithms
-   [x] Save and load model checkpoints
-   [ ] Supervised learning module (Teacher forcing)
-   [ ] Padding environment (for variable input lengths)
