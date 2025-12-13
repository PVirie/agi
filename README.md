# AGI

An attempt to solve AGI.

## Setup

1. Setup environment in `secrets.env` file, place it in the root directory with the following content:

```env
    export SCHEME="https"
    export HOST="three.arcprize.org"
    export PORT="443"

    export ARC_API_KEY="your_arc_api_key_here"
```

## To do

-   [x] Implement core AGI algorithms
-   [x] Save and load model checkpoints
-   [x] Supervised learning module (Teacher forcing)
-   [ ] Padding environment (for variable input lengths)
-   [x] Asynchronous external action handle
