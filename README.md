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

## Run

There are two approaches to run the AGI agent:

1. Use vscode debugger with the provided launch configuration.
2. Run using the shell script:

```bash
./run_manual.sh {torch-rocm} tasks/{arcagi3}.py {flags}
```

For example, to run the ARC AGI task with supervised learning for 360 seconds:

```bash
./run_manual.sh torch-rocm tasks/arcagi3.py -svl -hr 0.1
```

## To do

-   [x] Implement core AGI algorithms
-   [x] Save and load model checkpoints
-   [x] Supervised learning module (Teacher forcing)
-   [x] Asynchronous external action handle
-   [x] External memory module
