# AGI

An attempt to solve AGI.

| Generation | Model Name      | Description                |
| ---------- | --------------- | -------------------------- |
| LXXIV      | Graph Automaton | Full Quest Graph Automaton |

## Prerequisites

1.  Install Docker Desktop
    - Linux, please follow [docker-ce](https://www.linode.com/docs/guides/installing-and-using-docker-on-ubuntu-and-debian/)
    - Linux, also add your user to docker group `sudo usermod -aG docker $USER`
    - Windows and Mac, please install [Docker Desktop](https://www.docker.com/products/docker-desktop)

2.  Accelerator support: Follow the installation guide for your machine configuration. I would recommend using Linux for the best experience.
    2.1 CUDA support
    - Nvidia driver version 555.xx or higher (for CUDA 12.5.1+)
    - Linux, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    - Windows, follow [this guide](https://docs.docker.com/desktop/gpu/) to enable gpu support in docker desktop.

        2.2 ROCm support

    - Install [ROCm-kernel (amdgpu-dkms)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)

3.  Setup environment in `secrets.env` file, place it in the root directory with the following content:

    ```env
        export SCHEME="https"
        export HOST="three.arcprize.org"
        export PORT="443"

        export ARC_API_KEY="your_arc_api_key_here"
    ```

## Running experiments

- By default, use program script `./run_manual.sh {configuration} {path to file} {optional flags}` to execute the python file with the selected configuration:

```bash
./run_manual.sh {torch-rocm} tasks/{arcagi3}.py {flags}
```

For example, to run the ARC AGI task with auxiliary loss for 360 seconds:

```bash
./run_manual.sh torch-rocm tasks/arcagi3.py -aux -hr 0.1
```

- To clear the cache and reset the experiment, use `./run_manual.sh {configuration} {path to file} --reset`.

- For VSCode, press `F5` to run the selected configuration:
    - Launch `Container: Run current file` to run the experiment in the opening file.
        - You will also need to choose the configuration from the dropdown list.
            - `torch-cpu` for torch in cpu environment
            - `torch-cuda` for torch in CUDA environment.
            - `torch-rocm` for torch in ROCm environment.
            - `jax-cpu` for jax in cpu environment.
            - `jax-cuda` for jax in CUDA environment.
            - `jax-rocm` for jax in ROCm environment.
        - VSCode will also ask for additional program arguments. Pass nothing if you want to use the default arguments.
    - Launch `Container: Reset` to clear the cache and reset the experiment.

- Running on Windows
    - The relative path for docker launch configuration only support POSIX path separators, which is different from the default path separators in Windows.
    - To fix this, install the [Command Variables extension](https://marketplace.visualstudio.com/items?itemName=rioj7.command-variable) for VSCode to provide environment variable supports in launch configuration.
    - Alternatively, you can also manually _use POSIX path separators_ in the command line when passing `{path to file}` in `run_manual.sh` script, or create a new launch configuration in `.vscode/launch.json` with hard coded POSIX path separators. To enable debugger, add python debugger flags `./run_manual.sh {configuration} -m debugpy --listen 0.0.0.0:43690 --wait-for-client {path to file}` to the end of the command, and then attach the VSCode debugger to the running process with the `Attach debugger only` configuration.

- The program **may fail** to run on the first attempt due to the failure to find package directories. If this happens, run the program again.

### Running plots:

- Plots use graphics therefore they cannot be run in the docker container.
- Create python virtual environment in the root directory of the project.
    - We recommend using `.venv` as the default virtual environment directory, e.g. `python -m venv .venv`.
- Install matplotlib and other dependencies in the virtual environment, e.g. `pip install matplotlib scipy`.
- Run the plot script in the virtual environment, e.g. `python tasks/plot.py`.

## Development

We recommend using [VSCode](https://code.visualstudio.com/) as the IDE for development. It has great support for Python and Docker.

- To assist pylance, add paths to local install python packages in `.vscode/settings.json`:

    ```json
    {
        "python.analysis.extraPaths": ["${workspaceFolder}/.venv/lib/python3.12/site-packages", "${workspaceFolder}/artifacts/pip_modules/lib/python3.12/site-packages"]
    }
    ```

    - We recommend using `.venv` as the default virtual environment directory.
    - Note that when building docker, python packages required for the experiment will be installed under `artifacts/pip_modules` directory. Except for pytorch, which will be installed in the docker image. To fix pylance, either refer to local install pytorch or use virtual environment on top.

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
- [x] Auxiliary loss
    - [x] Supervised content
- [x] Model 53: Full cognitive ability
    - [x] Use RL to select two observation modes
        - [x] Override content at the predict position
        - [x] Jump to observed position
- [x] Model 63: Mem Ops
    - [x] Full memory operations
    - [x] Cache then Fetch stack memory execution
- [x] Model 64: Dualism
- [x] Model 68: Mem Hierarchy
- [x] minigrid environment
    - [x] full MDP
- [x] MAMBA
- [x] Model 71: $\nu$
    - [x] Success factor $\nu$
    - [x] Record $\nu$ in each step for analysis
    - [x] Carry on states
    - [x] Fixed base model to clear position on done, truncated
    - [x] Separate model for high level and low level policy
- [x] Model 74: Graph Memory
    - [x] flipflop dataset
- [ ] Spatio-temporal learning
    - [x] On graph automaton
    - [ ] On fixed environment graph
- [ ] Cognitive map building
    - [ ] "recognized" location linking
    - [ ] External to internal action transfer
