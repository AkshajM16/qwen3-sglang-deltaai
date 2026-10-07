# Qwen3-4B Inference with SGLang on DeltaAI

## Overview

- Goal: run `Qwen/Qwen3-4B` with SGLang on NCSA DeltaAI and send a batch of concurrent inference requests to it.
- SGLang was run using its Docker image through **Apptainer**, since DeltaAI does not run Docker containers directly.
- The model was run on one NVIDIA GH200 GPU.
- `inference.py` sends multiple prompts concurrently through SGLang's OpenAI-compatible API.
- Qwen3 thinking mode is also supported, with reasoning and the final answer printed separately.

## Repository Structure and Files

```text
.
├── inference.py
├── launch_server_container.sh
├── requirements.txt
├── logs/
│   ├── inference_output.txt
│   └── reasoning_output.txt
└── README.md
```

- `inference.py`
  - Sends `N` prompts concurrently to the SGLang server.
  - Supports options such as thinking mode, max tokens, temperature, top-p, and top-k.

- `launch_server_container.sh`
  - Starts the SGLang server inside the Apptainer container.
  - Loads Qwen3-4B and serves it on `127.0.0.1:30000`.

- `logs/`
  - `inference_output.txt`: example output from a normal 4-request inference run.
  - `reasoning_output.txt`: example output with Qwen3 thinking mode enabled.

## Step-by-Step Instructions: Running Inference and DeltaAI Setup

### Running Inference Using the Repository

First, request a GPU and move onto the DeltaAI compute node:

```bash
srun \
  --account=<YOUR_DELTA_AI_ACCOUNT> \
  --partition=ghx4 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --mem=64g \
  --gpus-per-node=1 \
  --time=00:30:00 \
  --pty /bin/bash
```

Start the SGLang server:

```bash
./launch_server_container.sh
```

The server runs at:

```text
http://127.0.0.1:30000
```

In another shell on the same compute node, activate the Python environment:

```bash
source venv/bin/activate
```

Run normal inference:

```bash
python inference.py -n 4
```

`-n` controls how many prompts are sent concurrently.

For example:

```bash
python inference.py -n 8
```

sends 8 concurrent requests.

To run Qwen3 with thinking enabled:

```bash
python inference.py -n 1 --thinking
```

Other inference options can also be changed:

```bash
python inference.py \
  -n 4 \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20
```

### Setting Up DeltaAI to Use the Repository

The following setup only needs to be done once.

Set the storage location:

```bash
export STORAGE_ROOT=/work/nvme/bivc/$USER
```

Pull the SGLang Docker image using Apptainer:

```bash
apptainer pull \
  "$STORAGE_ROOT/containers/sglang/sglang-v0.5.15.post1-cu130.sif" \
  docker://lmsysorg/sglang:v0.5.15.post1-cu130
```

This creates a reusable SGLang `.sif` container.

Download Qwen3-4B into a shared Hugging Face cache:

```bash
export HF_HOME="$STORAGE_ROOT/huggingface"

module load python/miniforge3_pytorch/2.11.0
```

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-4B",
    revision="1cfa9a7208912126459214e8b04321603b3df60c",
)
PY
```

Create the Python environment for `inference.py`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The server uses:

```text
--reasoning-parser qwen3
--attention-backend flashinfer
```

`flashinfer` is used because the default FlashAttention 3 backend was not available in the tested ARM64 SGLang container.

Once the container, model weights, and Python environment are set up, the normal workflow is just:

```text
request GPU
→ run launch_server_container.sh
→ run inference.py
```
