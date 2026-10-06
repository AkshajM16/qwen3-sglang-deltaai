#!/bin/bash
set -e

STORAGE_ROOT="/work/nvme/bivc/$USER"

SIF="$STORAGE_ROOT/containers/sglang/sglang-v0.5.15.post1-cu130.sif"
HF_HOME="$STORAGE_ROOT/huggingface"
MODEL_DIR="$HF_HOME/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

apptainer exec \
    --nv \
    --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
    --env HF_HOME="$HF_HOME" \
    "$SIF" \
    sglang serve \
        --model-path "$MODEL_DIR" \
        --served-model-name Qwen/Qwen3-4B \
        --host 127.0.0.1 \
        --port 30000 \
        --reasoning-parser qwen3 \
        --attention-backend flashinfer