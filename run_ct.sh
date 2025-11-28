#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
# export XLA_PYTHON_CLIENT_MEM_FRACTION=1
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS}"

uv run -m scripts.train \
-d ~/base/DATA/KITECH/BORGWARNER/ \
-o results/ct_convS5_novq_slice_10 \
-c configs/CT/ct_convS5_novq_slice.yaml