# 3D Reordering

## Overview
- Self-supervised task: slice 3D volumes (e.g., CT) into fixed-length sequences, shuffle their order, and train the model to recover the original ordering.
- Uses an encoder–ConvS5 (Conv-SSM)–decoder stack. ConvS5 in `src/models/sequence_models/noVQ/convS5.py` extracts spatiotemporal representations for long contexts.
- Rather than class labels, the model predicts a one-hot permutation matrix describing the correct ordering, which encourages stronger representations for downstream tasks (e.g., defect detection, classification).

## Directory Layout
- `configs/`: Dataset-specific hyperparameters (see `configs/CT/ct_convS5_novq_slice.yaml`).
- `src/data.py`: CT slice extraction and permutation labeling logic.
- `src/models/`: Encoder/decoder stacks plus ConvS5 sequence modules.
- `scripts.train`: JAX/Flax training entry point (executed via `uv`).
- `run_ct.sh`: Example shell launcher for CT experiments.

## Data Pipeline
1. **Volume Loading**: `load_ct_slices` converts product-level high-resolution binary (`>i2`) files into `(1024,1024,1)` tensors.
2. **Slicing**: Extract sequences of length `slice_seq_len` every `slice_interval` frames (default 10).
3. **Order Shuffling**: Keep the first frame fixed, shuffle the rest, and repeat 10× for augmentation. Store the inverse permutation as a one-hot matrix that serves as the supervision signal.
4. **Normalization**: Divide slices by 150 to keep values within a stable range close to `[-1, 1]`.
5. **Splitting**: Deterministically shuffle with a PRNG key, take 90%/10% for train/test, and shard batches across devices.

## Model Highlights
- **Encoder/Decoder**: Three convolutional stages expanding and contracting channels 16→64 with group norm (`num_groups`).
- **ConvS5 Sequence Model**: Six layers, 128-dim SSM state, 3×3 depthwise kernels, per-layer skip, and optional squeeze-excite.
- **Order Prediction Head**: Outputs `slice_seq_len × slice_seq_len` logits, optimized with permutation cross-entropy (labels supplied via `batch_keys`).

## How to Run
```bash
cd ~/3D-reordering
chmod +x run_ct.sh
./run_ct.sh
```
- `run_ct.sh` wraps `uv run -m scripts.train` with the CT configuration.
- Key flags:
  - `-d`: Raw CT data directory
  - `-o`: Output directory for checkpoints, logs, and visualizations.
  - `-c`: Config file path.
