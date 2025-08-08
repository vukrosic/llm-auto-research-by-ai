# RTX 5090 Quantization Test

Real FP4 and FP8 quantization test using NVIDIA Transformer Engine and native PyTorch support.

## Quick Start

```bash
pip install -r requirements.txt
python simple_quantization.py
```

## What it does

- Creates a simple transformer model
- Tests up to 3 configurations:
  - Baseline (FP32)
  - FP4 quantized (using NVIDIA Transformer Engine)
  - FP8 quantized (using native PyTorch FP8 support)
- Compares memory usage, training loss, and speed

## Requirements

- RTX 5090 GPU (or other Hopper/Ada Lovelace GPU)
- PyTorch with CUDA support
- NVIDIA Transformer Engine (for FP4)
- transformers, datasets

## Installation

```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install Transformer Engine
pip install transformer-engine

# Install other requirements
pip install transformers datasets tqdm
```

Real quantization, no simulation. Designed for RTX 5090.