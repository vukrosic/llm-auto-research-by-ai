# RTX 5090 Quantization Test

Real FP4 and FP8 quantization test using manual FP4 implementation and native PyTorch FP8 support.

## Quick Start

```bash
pip install -r requirements.txt
python simple_quantization.py
```

## What it does

- Creates a simple transformer model
- Tests 3 configurations:
  - Baseline (FP32)
  - FP4 quantized (manual 4-bit quantization)
  - FP8 quantized (native PyTorch FP8 or FP16 fallback)
- Compares memory usage, training loss, and speed

## Requirements

- RTX 5090 GPU (or any CUDA GPU)
- PyTorch with CUDA support
- transformers, datasets

## Installation

```bash
pip install -r requirements.txt
```

Simple, reliable quantization test without complex dependencies.