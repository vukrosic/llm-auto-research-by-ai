# Simple Quantization Test

Minimal script to test FP4 and FP8 quantization on a single transformer model.

## Quick Start

```bash
pip install -r requirements.txt
python simple_quantization.py
```

## What it does

- Creates a simple transformer model
- Tests 3 configurations:
  - Baseline (FP32)
  - FP4 quantized (using bitsandbytes)
  - FP8 quantized (simulated with FP16)
- Compares memory usage, training loss, and speed

## Requirements

- PyTorch
- transformers
- datasets
- bitsandbytes (for FP4 quantization)

That's it. No complex experiments, no dashboards, just a simple quantization comparison.