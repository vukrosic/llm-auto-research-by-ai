#!/usr/bin/env python3
"""
Simple FP4/FP8 Quantization Test
================================

Minimal script to test FP4 and FP8 quantization on a single model.
No complex experiments, no GQA, just basic quantization comparison.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("❌ bitsandbytes not available. Install with: pip install bitsandbytes")

class SimpleTransformer(nn.Module):
    """Minimal transformer model"""
    def __init__(self, vocab_size=50000, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

def apply_fp4_quantization(model):
    """Apply FP4 quantization to linear layers"""
    if not HAS_BNB:
        print("❌ Cannot apply FP4 quantization - bitsandbytes not available")
        return model
        
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
            # Replace with 4-bit quantized version
            quantized_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float16,
                quant_type="nf4"
            )
            # Copy weights
            quantized_layer.weight.data = module.weight.data
            if module.bias is not None:
                quantized_layer.bias.data = module.bias.data
            
            # Replace the module
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], quantized_layer)
    
    return model

def apply_fp8_quantization(model):
    """Apply FP8 quantization (simulated with FP16 for now)"""
    # Note: True FP8 requires specific hardware support
    # This is a placeholder that converts to FP16 as approximation
    model = model.half()  # Convert to FP16 as FP8 approximation
    return model

def get_model_memory(model):
    """Get model memory usage in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 * 1024)

def simple_training_step(model, batch, optimizer):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def prepare_data():
    """Load and prepare simple dataset"""
    print("📚 Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    # Simple tokenization
    texts = []
    for i, item in enumerate(dataset):
        if i >= 100:  # Just 100 samples for simplicity
            break
        texts.append(item['text'][:512])  # Truncate to 512 chars
    
    # Tokenize
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return tokens['input_ids']

def run_quantization_test():
    """Run simple quantization comparison"""
    print("🚀 Starting Simple Quantization Test")
    print("=" * 50)
    
    # Prepare data
    data = prepare_data()
    print(f"📊 Data shape: {data.shape}")
    
    # Test configurations
    configs = [
        ("Baseline (FP32)", lambda m: m),
        ("FP4 Quantized", apply_fp4_quantization),
        ("FP8 Quantized", apply_fp8_quantization),
    ]
    
    results = []
    
    for name, quantize_fn in configs:
        print(f"\n🧪 Testing {name}")
        print("-" * 30)
        
        # Create model
        model = SimpleTransformer()
        model = quantize_fn(model)
        
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()
        
        # Measure memory
        memory_mb = get_model_memory(model)
        print(f"💾 Model memory: {memory_mb:.1f} MB")
        
        # Simple training test
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        start_time = time.time()
        total_loss = 0
        num_steps = 10
        
        for step in tqdm(range(num_steps), desc="Training"):
            # Simple batch
            batch_size = min(8, data.size(0))
            batch = data[:batch_size]
            
            loss = simple_training_step(model, batch, optimizer)
            total_loss += loss
        
        avg_loss = total_loss / num_steps
        elapsed = time.time() - start_time
        
        print(f"📈 Average loss: {avg_loss:.4f}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print(f"🚀 Speed: {num_steps/elapsed:.2f} steps/sec")
        
        results.append({
            'name': name,
            'memory_mb': memory_mb,
            'avg_loss': avg_loss,
            'time_sec': elapsed,
            'steps_per_sec': num_steps/elapsed
        })
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n📊 SUMMARY")
    print("=" * 50)
    baseline_memory = results[0]['memory_mb']
    
    for result in results:
        memory_saving = (1 - result['memory_mb'] / baseline_memory) * 100
        print(f"{result['name']:15} | "
              f"Memory: {result['memory_mb']:6.1f}MB ({memory_saving:+5.1f}%) | "
              f"Loss: {result['avg_loss']:.4f} | "
              f"Speed: {result['steps_per_sec']:.2f} steps/sec")

if __name__ == "__main__":
    run_quantization_test()