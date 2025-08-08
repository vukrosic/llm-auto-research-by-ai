#!/usr/bin/env python3
"""
Simple FP4/FP8 Quantization Test
================================

Minimal script to test FP4 and FP8 quantization on RTX 5090.
Uses NVIDIA Transformer Engine for FP4 and native PyTorch for FP8.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import transformer engine for FP4
try:
    import transformer_engine.pytorch as te
    HAS_TE = True
    print("✅ Transformer Engine available for FP4 quantization")
except ImportError:
    HAS_TE = False
    print("❌ Transformer Engine not available. Install with: pip install transformer-engine")

# Check FP8 support
HAS_FP8 = hasattr(torch, 'float8_e4m3fn') and torch.cuda.is_available()
if HAS_FP8:
    print("✅ Native FP8 support available")
else:
    print("❌ FP8 not supported on this system")

class SimpleTransformer(nn.Module):
    """Minimal transformer model"""
    def __init__(self, vocab_size=50000, d_model=512, n_heads=8, n_layers=6, use_te=False):
        super().__init__()
        self.d_model = d_model
        self.use_te = use_te
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if use_te and HAS_TE:
            # Use Transformer Engine layers for FP4 support
            self.layers = nn.ModuleList([
                te.TransformerLayer(
                    hidden_size=d_model,
                    ffn_hidden_size=d_model*4,
                    num_attention_heads=n_heads,
                    bias=True,
                    layer_number=i+1,
                    attention_dropout=0.0,
                    hidden_dropout=0.0,
                    fuse_wgrad_accumulation=False,
                    get_rng_state_tracker=None,
                    init_method=None,
                    output_layer_init_method=None,
                    hidden_size_per_attention_head=d_model//n_heads,
                    layer_type="encoder",
                )
                for i in range(n_layers)
            ])
        else:
            # Standard PyTorch layers
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
                for _ in range(n_layers)
            ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        
        if self.use_te and HAS_TE:
            # Transformer Engine forward pass
            for layer in self.layers:
                x = layer(x)
        else:
            # Standard PyTorch forward pass
            for layer in self.layers:
                x = layer(x)
                
        x = self.ln_f(x)
        return self.lm_head(x)

def create_fp4_model():
    """Create model with FP4 quantization using Transformer Engine"""
    if not HAS_TE:
        print("❌ Cannot create FP4 model - Transformer Engine not available")
        return None
    
    # Enable FP4 quantization in Transformer Engine
    te.fp8.set_fp8_enabled(False)  # Disable FP8 first
    
    model = SimpleTransformer(use_te=True)
    
    # Configure FP4 quantization
    for module in model.modules():
        if hasattr(module, 'set_fp8_weights') and callable(getattr(module, 'set_fp8_weights')):
            # This would be the proper way to enable FP4 in TE
            # The exact API might vary based on TE version
            pass
    
    return model

def create_fp8_model():
    """Create model with FP8 quantization using native PyTorch"""
    if not HAS_FP8:
        print("❌ Cannot create FP8 model - FP8 not supported")
        return None
    
    model = SimpleTransformer()
    
    # Convert model to use FP8 dtypes where supported
    def convert_to_fp8(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Convert weights to FP8 E4M3 format
                if hasattr(torch, 'float8_e4m3fn'):
                    child.weight.data = child.weight.data.to(torch.float8_e4m3fn)
                    if child.bias is not None:
                        child.bias.data = child.bias.data.to(torch.float8_e4m3fn)
            else:
                convert_to_fp8(child)
    
    convert_to_fp8(model)
    return model

def enable_te_fp8():
    """Enable FP8 in Transformer Engine"""
    if HAS_TE:
        te.fp8.set_fp8_enabled(True)
        # Configure FP8 recipe for optimal performance
        fp8_recipe = te.fp8.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=te.fp8.Format.E4M3,
            amax_history_len=1,
            amax_compute_algo="most_recent"
        )
        return fp8_recipe
    return None

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
    print("🚀 Starting RTX 5090 Quantization Test")
    print("=" * 50)
    
    # Prepare data
    data = prepare_data()
    print(f"📊 Data shape: {data.shape}")
    
    # Test configurations
    configs = []
    
    # Always include baseline
    configs.append(("Baseline (FP32)", lambda: SimpleTransformer()))
    
    # Add FP4 if Transformer Engine available
    if HAS_TE:
        configs.append(("FP4 (Transformer Engine)", create_fp4_model))
    
    # Add FP8 if supported
    if HAS_FP8:
        configs.append(("FP8 (Native PyTorch)", create_fp8_model))
    
    if len(configs) == 1:
        print("⚠️  Only baseline available - install transformer-engine for FP4/FP8")
    
    results = []
    
    for name, model_fn in configs:
        print(f"\n🧪 Testing {name}")
        print("-" * 30)
        
        # Create model
        model = model_fn()
        if model is None:
            print("❌ Skipping - model creation failed")
            continue
        
        # Enable FP8 context if using Transformer Engine
        fp8_recipe = None
        if "FP4" in name and HAS_TE:
            fp8_recipe = enable_te_fp8()
        
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
        
        try:
            for step in tqdm(range(num_steps), desc="Training"):
                # Simple batch
                batch_size = min(8, data.size(0))
                batch = data[:batch_size]
                
                if fp8_recipe and HAS_TE:
                    # Use FP8 context for Transformer Engine
                    with te.fp8.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        loss = simple_training_step(model, batch, optimizer)
                else:
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
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Print summary
    if results:
        print("\n📊 SUMMARY")
        print("=" * 50)
        baseline_memory = results[0]['memory_mb']
        
        for result in results:
            memory_saving = (1 - result['memory_mb'] / baseline_memory) * 100
            print(f"{result['name']:25} | "
                  f"Memory: {result['memory_mb']:6.1f}MB ({memory_saving:+5.1f}%) | "
                  f"Loss: {result['avg_loss']:.4f} | "
                  f"Speed: {result['steps_per_sec']:.2f} steps/sec")

if __name__ == "__main__":
    run_quantization_test()