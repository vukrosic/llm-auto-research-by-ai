#!/usr/bin/env python3
"""
Simple FP4/FP8 Quantization Test
================================

Minimal script to test FP4 and FP8 quantization on RTX 5090.
Uses torch.compile with FP8 and manual FP4 quantization.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check CUDA and quantization support
print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU: {torch.cuda.get_device_name()}")

# Check FP8 support (requires PyTorch 2.1+ and compatible GPU)
HAS_FP8 = hasattr(torch, 'float8_e4m3fn') and torch.cuda.is_available()
if HAS_FP8:
    print("✅ Native FP8 support available")
else:
    print("❌ FP8 not supported - requires PyTorch 2.1+ and compatible GPU")

# FP4 will be implemented manually
print("✅ Manual FP4 quantization available")

class FP4Linear(nn.Module):
    """Manual FP4 quantized linear layer"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights as int8 and scale factors
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features, dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def quantize_fp4(self, weight):
        """Quantize weights to 4-bit"""
        # Simple symmetric quantization to 4-bit range [-8, 7]
        scale = weight.abs().max(dim=1, keepdim=True)[0] / 7.0
        scale = scale.clamp(min=1e-8)
        quantized = torch.round(weight / scale).clamp(-8, 7)
        return quantized.to(torch.int8), scale.squeeze()
    
    def dequantize_fp4(self, quantized_weight, scale):
        """Dequantize 4-bit weights back to float"""
        return quantized_weight.float() * scale.unsqueeze(1)
    
    def forward(self, x):
        # Dequantize weights for computation
        weight_fp32 = self.dequantize_fp4(self.weight_quantized, self.weight_scale)
        return F.linear(x, weight_fp32, self.bias)

class SimpleTransformer(nn.Module):
    """Minimal transformer model"""
    def __init__(self, vocab_size=50000, d_model=512, n_heads=8, n_layers=6, quantization=None):
        super().__init__()
        self.d_model = d_model
        self.quantization = quantization
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Standard PyTorch transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Apply quantization after initialization
        if quantization == "fp4":
            self._apply_fp4_quantization()
        elif quantization == "fp8":
            self._apply_fp8_quantization()
    
    def _apply_fp4_quantization(self):
        """Replace linear layers with FP4 quantized versions"""
        def replace_linear_with_fp4(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear) and child.weight.numel() > 1000:
                    # Create FP4 layer
                    fp4_layer = FP4Linear(child.in_features, child.out_features, child.bias is not None)
                    
                    # Quantize and store weights
                    with torch.no_grad():
                        quantized_weight, scale = fp4_layer.quantize_fp4(child.weight.data)
                        fp4_layer.weight_quantized.copy_(quantized_weight)
                        fp4_layer.weight_scale.copy_(scale)
                        if child.bias is not None:
                            fp4_layer.bias.copy_(child.bias.data)
                    
                    # Replace the layer
                    setattr(module, name, fp4_layer)
                else:
                    replace_linear_with_fp4(child)
        
        replace_linear_with_fp4(self)
    
    def _apply_fp8_quantization(self):
        """Convert model to FP8 where supported"""
        if HAS_FP8:
            def convert_to_fp8(module):
                for child in module.children():
                    if isinstance(child, nn.Linear):
                        # Convert weights to FP8
                        child.weight.data = child.weight.data.to(torch.float8_e4m3fn)
                        if child.bias is not None:
                            child.bias.data = child.bias.data.to(torch.float8_e4m3fn)
                    else:
                        convert_to_fp8(child)
            convert_to_fp8(self)
        else:
            # Fallback to FP16
            self.half()
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

def create_baseline_model():
    """Create baseline FP32 model"""
    return SimpleTransformer()

def create_fp4_model():
    """Create model with manual FP4 quantization"""
    return SimpleTransformer(quantization="fp4")

def create_fp8_model():
    """Create model with FP8 quantization"""
    return SimpleTransformer(quantization="fp8")

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
    configs = [
        ("Baseline (FP32)", create_baseline_model),
        ("FP4 (Manual)", create_fp4_model),
    ]
    
    # Add FP8 if supported
    if HAS_FP8:
        configs.append(("FP8 (Native)", create_fp8_model))
    else:
        configs.append(("FP8 (FP16 fallback)", create_fp8_model))
    
    results = []
    
    for name, model_fn in configs:
        print(f"\n🧪 Testing {name}")
        print("-" * 30)
        
        try:
            # Create model
            model = model_fn()
            
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
            
        except Exception as e:
            print(f"❌ Training failed for {name}: {e}")
        
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
    
    # Print summary
    if results:
        print("\n📊 SUMMARY")
        print("=" * 50)
        baseline_memory = results[0]['memory_mb']
        
        for result in results:
            memory_saving = (1 - result['memory_mb'] / baseline_memory) * 100
            print(f"{result['name']:20} | "
                  f"Memory: {result['memory_mb']:6.1f}MB ({memory_saving:+5.1f}%) | "
                  f"Loss: {result['avg_loss']:.4f} | "
                  f"Speed: {result['steps_per_sec']:.2f} steps/sec")

if __name__ == "__main__":
    run_quantization_test()