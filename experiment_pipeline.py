#!/usr/bin/env python3
"""
RTX 5090 Optimization Experiment Pipeline
=========================================

Automated pipeline for testing various optimization techniques:
- FP4 quantization (selective parameter quantization)
- Grouped Query Attention (GQA)
- Memory optimization strategies
- Auto parameter search to maximize GPU utilization

Run with: python experiment_pipeline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import time
import os
import gc
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import psutil
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings('ignore')

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️ pynvml not available. Install with: pip install pynvml")
    print("   GPU utilization monitoring will use fallback method")

# Import base components
from llm import (
    ModelConfig, MinimalLLM, TextTokenDataset, load_and_cache_data,
    evaluate_model, Muon, set_seed, zeropower_via_newtonschulz5
)

@dataclass
class ExperimentConfig:
    """Configuration for optimization experiments"""
    # Base model configs to test
    base_configs: List[Dict] = None
    
    # Optimization techniques
    use_fp4_quantization: bool = True
    use_grouped_query_attention: bool = True
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # Memory optimization
    max_memory_gb: float = 24.0  # RTX 5090 memory
    memory_safety_margin: float = 0.85  # Use 85% of available memory
    
    # Training constraints
    max_training_time_minutes: int = 30  # Fair comparison time limit
    warmup_steps_ratio: float = 0.1
    
    # Experiment settings
    num_experiments: int = 5
    results_dir: str = "experiment_results"
    
    @classmethod
    def from_json(cls, config_path: str = "experiment_config.json"):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract relevant fields
            instance = cls()
            instance.max_memory_gb = config_data["hardware"]["max_memory_gb"]
            instance.memory_safety_margin = config_data["hardware"]["memory_safety_margin"]
            instance.max_training_time_minutes = config_data["training"]["max_training_time_minutes"]
            instance.warmup_steps_ratio = config_data["training"]["warmup_steps_ratio"]
            instance.use_fp4_quantization = config_data["optimizations"]["use_fp4_quantization"]
            instance.use_grouped_query_attention = config_data["optimizations"]["use_grouped_query_attention"]
            instance.use_flash_attention = config_data["optimizations"]["use_flash_attention"]
            instance.use_gradient_checkpointing = config_data["optimizations"]["use_gradient_checkpointing"]
            instance.results_dir = config_data["output"]["results_dir"]
            
            # Convert model configs
            instance.base_configs = []
            for model_config in config_data["model_configs"]:
                config_dict = {k: v for k, v in model_config.items() if k != "name"}
                instance.base_configs.append(config_dict)
                
            return instance
            
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found, using defaults")
            return cls()
    
    def __post_init__(self):
        if self.base_configs is None:
            self.base_configs = [
                # Small efficient model
                {"d_model": 512, "n_heads": 8, "n_layers": 8, "d_ff": 2048, "batch_size": 32},
                # Medium model
                {"d_model": 768, "n_heads": 12, "n_layers": 12, "d_ff": 3072, "batch_size": 24},
                # Large model (memory constrained)
                {"d_model": 1024, "n_heads": 16, "n_layers": 16, "d_ff": 4096, "batch_size": 16},
                # Very large model (aggressive optimization needed)
                {"d_model": 1280, "n_heads": 20, "n_layers": 20, "d_ff": 5120, "batch_size": 12},
            ]

class MemoryMonitor:
    """Monitor GPU memory usage during training"""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        
    def update(self):
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1e9
            self.peak_memory = max(self.peak_memory, self.current_memory)
            
    def get_stats(self) -> Dict[str, float]:
        return {
            "current_memory_gb": self.current_memory,
            "peak_memory_gb": self.peak_memory,
            "memory_utilization": self.peak_memory / 24.0  # RTX 5090 has 24GB
        }
        
    def reset(self):
        self.peak_memory = 0
        self.current_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

class DynamicResourceOptimizer:
    """Dynamically optimize hyperparameters to maximize GPU utilization"""
    
    def __init__(self, target_memory_utilization: float = 0.90, target_gpu_utilization: float = 0.95):
        self.target_memory_util = target_memory_utilization
        self.target_gpu_util = target_gpu_utilization
        self.adjustment_history = []
        
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return util.gpu / 100.0
            except:
                pass
        
        # Fallback: estimate from memory usage
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        return 0.0
    
    def get_memory_utilization(self) -> float:
        """Get current memory utilization percentage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        return 0.0
    
    def optimize_batch_size_and_seq_len(self, model_config: ModelConfig, model: nn.Module, 
                                      sample_batch, device, use_gqa: bool = False, 
                                      n_kv_heads: Optional[int] = None) -> Tuple[int, int]:
        """Dynamically find optimal batch size and sequence length"""
        
        print("🔧 Optimizing batch size and sequence length for maximum GPU utilization...")
        
        original_batch_size = model_config.batch_size
        original_seq_len = model_config.max_seq_len
        
        best_batch_size = original_batch_size
        best_seq_len = original_seq_len
        best_throughput = 0
        
        # Test different combinations
        batch_sizes = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
        seq_lengths = [256, 384, 512, 640, 768, 896, 1024]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                try:
                    # Clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Create test batch
                    test_x = torch.randint(0, model_config.vocab_size, (batch_size, seq_len)).to(device)
                    test_y = torch.randint(0, model_config.vocab_size, (batch_size, seq_len)).to(device)
                    
                    # Test forward pass
                    model.train()
                    start_time = time.time()
                    
                    with torch.cuda.amp.autocast():
                        logits = model(test_x)
                        loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), test_y.view(-1))
                        loss.backward()
                    
                    torch.cuda.synchronize()
                    forward_time = time.time() - start_time
                    
                    # Check memory usage
                    memory_util = self.get_memory_utilization()
                    gpu_util = self.get_gpu_utilization()
                    
                    # Calculate throughput (tokens per second)
                    tokens_processed = batch_size * seq_len
                    throughput = tokens_processed / forward_time
                    
                    # Check if this configuration is viable
                    if memory_util < self.target_memory_util and gpu_util > 0.5:  # Minimum GPU usage
                        if throughput > best_throughput:
                            best_throughput = throughput
                            best_batch_size = batch_size
                            best_seq_len = seq_len
                            
                        print(f"   ✓ Batch: {batch_size}, SeqLen: {seq_len}, "
                              f"Memory: {memory_util:.1%}, GPU: {gpu_util:.1%}, "
                              f"Throughput: {throughput:.0f} tok/s")
                    else:
                        if memory_util >= self.target_memory_util:
                            # Memory limit reached, try smaller configurations
                            break
                            
                    # Clear gradients
                    model.zero_grad()
                    del test_x, test_y, logits, loss
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # OOM, try smaller configurations
                        torch.cuda.empty_cache()
                        break
                    else:
                        print(f"   ❌ Error with batch: {batch_size}, seq_len: {seq_len}: {e}")
                        continue
                        
        print(f"🎯 Optimal configuration: Batch size: {best_batch_size}, Seq length: {best_seq_len}")
        print(f"   Expected throughput: {best_throughput:.0f} tokens/s")
        
        return best_batch_size, best_seq_len
    
    def monitor_and_adjust_during_training(self, step: int, total_steps: int) -> Dict[str, Any]:
        """Monitor resource usage during training and suggest adjustments"""
        
        memory_util = self.get_memory_utilization()
        gpu_util = self.get_gpu_utilization()
        
        suggestions = {
            "memory_utilization": memory_util,
            "gpu_utilization": gpu_util,
            "adjustments": []
        }
        
        # Memory utilization feedback
        if memory_util < 0.7:
            suggestions["adjustments"].append("Consider increasing batch size or sequence length")
        elif memory_util > 0.95:
            suggestions["adjustments"].append("Memory usage high - consider reducing batch size")
            
        # GPU utilization feedback
        if gpu_util < 0.8:
            suggestions["adjustments"].append("GPU underutilized - consider larger model or batch size")
        elif gpu_util > 0.98:
            suggestions["adjustments"].append("GPU fully utilized - optimal performance")
            
        return suggestions

def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory information"""
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}
        
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    free = total - cached
    
    return {
        "total_gb": total,
        "allocated_gb": allocated,
        "cached_gb": cached,
        "free_gb": free
    }

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention implementation for memory efficiency"""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        from llm import Rotary
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project to Q, K, V
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [B, n_heads, seq_len, d_k]
        K = K.transpose(1, 2)  # [B, n_kv_heads, seq_len, d_k]
        V = V.transpose(1, 2)  # [B, n_kv_heads, seq_len, d_k]
        
        # Apply rotary embeddings
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)
        
        # Expand K, V to match Q's head dimension
        K = K.repeat_interleave(self.group_size, dim=1)  # [B, n_heads, seq_len, d_k]
        V = V.repeat_interleave(self.group_size, dim=1)  # [B, n_heads, seq_len, d_k]
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class OptimizedTransformerBlock(nn.Module):
    """Transformer block with optimization options"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, 
                 dropout: float = 0.1, use_gqa: bool = False, n_kv_heads: Optional[int] = None,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if use_gqa and n_kv_heads:
            self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len, dropout)
        else:
            from llm import MultiHeadAttention
            self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
            
        from llm import FeedForward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
            
    def _forward_impl(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class OptimizedLLM(nn.Module):
    """Optimized LLM with various efficiency techniques"""
    
    def __init__(self, config: ModelConfig, use_gqa: bool = False, n_kv_heads: Optional[int] = None,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.use_gqa = use_gqa
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        
        # Calculate n_kv_heads for GQA if not provided
        if use_gqa and n_kv_heads is None:
            # Use 1/4 of the query heads for key-value heads (common ratio)
            n_kv_heads = max(1, config.n_heads // 4)
            
        self.transformer_blocks = nn.ModuleList([
            OptimizedTransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.max_seq_len, 
                config.dropout, use_gqa, n_kv_heads, use_gradient_checkpointing
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def apply_fp4_quantization(model: nn.Module, quantize_embeddings: bool = False) -> nn.Module:
    """
    Apply FP4 quantization to suitable parameters
    
    FP4 works best for:
    - Linear layer weights (not biases)
    - Large parameter tensors
    - Not suitable for: embeddings, layer norms, small tensors
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        print("⚠️ bitsandbytes not available, skipping FP4 quantization")
        return model
        
    quantized_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip certain layers that don't benefit from quantization
            skip_conditions = [
                'lm_head' in name,  # Output projection
                'norm' in name,     # Layer norms
                module.weight.numel() < 1000,  # Very small layers
            ]
            
            if not quantize_embeddings and 'embedding' in name:
                skip_conditions.append(True)
                
            if not any(skip_conditions):
                # Replace with 4-bit quantized linear layer
                quantized_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                    quant_type="fp4"
                )
                
                # Copy weights
                with torch.no_grad():
                    quantized_layer.weight.data = module.weight.data
                    if module.bias is not None:
                        quantized_layer.bias.data = module.bias.data
                        
                # Replace the module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, quantized_layer)
                else:
                    setattr(model, child_name, quantized_layer)
                    
                quantized_layers.append(name)
                
    print(f"🔧 Applied FP4 quantization to {len(quantized_layers)} layers: {quantized_layers}")
    return model

def estimate_memory_usage(config: ModelConfig, use_gqa: bool = False, n_kv_heads: Optional[int] = None) -> float:
    """Estimate memory usage in GB for a given configuration"""
    
    # Parameter count estimation
    vocab_size = config.vocab_size or 50000
    
    # Embedding parameters
    embedding_params = vocab_size * config.d_model
    
    # Transformer block parameters
    if use_gqa and n_kv_heads:
        # GQA reduces K,V projection parameters
        attention_params = (
            config.d_model * config.d_model +  # Q projection
            config.d_model * n_kv_heads * (config.d_model // config.n_heads) * 2 +  # K,V projections
            config.d_model * config.d_model  # Output projection
        )
    else:
        # Standard multi-head attention
        attention_params = config.d_model * config.d_model * 4  # Q,K,V,O projections
        
    ff_params = config.d_model * config.d_ff * 2  # Two linear layers
    norm_params = config.d_model * 2  # Two RMSNorm layers
    
    block_params = attention_params + ff_params + norm_params
    total_params = embedding_params + block_params * config.n_layers + config.d_model  # Final norm
    
    # Memory estimation (rough)
    # Parameters: 4 bytes per param (fp32) or 2 bytes (fp16) or 0.5 bytes (fp4 for some)
    param_memory = total_params * 2 / 1e9  # Assume fp16 on average
    
    # Activations (depends on batch size and sequence length)
    activation_memory = (
        config.batch_size * config.max_seq_len * config.d_model * config.n_layers * 4 / 1e9
    )
    
    # Optimizer states (roughly 2x parameters for Adam-like optimizers)
    optimizer_memory = param_memory * 2
    
    total_memory = param_memory + activation_memory + optimizer_memory
    
    return total_memory

def find_optimal_batch_size(config: ModelConfig, max_memory_gb: float, use_gqa: bool = False, 
                          n_kv_heads: Optional[int] = None) -> int:
    """Find the largest batch size that fits in memory"""
    
    min_batch = 1
    max_batch = 128
    optimal_batch = config.batch_size
    
    for batch_size in range(min_batch, max_batch + 1, 2):
        test_config = ModelConfig(**{**asdict(config), 'batch_size': batch_size})
        estimated_memory = estimate_memory_usage(test_config, use_gqa, n_kv_heads)
        
        if estimated_memory <= max_memory_gb * 0.85:  # Safety margin
            optimal_batch = batch_size
        else:
            break
            
    return optimal_batch

class ExperimentRunner:
    """Main experiment runner class"""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.exp_config = exp_config
        self.results = []
        
        # Create results directory
        os.makedirs(exp_config.results_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(exp_config.results_dir, "experiment_log.txt")
        
    def log(self, message: str):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
            
    def run_single_experiment(self, exp_id: int, base_config: Dict, 
                            use_gqa: bool = False, use_fp4: bool = False) -> Dict[str, Any]:
        """Run a single optimization experiment"""
        
        self.log(f"\n🧪 Starting Experiment {exp_id}")
        self.log(f"   Config: {base_config}")
        self.log(f"   GQA: {use_gqa}, FP4: {use_fp4}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        memory_monitor = MemoryMonitor()
        memory_monitor.reset()
        
        # Initialize dynamic resource optimizer
        resource_optimizer = DynamicResourceOptimizer()
        
        try:
            # Create model config
            model_config = ModelConfig(**base_config)
            
            # Create model first for dynamic optimization
            n_kv_heads = max(1, model_config.n_heads // 4) if use_gqa else None
            set_seed(42)
            model = OptimizedLLM(
                model_config, 
                use_gqa=use_gqa, 
                n_kv_heads=n_kv_heads,
                use_gradient_checkpointing=self.exp_config.use_gradient_checkpointing
            )
            
            # Apply FP4 quantization if requested
            if use_fp4:
                model = apply_fp4_quantization(model, quantize_embeddings=False)
                
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Dynamic optimization of batch size and sequence length
            sample_batch = torch.randint(0, model_config.vocab_size, (1, model_config.max_seq_len)).to(device)
            optimal_batch, optimal_seq_len = resource_optimizer.optimize_batch_size_and_seq_len(
                model_config, model, sample_batch, device, use_gqa, n_kv_heads
            )
            
            # Update model config with optimized values
            model_config.batch_size = optimal_batch
            model_config.max_seq_len = optimal_seq_len
            
            self.log(f"   🎯 Optimized batch size: {optimal_batch}, seq length: {optimal_seq_len}")
            
            # Load data (cached)
            texts, tokenizer, tokens = load_and_cache_data(model_config)
            dataset = TextTokenDataset(tokens, model_config.max_seq_len)
            
            # Train/val split
            val_size = len(dataset) // 10
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, 
                                    shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size, 
                                  shuffle=False, num_workers=2, pin_memory=True)
            
            # Create model
            set_seed(42)
            model = OptimizedLLM(
                model_config, 
                use_gqa=use_gqa, 
                n_kv_heads=n_kv_heads,
                use_gradient_checkpointing=self.exp_config.use_gradient_checkpointing
            )
            
            # Apply FP4 quantization if requested
            if use_fp4:
                model = apply_fp4_quantization(model, quantize_embeddings=False)
                
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            self.log(f"   Total parameters: {total_params:,}")
            
            memory_monitor.update()
            
            # Setup optimizer
            muon_params = []
            adamw_params = []
            
            for name, param in model.named_parameters():
                if (param.ndim == 2 and 
                    'token_embedding' not in name and 
                    'norm' not in name and 
                    param.requires_grad):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
                    
            muon_optimizer = Muon(muon_params, lr=model_config.muon_lr, momentum=0.95)
            adamw_optimizer = torch.optim.AdamW(adamw_params, lr=model_config.muon_lr*0.1, 
                                              weight_decay=model_config.weight_decay)
            optimizers = [muon_optimizer, adamw_optimizer]
            
            # Training with time limit
            start_time = time.time()
            max_training_seconds = self.exp_config.max_training_time_minutes * 60
            
            model.train()
            step = 0
            total_loss = 0
            total_tokens = 0
            
            # Calculate max steps based on time limit
            estimated_steps_per_second = 2  # Conservative estimate
            max_steps = min(model_config.max_steps, 
                          int(max_training_seconds * estimated_steps_per_second))
            
            pbar = tqdm(total=max_steps, desc=f"Exp {exp_id}")
            
            resource_stats = []
            
            while step < max_steps and (time.time() - start_time) < max_training_seconds:
                for batch_idx, (x, y) in enumerate(train_loader):
                    if step >= max_steps or (time.time() - start_time) >= max_training_seconds:
                        break
                        
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    
                    # Forward pass
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))
                    loss.backward()
                    
                    # Optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    total_loss += loss.item()
                    total_tokens += y.numel()
                    
                    memory_monitor.update()
                    
                    # Monitor resource usage every 50 steps
                    if step % 50 == 0:
                        resource_info = resource_optimizer.monitor_and_adjust_during_training(step, max_steps)
                        resource_stats.append({
                            'step': step,
                            'memory_util': resource_info['memory_utilization'],
                            'gpu_util': resource_info['gpu_utilization'],
                            'suggestions': resource_info['adjustments']
                        })
                    
                    if step % 100 == 0:
                        avg_loss = total_loss / max(1, step + 1)
                        mem_util = resource_optimizer.get_memory_utilization()
                        gpu_util = resource_optimizer.get_gpu_utilization()
                        
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'mem': f'{memory_monitor.current_memory:.1f}GB ({mem_util:.0%})',
                            'gpu': f'{gpu_util:.0%}'
                        })
                        
                    step += 1
                    pbar.update(1)
                    
            pbar.close()
            
            training_time = time.time() - start_time
            
            # Final evaluation
            final_eval = evaluate_model(model, val_loader, model_config)
            memory_stats = memory_monitor.get_stats()
            
            # Calculate average resource utilization
            avg_memory_util = sum(r['memory_util'] for r in resource_stats) / max(1, len(resource_stats))
            avg_gpu_util = sum(r['gpu_util'] for r in resource_stats) / max(1, len(resource_stats))
            
            # Collect results
            result = {
                'experiment_id': exp_id,
                'config': base_config,
                'optimizations': {
                    'use_gqa': use_gqa,
                    'use_fp4': use_fp4,
                    'n_kv_heads': n_kv_heads,
                    'gradient_checkpointing': self.exp_config.use_gradient_checkpointing
                },
                'model_stats': {
                    'total_parameters': total_params,
                    'optimal_batch_size': optimal_batch,
                    'optimal_seq_length': optimal_seq_len
                },
                'training_stats': {
                    'training_time_seconds': training_time,
                    'steps_completed': step,
                    'avg_loss': total_loss / max(1, step),
                    'tokens_processed': total_tokens
                },
                'evaluation': final_eval,
                'memory_stats': memory_stats,
                'resource_utilization': {
                    'avg_memory_utilization': avg_memory_util,
                    'avg_gpu_utilization': avg_gpu_util,
                    'peak_memory_utilization': memory_stats['memory_utilization'],
                    'resource_optimization_applied': True
                },
                'throughput': {
                    'steps_per_second': step / training_time,
                    'tokens_per_second': total_tokens / training_time
                },
                'resource_monitoring': resource_stats[-5:] if resource_stats else []  # Last 5 measurements
            }
            
            self.log(f"   ✅ Completed - Loss: {final_eval['val_loss']:.4f}, "
                    f"Memory: {memory_stats['peak_memory_gb']:.1f}GB ({avg_memory_util:.0%}), "
                    f"GPU: {avg_gpu_util:.0%}, Time: {training_time:.1f}s, "
                    f"Throughput: {total_tokens / training_time:.0f} tok/s")
            
            return result
            
        except Exception as e:
            self.log(f"   ❌ Failed: {str(e)}")
            return {
                'experiment_id': exp_id,
                'config': base_config,
                'error': str(e),
                'status': 'failed'
            }
            
    def run_all_experiments(self):
        """Run all optimization experiments"""
        
        self.log("🚀 Starting RTX 5090 Optimization Experiments")
        self.log(f"   GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        self.log(f"   Memory: {get_gpu_memory_info()}")
        
        exp_id = 0
        
        for base_config in self.exp_config.base_configs:
            # Baseline (no optimizations)
            result = self.run_single_experiment(exp_id, base_config, use_gqa=False, use_fp4=False)
            self.results.append(result)
            exp_id += 1
            
            # GQA only
            if self.exp_config.use_grouped_query_attention:
                result = self.run_single_experiment(exp_id, base_config, use_gqa=True, use_fp4=False)
                self.results.append(result)
                exp_id += 1
                
            # FP4 only
            if self.exp_config.use_fp4_quantization:
                result = self.run_single_experiment(exp_id, base_config, use_gqa=False, use_fp4=True)
                self.results.append(result)
                exp_id += 1
                
            # Both optimizations
            if self.exp_config.use_grouped_query_attention and self.exp_config.use_fp4_quantization:
                result = self.run_single_experiment(exp_id, base_config, use_gqa=True, use_fp4=True)
                self.results.append(result)
                exp_id += 1
                
        # Save results
        self.save_results()
        self.generate_report()
        
    def save_results(self):
        """Save experiment results to JSON"""
        results_file = os.path.join(self.exp_config.results_dir, "experiment_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.log(f"💾 Results saved to {results_file}")
        
    def generate_report(self):
        """Generate a summary report"""
        report_file = os.path.join(self.exp_config.results_dir, "experiment_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# RTX 5090 Optimization Experiment Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Experiment Summary\n\n")
            f.write("| ID | Config | GQA | FP4 | Params | Batch | Val Loss | Memory (GB) | Time (s) | Tokens/s |\n")
            f.write("|----|----|----|----|----|----|----|----|----|\n")
            
            for result in self.results:
                if 'error' not in result:
                    config_str = f"{result['config']['d_model']}d-{result['config']['n_layers']}L"
                    gqa = "✓" if result['optimizations']['use_gqa'] else "✗"
                    fp4 = "✓" if result['optimizations']['use_fp4'] else "✗"
                    params = f"{result['model_stats']['total_parameters']:,}"
                    batch = result['model_stats']['optimal_batch_size']
                    val_loss = f"{result['evaluation']['val_loss']:.4f}"
                    memory = f"{result['memory_stats']['peak_memory_gb']:.1f}"
                    time_s = f"{result['training_stats']['training_time_seconds']:.1f}"
                    tokens_s = f"{result['throughput']['tokens_per_second']:.0f}"
                    
                    f.write(f"| {result['experiment_id']} | {config_str} | {gqa} | {fp4} | {params} | {batch} | {val_loss} | {memory} | {time_s} | {tokens_s} |\n")
                    
            # Best results
            f.write("\n## Best Results\n\n")
            
            successful_results = [r for r in self.results if 'error' not in r]
            
            if successful_results:
                # Best validation loss
                best_loss = min(successful_results, key=lambda x: x['evaluation']['val_loss'])
                f.write(f"**Best Validation Loss:** {best_loss['evaluation']['val_loss']:.4f} (Experiment {best_loss['experiment_id']})\n\n")
                
                # Best throughput
                best_throughput = max(successful_results, key=lambda x: x['throughput']['tokens_per_second'])
                f.write(f"**Best Throughput:** {best_throughput['throughput']['tokens_per_second']:.0f} tokens/s (Experiment {best_throughput['experiment_id']})\n\n")
                
                # Best memory efficiency
                best_memory = min(successful_results, key=lambda x: x['memory_stats']['peak_memory_gb'])
                f.write(f"**Best Memory Efficiency:** {best_memory['memory_stats']['peak_memory_gb']:.1f}GB (Experiment {best_memory['experiment_id']})\n\n")
                
        self.log(f"📊 Report generated: {report_file}")

def main():
    """Main experiment pipeline"""
    
    # Check system requirements
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This pipeline is designed for RTX 5090.")
        return
        
    gpu_name = torch.cuda.get_device_name()
    if "5090" not in gpu_name:
        print(f"⚠️ Warning: Detected {gpu_name}, but pipeline is optimized for RTX 5090")
        
    # Create experiment configuration from JSON
    exp_config = ExperimentConfig.from_json()
    
    print(f"📋 Loaded experiment configuration:")
    print(f"   Models to test: {len(exp_config.base_configs)}")
    print(f"   Training time limit: {exp_config.max_training_time_minutes} minutes")
    print(f"   Memory limit: {exp_config.max_memory_gb}GB")
    print(f"   Optimizations: FP4={exp_config.use_fp4_quantization}, GQA={exp_config.use_grouped_query_attention}")
    
    # Run experiments
    runner = ExperimentRunner(exp_config)
    runner.run_all_experiments()
    
    print(f"\n🎉 All experiments completed!")
    print(f"📁 Results saved in: {exp_config.results_dir}")
    print(f"📊 View the report: {exp_config.results_dir}/experiment_report.md")

if __name__ == "__main__":
    main()