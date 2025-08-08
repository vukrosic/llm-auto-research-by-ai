# RTX 5090 LLM Training Optimization Pipeline

An advanced language model training pipeline optimized for RTX 5090, featuring automated experiments with FP4 quantization, Grouped Query Attention (GQA), and memory optimization techniques.

## 🚀 Quick Start

### Option 1: Automated Pipeline (Recommended)
```bash
python run_experiments.py
```

This will automatically:
1. ✅ Check system compatibility
2. 🧪 Run all optimization experiments  
3. 📊 Generate comprehensive reports

### Option 2: Manual Steps
```bash
# Check system first
python check_system.py

# Run experiments
python experiment_pipeline.py

# Or run original baseline model
python llm.py
```

The pipeline will automatically:
- Test multiple model configurations
- Apply FP4 quantization to suitable parameters
- Implement Grouped Query Attention for memory efficiency
- Maximize GPU memory utilization
- Generate comprehensive performance reports

## 🧪 Experiment Design

### Optimization Techniques Tested

1. **FP4 Quantization**
   - Applied selectively to linear layer weights
   - Skips embeddings, layer norms, and small tensors
   - Uses bitsandbytes for efficient 4-bit computation
   - Reduces memory usage by ~75% for quantized parameters

2. **Grouped Query Attention (GQA)**
   - Reduces key-value projection parameters
   - Uses 1/4 ratio (n_kv_heads = n_heads // 4)
   - Maintains quality while reducing memory and computation
   - Particularly effective for larger models

3. **Memory Optimization**
   - Gradient checkpointing for activation memory
   - Automatic batch size optimization
   - Mixed precision training (FP16)
   - Memory monitoring and safety margins

### Model Configurations Tested

| Config | d_model | n_layers | n_heads | d_ff | Target Use Case |
|--------|---------|----------|---------|------|----------------|
| Small  | 512     | 8        | 8       | 2048 | Baseline efficiency |
| Medium | 768     | 12       | 12      | 3072 | Balanced performance |
| Large  | 1024    | 16       | 16      | 4096 | High capacity |
| XLarge | 1280    | 20       | 20      | 5120 | Maximum scale |

### Experiment Matrix

For each model configuration, we test:
- ✅ Baseline (no optimizations)
- ✅ GQA only
- ✅ FP4 only  
- ✅ GQA + FP4 combined

**Total Experiments**: 16 (4 configs × 4 optimization combinations)

## 📊 Performance Metrics

Each experiment measures:
- **Training Loss & Validation Metrics**
- **Memory Usage** (peak GPU memory)
- **Training Throughput** (tokens/second)
- **Parameter Count** (total and quantized)
- **Training Time** (fixed 30-minute limit for fair comparison)

## 🔧 Technical Implementation

### FP4 Quantization Strategy

```python
# Applied to:
✅ Linear layer weights (> 1000 parameters)
✅ Attention projections (Q, K, V, O)
✅ Feed-forward layers

# Skipped for:
❌ Embeddings (quality sensitive)
❌ Layer norms (small parameter count)
❌ Output projection (final layer)
❌ Very small layers (< 1000 params)
```

### Grouped Query Attention

```python
# Standard MHA: n_heads = 16
Q: [batch, seq, 16, d_k]  # 16 query heads
K: [batch, seq, 16, d_k]  # 16 key heads  
V: [batch, seq, 16, d_k]  # 16 value heads

# GQA: n_kv_heads = 4
Q: [batch, seq, 16, d_k]  # 16 query heads
K: [batch, seq, 4, d_k]   # 4 key heads (shared)
V: [batch, seq, 4, d_k]   # 4 value heads (shared)
```

### Memory Optimization

- **Automatic Batch Size**: Finds largest batch that fits in 85% of GPU memory
- **Gradient Checkpointing**: Trades computation for memory in transformer blocks
- **Memory Monitoring**: Real-time tracking of GPU memory usage
- **Safety Margins**: Prevents OOM with conservative memory estimates

## 📈 Expected Results

Based on research and similar experiments:

| Optimization | Memory Reduction | Speed Impact | Quality Impact |
|-------------|------------------|--------------|----------------|
| FP4 Quantization | 60-75% | +10-20% | -2-5% perplexity |
| GQA (4:1 ratio) | 25-40% | +15-25% | -1-3% perplexity |
| Combined | 70-80% | +20-35% | -3-7% perplexity |

## 🛠 Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm numpy bitsandbytes
```

**Hardware Requirements:**
- RTX 5090 (24GB VRAM) - primary target
- CUDA 12.1+ compatible GPU
- 32GB+ system RAM recommended

## 🌐 Interactive Dashboard

The Streamlit dashboard provides an intuitive way to explore and understand your optimization experiments:

### Dashboard Features

- **📚 Overview Tab**: Explains optimization techniques with visual examples
- **📊 Memory Analysis**: Interactive charts showing memory usage and savings
- **⚡ Performance Tab**: Training speed and model quality comparisons  
- **🎯 Efficiency Tab**: Memory vs speed scatter plots with parameter sizing
- **📋 Detailed Results**: Sortable table with all experiment metrics
- **💡 Insights Tab**: Key findings and personalized recommendations

### Dashboard Screenshots

The dashboard includes:
- **Interactive filters** in the sidebar to focus on specific configurations
- **Hover tooltips** with detailed information on all charts
- **Responsive design** that works on different screen sizes
- **Export capabilities** for charts and data tables
- **Real-time updates** when new experiment results are available

### Visual Explanations

- **Memory Usage Charts**: Bar charts comparing peak memory across optimizations
- **Performance Scatter Plots**: Efficiency analysis with bubble sizing by parameters
- **Savings Analysis**: Percentage improvements over baseline configurations
- **Quality vs Speed Trade-offs**: Interactive exploration of optimization impacts

## 📁 Project Structure

```
├── run_experiments.py          # 🚀 Main launcher (START HERE)
├── streamlit_dashboard.py      # 🌐 Interactive dashboard
├── launch_dashboard.py         # 🎛️ Dashboard launcher
├── experiment_pipeline.py      # 🧪 Core experiment runner
├── llm.py                     # 📚 Original baseline model
├── check_system.py            # 🔍 System compatibility checker
├── visualize_results.py        # 📈 Static plot generator
├── experiment_config.json     # ⚙️ Experiment configuration
├── requirements.txt           # 📦 Python dependencies
└── README.md                  # 📖 This file

# Generated during experiments:
experiment_results/
├── experiment_log.txt          # Detailed training logs
├── experiment_results.json     # Raw experiment data
├── experiment_report.md        # Summary report with tables
└── *.png                      # Generated visualization plots

data_cache/                     # Cached datasets (auto-generated)
```

## 🎯 Usage Examples

### Run Full Pipeline with Interactive Dashboard
```bash
python run_experiments.py
```
This will run all experiments and optionally launch an interactive Streamlit dashboard.

### Launch Interactive Dashboard Only (after experiments)
```bash
streamlit run streamlit_dashboard.py
# or
python launch_dashboard.py
```

### Run Experiments Only
```bash
python experiment_pipeline.py
```

### Run Single Model (Original)
```bash
python llm.py
```

### Custom Experiment Configuration
```python
from experiment_pipeline import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    max_training_time_minutes=60,  # Longer training
    base_configs=[
        {"d_model": 1024, "n_heads": 16, "n_layers": 24, "d_ff": 4096}
    ]
)

runner = ExperimentRunner(config)
runner.run_all_experiments()
```

## 🔬 Research Background

### FP4 Quantization
- Based on "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- Uses NormalFloat4 (NF4) format optimized for neural network weights
- Maintains training stability through mixed-precision techniques

### Grouped Query Attention
- Introduced in "GQA: Training Generalized Multi-Query Transformer Models"
- Reduces KV cache size for inference efficiency
- Maintains most of the quality of full multi-head attention

### Memory Optimization
- Gradient checkpointing from "Training Deep Nets with Sublinear Memory Cost"
- Automatic batch size optimization prevents OOM errors
- Mixed precision training reduces memory by ~50%

## 📊 Baseline Model (llm.py)

The original implementation features:
- **Muon Optimizer**: MomentUm Orthogonalized by Newton-schulz
- **Efficient Architecture**: RoPE, RMSNorm, SiLU activation
- **Mixed Precision**: Automatic mixed precision with gradient scaling
- **Data Caching**: Intelligent dataset caching system

### Original Configuration
- 6 layers, 8 heads, 384 dimensions
- SmolLM tokenizer (~50k vocabulary)
- 512 token context length
- Hybrid Muon + AdamW optimization

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # or manually:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers datasets tqdm numpy bitsandbytes streamlit plotly pandas
   ```

2. **Check System Compatibility**
   ```bash
   python check_system.py
   ```

3. **Run Complete Pipeline**
   ```bash
   python run_experiments.py
   ```
   This will:
   - ✅ Check system compatibility
   - 🧪 Run all optimization experiments
   - 📊 Generate visualizations
   - 🌐 Optionally launch interactive dashboard

4. **Explore Results**
   ```bash
   # View static report
   cat experiment_results/experiment_report.md
   
   # Launch interactive dashboard
   streamlit run streamlit_dashboard.py
   ```

## 🎯 Key Features

- **Automated Optimization**: No manual tuning required
- **Memory Safety**: Prevents OOM with intelligent batch sizing
- **Fair Comparison**: Fixed time limits ensure comparable results
- **Comprehensive Metrics**: Loss, throughput, memory, and quality metrics
- **Research-Based**: Implements proven optimization techniques
- **RTX 5090 Optimized**: Designed specifically for 24GB VRAM

## 📝 License

MIT License - see LICENSE file for details.