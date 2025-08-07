#!/usr/bin/env python3
"""
Results Visualization for RTX 5090 Optimization Experiments
===========================================================

Generates plots and visualizations from experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Any

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_file: str = "experiment_results/experiment_results.json") -> List[Dict]:
    """Load experiment results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return [r for r in results if 'error' not in r]  # Filter out failed experiments
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return []

def create_memory_usage_plot(results: List[Dict], output_dir: Path):
    """Create memory usage comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    configs = []
    memory_baseline = []
    memory_gqa = []
    memory_fp4 = []
    memory_both = []
    
    # Group by model configuration
    config_groups = {}
    for result in results:
        config_key = f"{result['config']['d_model']}d-{result['config']['n_layers']}L"
        if config_key not in config_groups:
            config_groups[config_key] = {}
        
        opt = result['optimizations']
        if not opt['use_gqa'] and not opt['use_fp4']:
            config_groups[config_key]['baseline'] = result
        elif opt['use_gqa'] and not opt['use_fp4']:
            config_groups[config_key]['gqa'] = result
        elif not opt['use_gqa'] and opt['use_fp4']:
            config_groups[config_key]['fp4'] = result
        elif opt['use_gqa'] and opt['use_fp4']:
            config_groups[config_key]['both'] = result
    
    # Prepare data for plotting
    for config_key, group in config_groups.items():
        if 'baseline' in group:  # Only include configs with baseline
            configs.append(config_key)
            memory_baseline.append(group.get('baseline', {}).get('memory_stats', {}).get('peak_memory_gb', 0))
            memory_gqa.append(group.get('gqa', {}).get('memory_stats', {}).get('peak_memory_gb', 0))
            memory_fp4.append(group.get('fp4', {}).get('memory_stats', {}).get('peak_memory_gb', 0))
            memory_both.append(group.get('both', {}).get('memory_stats', {}).get('peak_memory_gb', 0))
    
    # Plot 1: Memory usage comparison
    x = np.arange(len(configs))
    width = 0.2
    
    ax1.bar(x - 1.5*width, memory_baseline, width, label='Baseline', alpha=0.8)
    ax1.bar(x - 0.5*width, memory_gqa, width, label='GQA Only', alpha=0.8)
    ax1.bar(x + 0.5*width, memory_fp4, width, label='FP4 Only', alpha=0.8)
    ax1.bar(x + 1.5*width, memory_both, width, label='GQA + FP4', alpha=0.8)
    
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Peak Memory Usage (GB)')
    ax1.set_title('Memory Usage by Optimization')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add RTX 5090 memory limit line
    ax1.axhline(y=24, color='red', linestyle='--', alpha=0.7, label='RTX 5090 Limit (24GB)')
    
    # Plot 2: Memory savings percentage
    memory_savings_gqa = [(b - g) / b * 100 if b > 0 else 0 for b, g in zip(memory_baseline, memory_gqa)]
    memory_savings_fp4 = [(b - f) / b * 100 if b > 0 else 0 for b, f in zip(memory_baseline, memory_fp4)]
    memory_savings_both = [(b - c) / b * 100 if b > 0 else 0 for b, c in zip(memory_baseline, memory_both)]
    
    ax2.bar(x - width, memory_savings_gqa, width, label='GQA Only', alpha=0.8)
    ax2.bar(x, memory_savings_fp4, width, label='FP4 Only', alpha=0.8)
    ax2.bar(x + width, memory_savings_both, width, label='GQA + FP4', alpha=0.8)
    
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('Memory Savings (%)')
    ax2.set_title('Memory Savings vs Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_plot(results: List[Dict], output_dir: Path):
    """Create performance comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    configs = []
    throughput_baseline = []
    throughput_gqa = []
    throughput_fp4 = []
    throughput_both = []
    
    loss_baseline = []
    loss_gqa = []
    loss_fp4 = []
    loss_both = []
    
    # Group by model configuration
    config_groups = {}
    for result in results:
        config_key = f"{result['config']['d_model']}d-{result['config']['n_layers']}L"
        if config_key not in config_groups:
            config_groups[config_key] = {}
        
        opt = result['optimizations']
        if not opt['use_gqa'] and not opt['use_fp4']:
            config_groups[config_key]['baseline'] = result
        elif opt['use_gqa'] and not opt['use_fp4']:
            config_groups[config_key]['gqa'] = result
        elif not opt['use_gqa'] and opt['use_fp4']:
            config_groups[config_key]['fp4'] = result
        elif opt['use_gqa'] and opt['use_fp4']:
            config_groups[config_key]['both'] = result
    
    # Prepare data
    for config_key, group in config_groups.items():
        if 'baseline' in group:
            configs.append(config_key)
            
            # Throughput data
            throughput_baseline.append(group.get('baseline', {}).get('throughput', {}).get('tokens_per_second', 0))
            throughput_gqa.append(group.get('gqa', {}).get('throughput', {}).get('tokens_per_second', 0))
            throughput_fp4.append(group.get('fp4', {}).get('throughput', {}).get('tokens_per_second', 0))
            throughput_both.append(group.get('both', {}).get('throughput', {}).get('tokens_per_second', 0))
            
            # Loss data
            loss_baseline.append(group.get('baseline', {}).get('evaluation', {}).get('val_loss', 0))
            loss_gqa.append(group.get('gqa', {}).get('evaluation', {}).get('val_loss', 0))
            loss_fp4.append(group.get('fp4', {}).get('evaluation', {}).get('val_loss', 0))
            loss_both.append(group.get('both', {}).get('evaluation', {}).get('val_loss', 0))
    
    # Plot 1: Throughput comparison
    x = np.arange(len(configs))
    width = 0.2
    
    ax1.bar(x - 1.5*width, throughput_baseline, width, label='Baseline', alpha=0.8)
    ax1.bar(x - 0.5*width, throughput_gqa, width, label='GQA Only', alpha=0.8)
    ax1.bar(x + 0.5*width, throughput_fp4, width, label='FP4 Only', alpha=0.8)
    ax1.bar(x + 1.5*width, throughput_both, width, label='GQA + FP4', alpha=0.8)
    
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Throughput (tokens/second)')
    ax1.set_title('Training Throughput by Optimization')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation loss comparison
    ax2.bar(x - 1.5*width, loss_baseline, width, label='Baseline', alpha=0.8)
    ax2.bar(x - 0.5*width, loss_gqa, width, label='GQA Only', alpha=0.8)
    ax2.bar(x + 0.5*width, loss_fp4, width, label='FP4 Only', alpha=0.8)
    ax2.bar(x + 1.5*width, loss_both, width, label='GQA + FP4', alpha=0.8)
    
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Model Quality by Optimization')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_efficiency_scatter(results: List[Dict], output_dir: Path):
    """Create efficiency scatter plot (memory vs throughput)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract data for scatter plot
    memory_usage = []
    throughput = []
    colors = []
    labels = []
    sizes = []
    
    color_map = {
        'baseline': 'red',
        'gqa': 'blue', 
        'fp4': 'green',
        'both': 'purple'
    }
    
    for result in results:
        memory = result.get('memory_stats', {}).get('peak_memory_gb', 0)
        tokens_per_sec = result.get('throughput', {}).get('tokens_per_second', 0)
        params = result.get('model_stats', {}).get('total_parameters', 0)
        
        opt = result['optimizations']
        if not opt['use_gqa'] and not opt['use_fp4']:
            opt_type = 'baseline'
        elif opt['use_gqa'] and not opt['use_fp4']:
            opt_type = 'gqa'
        elif not opt['use_gqa'] and opt['use_fp4']:
            opt_type = 'fp4'
        else:
            opt_type = 'both'
        
        memory_usage.append(memory)
        throughput.append(tokens_per_sec)
        colors.append(color_map[opt_type])
        labels.append(opt_type)
        sizes.append(params / 1e6)  # Size proportional to parameters (in millions)
    
    # Create scatter plot
    for opt_type, color in color_map.items():
        mask = [l == opt_type for l in labels]
        if any(mask):
            x_vals = [x for x, m in zip(memory_usage, mask) if m]
            y_vals = [y for y, m in zip(throughput, mask) if m]
            s_vals = [s for s, m in zip(sizes, mask) if m]
            
            ax.scatter(x_vals, y_vals, c=color, s=s_vals, alpha=0.7, 
                      label=opt_type.replace('_', ' ').title(), edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Peak Memory Usage (GB)')
    ax.set_ylabel('Training Throughput (tokens/second)')
    ax.set_title('Memory Efficiency vs Training Speed\n(Bubble size = Model parameters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add RTX 5090 memory limit line
    ax.axvline(x=24, color='red', linestyle='--', alpha=0.7, label='RTX 5090 Limit (24GB)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(results: List[Dict], output_dir: Path):
    """Generate a summary table as an image"""
    # Prepare data for table
    table_data = []
    
    for result in results:
        config = result['config']
        opt = result['optimizations']
        
        # Optimization type
        if not opt['use_gqa'] and not opt['use_fp4']:
            opt_type = 'Baseline'
        elif opt['use_gqa'] and not opt['use_fp4']:
            opt_type = 'GQA Only'
        elif not opt['use_gqa'] and opt['use_fp4']:
            opt_type = 'FP4 Only'
        else:
            opt_type = 'GQA + FP4'
        
        row = [
            f"{config['d_model']}d-{config['n_layers']}L",
            opt_type,
            f"{result.get('model_stats', {}).get('total_parameters', 0):,}",
            f"{result.get('memory_stats', {}).get('peak_memory_gb', 0):.1f}",
            f"{result.get('throughput', {}).get('tokens_per_second', 0):.0f}",
            f"{result.get('evaluation', {}).get('val_loss', 0):.4f}",
            f"{result.get('training_stats', {}).get('training_time_seconds', 0):.0f}"
        ]
        table_data.append(row)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Config', 'Optimization', 'Parameters', 'Memory (GB)', 'Tokens/s', 'Val Loss', 'Time (s)']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('RTX 5090 Optimization Experiment Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'results_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations"""
    print("📊 Generating visualizations from experiment results...")
    
    # Load results
    results = load_results()
    if not results:
        print("❌ No results found. Run experiments first.")
        return
    
    # Create output directory
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"   Found {len(results)} successful experiments")
    
    try:
        # Generate plots
        print("   Creating memory usage plots...")
        create_memory_usage_plot(results, output_dir)
        
        print("   Creating performance plots...")
        create_performance_plot(results, output_dir)
        
        print("   Creating efficiency scatter plot...")
        create_efficiency_scatter(results, output_dir)
        
        print("   Creating summary table...")
        generate_summary_table(results, output_dir)
        
        print(f"✅ Visualizations saved to {output_dir}/")
        print("   Generated files:")
        for plot_file in output_dir.glob("*.png"):
            print(f"     📈 {plot_file.name}")
            
    except ImportError as e:
        print(f"⚠️ Visualization requires matplotlib and seaborn: {e}")
        print("   Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

if __name__ == "__main__":
    main()