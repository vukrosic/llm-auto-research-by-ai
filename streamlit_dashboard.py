#!/usr/bin/env python3
"""
RTX 5090 Optimization Experiments - Interactive Streamlit Dashboard
==================================================================

Interactive visualization and explanation of LLM training optimization experiments.

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="RTX 5090 LLM Optimization Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .optimization-explanation {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_experiment_results(results_file: str = "experiment_results/experiment_results.json") -> Optional[List[Dict]]:
    """Load experiment results with caching"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return [r for r in results if 'error' not in r]  # Filter successful experiments
    except FileNotFoundError:
        return None

@st.cache_data
def load_experiment_config(config_file: str = "experiment_config.json") -> Optional[Dict]:
    """Load experiment configuration"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def create_results_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert results to pandas DataFrame for easier analysis"""
    data = []
    
    for result in results:
        config = result['config']
        opt = result['optimizations']
        
        # Determine optimization type
        if not opt['use_gqa'] and not opt['use_fp4']:
            opt_type = 'Baseline'
            opt_short = 'Base'
        elif opt['use_gqa'] and not opt['use_fp4']:
            opt_type = 'GQA Only'
            opt_short = 'GQA'
        elif not opt['use_gqa'] and opt['use_fp4']:
            opt_type = 'FP4 Only'
            opt_short = 'FP4'
        else:
            opt_type = 'GQA + FP4'
            opt_short = 'Both'
        
        model_name = f"{config['d_model']}d-{config['n_layers']}L"
        
        row = {
            'experiment_id': result['experiment_id'],
            'model_config': model_name,
            'model_size': f"{config['d_model']}d",
            'n_layers': config['n_layers'],
            'n_heads': config['n_heads'],
            'd_model': config['d_model'],
            'd_ff': config['d_ff'],
            'optimization': opt_type,
            'optimization_short': opt_short,
            'use_gqa': opt['use_gqa'],
            'use_fp4': opt['use_fp4'],
            'n_kv_heads': opt.get('n_kv_heads'),
            'total_parameters': result.get('model_stats', {}).get('total_parameters', 0),
            'optimal_batch_size': result.get('model_stats', {}).get('optimal_batch_size', 0),
            'peak_memory_gb': result.get('memory_stats', {}).get('peak_memory_gb', 0),
            'memory_utilization': result.get('memory_stats', {}).get('memory_utilization', 0),
            'training_time_seconds': result.get('training_stats', {}).get('training_time_seconds', 0),
            'steps_completed': result.get('training_stats', {}).get('steps_completed', 0),
            'tokens_processed': result.get('training_stats', {}).get('tokens_processed', 0),
            'val_loss': result.get('evaluation', {}).get('val_loss', 0),
            'val_accuracy': result.get('evaluation', {}).get('val_accuracy', 0),
            'val_perplexity': result.get('evaluation', {}).get('val_perplexity', 0),
            'tokens_per_second': result.get('throughput', {}).get('tokens_per_second', 0),
            'steps_per_second': result.get('throughput', {}).get('steps_per_second', 0),
        }
        data.append(row)
    
    return pd.DataFrame(data)

def create_visual_architecture_diagram():
    """Create visual architecture comparison diagrams"""
    st.markdown("### 🏗️ Architecture Comparison")
    
    # Create attention mechanism visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Standard Multi-Head Attention', 'Grouped Query Attention (GQA)',
                       'FP32 Weight Storage', 'FP4 Quantized Storage'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Standard MHA visualization
    heads = list(range(1, 17))  # 16 heads
    q_heads = [1] * 16  # Query heads
    k_heads = [1] * 16  # Key heads  
    v_heads = [1] * 16  # Value heads
    
    fig.add_trace(go.Scatter(x=heads, y=q_heads, mode='markers', name='Query Heads',
                            marker=dict(size=15, color='blue'), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=heads, y=[2]*16, mode='markers', name='Key Heads',
                            marker=dict(size=15, color='red'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=heads, y=[3]*16, mode='markers', name='Value Heads',
                            marker=dict(size=15, color='green'), showlegend=False), row=1, col=1)
    
    # GQA visualization
    gqa_heads = list(range(1, 17))
    gqa_q = [1] * 16  # 16 query heads
    gqa_kv_x = [1, 5, 9, 13]  # 4 key-value heads
    gqa_kv_y = [2, 2, 2, 2]
    
    fig.add_trace(go.Scatter(x=gqa_heads, y=gqa_q, mode='markers', name='Query Heads (16)',
                            marker=dict(size=15, color='blue'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=gqa_kv_x, y=gqa_kv_y, mode='markers', name='Key Heads (4)',
                            marker=dict(size=20, color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=gqa_kv_x, y=[3]*4, mode='markers', name='Value Heads (4)',
                            marker=dict(size=20, color='green'), showlegend=False), row=1, col=2)
    
    # Memory usage comparison
    precision_types = ['FP32', 'FP16', 'FP8', 'FP4']
    memory_usage = [32, 16, 8, 4]  # bits per parameter
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig.add_trace(go.Bar(x=precision_types, y=memory_usage, name='Memory per Parameter (bits)',
                        marker_color=colors, showlegend=False), row=2, col=1)
    
    # Quantization savings
    model_sizes = ['Small (50M)', 'Medium (150M)', 'Large (500M)', 'XLarge (1B)']
    fp32_memory = [0.8, 2.4, 8.0, 16.0]  # GB
    fp4_memory = [0.2, 0.6, 2.0, 4.0]   # GB
    
    fig.add_trace(go.Bar(x=model_sizes, y=fp32_memory, name='FP32 Memory',
                        marker_color='#ff7f0e', showlegend=False), row=2, col=2)
    fig.add_trace(go.Bar(x=model_sizes, y=fp4_memory, name='FP4 Memory',
                        marker_color='#9467bd', showlegend=False), row=2, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Head Index", row=1, col=1)
    fig.update_xaxes(title_text="Head Index", row=1, col=2)
    fig.update_xaxes(title_text="Precision Type", row=2, col=1)
    fig.update_xaxes(title_text="Model Size", row=2, col=2)
    
    fig.update_yaxes(title_text="Head Type", row=1, col=1)
    fig.update_yaxes(title_text="Head Type", row=1, col=2)
    fig.update_yaxes(title_text="Bits per Parameter", row=2, col=1)
    fig.update_yaxes(title_text="Memory Usage (GB)", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def create_memory_savings_visualization():
    """Create theoretical memory savings visualization"""
    st.markdown("### 💾 Expected Memory Savings")
    
    # Create theoretical data for different model sizes
    model_configs = ['Small\n(512d-8L)', 'Medium\n(768d-12L)', 'Large\n(1024d-16L)', 'XLarge\n(1280d-20L)']
    
    # Theoretical memory usage (GB) for RTX 5090
    baseline_memory = [3.2, 7.8, 15.6, 22.4]
    gqa_memory = [2.4, 5.9, 11.7, 16.8]
    fp4_memory = [1.3, 3.1, 6.2, 8.9]
    combined_memory = [0.8, 2.0, 4.0, 5.7]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Baseline', x=model_configs, y=baseline_memory,
                        marker_color='#ff7f0e', text=[f'{m:.1f}GB' for m in baseline_memory],
                        textposition='auto'))
    fig.add_trace(go.Bar(name='GQA Only', x=model_configs, y=gqa_memory,
                        marker_color='#2ca02c', text=[f'{m:.1f}GB' for m in gqa_memory],
                        textposition='auto'))
    fig.add_trace(go.Bar(name='FP4 Only', x=model_configs, y=fp4_memory,
                        marker_color='#d62728', text=[f'{m:.1f}GB' for m in fp4_memory],
                        textposition='auto'))
    fig.add_trace(go.Bar(name='GQA + FP4', x=model_configs, y=combined_memory,
                        marker_color='#9467bd', text=[f'{m:.1f}GB' for m in combined_memory],
                        textposition='auto'))
    
    # Add RTX 5090 memory limit line
    fig.add_hline(y=24, line_dash="dash", line_color="red", line_width=3,
                  annotation_text="RTX 5090 Memory Limit (24GB)")
    
    fig.update_layout(
        title='Theoretical Memory Usage by Model Size and Optimization',
        xaxis_title='Model Configuration',
        yaxis_title='Memory Usage (GB)',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory savings table
    st.markdown("#### 📊 Memory Savings Breakdown")
    
    savings_data = []
    for i, config in enumerate(['Small', 'Medium', 'Large', 'XLarge']):
        baseline = baseline_memory[i]
        savings_data.append({
            'Model': config,
            'Baseline (GB)': f"{baseline:.1f}",
            'GQA Savings': f"{(baseline - gqa_memory[i])/baseline*100:.1f}%",
            'FP4 Savings': f"{(baseline - fp4_memory[i])/baseline*100:.1f}%",
            'Combined Savings': f"{(baseline - combined_memory[i])/baseline*100:.1f}%",
            'Combined Memory': f"{combined_memory[i]:.1f}GB"
        })
    
    savings_df = pd.DataFrame(savings_data)
    st.dataframe(savings_df, use_container_width=True)

def create_performance_expectations():
    """Create expected performance visualization"""
    st.markdown("### ⚡ Expected Performance Impact")
    
    # Theoretical performance data
    optimizations = ['Baseline', 'GQA Only', 'FP4 Only', 'GQA + FP4']
    
    # Expected relative performance (baseline = 1.0)
    memory_efficiency = [1.0, 0.7, 0.4, 0.3]  # Lower is better
    training_speed = [1.0, 1.15, 1.10, 1.25]  # Higher is better
    model_quality = [1.0, 0.98, 0.97, 0.95]   # Lower is worse (loss increase)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Memory Efficiency', 'Training Speed', 'Model Quality'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Memory efficiency (lower is better)
    fig.add_trace(go.Bar(x=optimizations, y=memory_efficiency, name='Memory Usage',
                        marker_color=colors, text=[f'{m:.1f}x' for m in memory_efficiency],
                        textposition='auto', showlegend=False), row=1, col=1)
    
    # Training speed (higher is better)
    fig.add_trace(go.Bar(x=optimizations, y=training_speed, name='Speed Multiplier',
                        marker_color=colors, text=[f'{s:.2f}x' for s in training_speed],
                        textposition='auto', showlegend=False), row=1, col=2)
    
    # Model quality (closer to 1.0 is better)
    fig.add_trace(go.Bar(x=optimizations, y=model_quality, name='Quality Retention',
                        marker_color=colors, text=[f'{q:.1%}' for q in model_quality],
                        textposition='auto', showlegend=False), row=1, col=3)
    
    fig.update_yaxes(title_text="Relative Memory Usage", row=1, col=1)
    fig.update_yaxes(title_text="Speed Multiplier", row=1, col=2)
    fig.update_yaxes(title_text="Quality Retention", row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def explain_optimization_techniques():
    """Explain the optimization techniques used with visual elements"""
    st.markdown("## 🔬 Optimization Techniques Explained")
    
    # Create tabs for different explanations
    tech_tab1, tech_tab2, tech_tab3 = st.tabs(["🧮 FP4 Quantization", "🎯 Grouped Query Attention", "📊 Visual Comparisons"])
    
    with tech_tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="optimization-explanation">
            <h3>🧮 FP4 Quantization</h3>
            <p><strong>What it does:</strong> Reduces precision of model weights from 32-bit to 4-bit</p>
            <p><strong>How it works:</strong></p>
            <ul>
                <li>Applied selectively to linear layers > 1000 parameters</li>
                <li>Skips embeddings, layer norms, and small tensors</li>
                <li>Uses NormalFloat4 (NF4) format optimized for neural networks</li>
                <li>Maintains training stability through mixed-precision</li>
            </ul>
            <p><strong>Expected Benefits:</strong></p>
            <ul>
                <li>📉 60-75% memory reduction for quantized parameters</li>
                <li>⚡ 10-20% speed improvement</li>
                <li>⚠️ 2-5% quality degradation (perplexity increase)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Precision comparison chart
            precision_data = pd.DataFrame({
                'Precision': ['FP32', 'FP16', 'FP8', 'FP4'],
                'Bits': [32, 16, 8, 4],
                'Memory': [100, 50, 25, 12.5]  # Relative percentage
            })
            
            fig = px.bar(precision_data, x='Precision', y='Memory',
                        title='Memory Usage by Precision',
                        color='Bits', color_continuous_scale='viridis')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tech_tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="optimization-explanation">
            <h3>🎯 Grouped Query Attention (GQA)</h3>
            <p><strong>What it does:</strong> Reduces key-value projection parameters in attention</p>
            <p><strong>How it works:</strong></p>
            <ul>
                <li>Uses fewer key-value heads than query heads (4:1 ratio)</li>
                <li>Query heads: 16, Key-Value heads: 4 (shared across queries)</li>
                <li>Reduces attention computation and memory</li>
                <li>Maintains most of the quality of full multi-head attention</li>
            </ul>
            <p><strong>Expected Benefits:</strong></p>
            <ul>
                <li>📉 25-40% memory reduction</li>
                <li>⚡ 15-25% speed improvement</li>
                <li>⚠️ 1-3% quality degradation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Attention heads comparison
            attention_data = pd.DataFrame({
                'Type': ['Standard MHA', 'GQA'],
                'Query Heads': [16, 16],
                'Key Heads': [16, 4],
                'Value Heads': [16, 4],
                'Total KV Params': [32, 8]
            })
            
            fig = px.bar(attention_data, x='Type', y=['Query Heads', 'Key Heads', 'Value Heads'],
                        title='Attention Heads Comparison',
                        barmode='group')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tech_tab3:
        create_visual_architecture_diagram()
        create_memory_savings_visualization()
        create_performance_expectations()
    
    st.markdown("""
    <div class="success-box">
    <h3>🚀 Combined Optimizations</h3>
    <p>When used together, FP4 + GQA can achieve:</p>
    <ul>
        <li><strong>70-80% memory reduction</strong> - Fit much larger models in 24GB</li>
        <li><strong>20-35% speed improvement</strong> - Faster training throughput</li>
        <li><strong>3-7% quality impact</strong> - Minimal degradation for significant gains</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def create_memory_comparison_chart(df: pd.DataFrame):
    """Create interactive memory usage comparison"""
    st.markdown("## 📊 Memory Usage Analysis")
    
    # Memory usage by configuration and optimization
    fig = px.bar(
        df, 
        x='model_config', 
        y='peak_memory_gb',
        color='optimization',
        title='Peak Memory Usage by Model Configuration and Optimization',
        labels={
            'peak_memory_gb': 'Peak Memory Usage (GB)',
            'model_config': 'Model Configuration',
            'optimization': 'Optimization Type'
        },
        color_discrete_map={
            'Baseline': '#ff7f0e',
            'GQA Only': '#2ca02c', 
            'FP4 Only': '#d62728',
            'GQA + FP4': '#9467bd'
        }
    )
    
    # Add RTX 5090 memory limit line
    fig.add_hline(y=24, line_dash="dash", line_color="red", 
                  annotation_text="RTX 5090 Memory Limit (24GB)")
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory savings calculation
    st.markdown("### 💾 Memory Savings Analysis")
    
    # Calculate savings relative to baseline
    baseline_memory = df[df['optimization'] == 'Baseline'].set_index('model_config')['peak_memory_gb']
    
    savings_data = []
    for config in df['model_config'].unique():
        config_df = df[df['model_config'] == config]
        baseline_mem = baseline_memory.get(config, 0)
        
        if baseline_mem > 0:
            for _, row in config_df.iterrows():
                if row['optimization'] != 'Baseline':
                    savings_pct = (baseline_mem - row['peak_memory_gb']) / baseline_mem * 100
                    savings_data.append({
                        'model_config': config,
                        'optimization': row['optimization'],
                        'memory_savings_pct': savings_pct,
                        'memory_saved_gb': baseline_mem - row['peak_memory_gb']
                    })
    
    if savings_data:
        savings_df = pd.DataFrame(savings_data)
        
        fig_savings = px.bar(
            savings_df,
            x='model_config',
            y='memory_savings_pct',
            color='optimization',
            title='Memory Savings vs Baseline (%)',
            labels={
                'memory_savings_pct': 'Memory Savings (%)',
                'model_config': 'Model Configuration'
            }
        )
        fig_savings.update_layout(height=400)
        st.plotly_chart(fig_savings, use_container_width=True)

def create_performance_analysis(df: pd.DataFrame):
    """Create performance analysis charts"""
    st.markdown("## ⚡ Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Throughput comparison
        fig_throughput = px.bar(
            df,
            x='model_config',
            y='tokens_per_second',
            color='optimization',
            title='Training Throughput (Tokens/Second)',
            labels={
                'tokens_per_second': 'Tokens per Second',
                'model_config': 'Model Configuration'
            }
        )
        fig_throughput.update_layout(height=400)
        st.plotly_chart(fig_throughput, use_container_width=True)
    
    with col2:
        # Validation loss comparison
        fig_loss = px.bar(
            df,
            x='model_config',
            y='val_loss',
            color='optimization',
            title='Model Quality (Validation Loss)',
            labels={
                'val_loss': 'Validation Loss (lower is better)',
                'model_config': 'Model Configuration'
            }
        )
        fig_loss.update_layout(height=400)
        st.plotly_chart(fig_loss, use_container_width=True)

def create_efficiency_scatter(df: pd.DataFrame):
    """Create efficiency scatter plot"""
    st.markdown("## 🎯 Efficiency Analysis: Memory vs Speed")
    
    fig = px.scatter(
        df,
        x='peak_memory_gb',
        y='tokens_per_second',
        color='optimization',
        size='total_parameters',
        hover_data=['model_config', 'val_loss', 'optimal_batch_size'],
        title='Memory Efficiency vs Training Speed (Bubble size = Parameters)',
        labels={
            'peak_memory_gb': 'Peak Memory Usage (GB)',
            'tokens_per_second': 'Training Throughput (tokens/second)',
            'total_parameters': 'Total Parameters'
        }
    )
    
    # Add RTX 5090 memory limit line
    fig.add_vline(x=24, line_dash="dash", line_color="red", 
                  annotation_text="RTX 5090 Limit")
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **💡 How to read this chart:**
    - **X-axis**: Memory usage (lower is better for fitting larger models)
    - **Y-axis**: Training speed (higher is better for faster experiments)
    - **Bubble size**: Model parameters (larger = more complex model)
    - **Colors**: Different optimization strategies
    - **Goal**: Find points in the top-left (fast + memory efficient)
    """)

def create_detailed_comparison_table(df: pd.DataFrame):
    """Create detailed comparison table"""
    st.markdown("## 📋 Detailed Results Table")
    
    # Select columns for display
    display_columns = [
        'model_config', 'optimization', 'total_parameters', 'optimal_batch_size',
        'peak_memory_gb', 'tokens_per_second', 'val_loss', 'val_perplexity',
        'training_time_seconds', 'steps_completed'
    ]
    
    # Format the dataframe for display
    display_df = df[display_columns].copy()
    display_df['total_parameters'] = display_df['total_parameters'].apply(lambda x: f"{x:,}")
    display_df['peak_memory_gb'] = display_df['peak_memory_gb'].round(2)
    display_df['tokens_per_second'] = display_df['tokens_per_second'].round(0)
    display_df['val_loss'] = display_df['val_loss'].round(4)
    display_df['val_perplexity'] = display_df['val_perplexity'].round(2)
    display_df['training_time_seconds'] = display_df['training_time_seconds'].round(0)
    
    # Rename columns for better display
    display_df.columns = [
        'Model Config', 'Optimization', 'Parameters', 'Batch Size',
        'Memory (GB)', 'Tokens/s', 'Val Loss', 'Perplexity',
        'Time (s)', 'Steps'
    ]
    
    st.dataframe(display_df, use_container_width=True)

def show_experiment_insights(df: pd.DataFrame):
    """Show key insights from experiments"""
    st.markdown("## 🔍 Key Insights")
    
    if len(df) == 0:
        st.warning("No experiment data available for analysis.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Best memory efficiency
        best_memory = df.loc[df['peak_memory_gb'].idxmin()]
        st.markdown(f"""
        <div class="metric-card">
        <h4>🏆 Most Memory Efficient</h4>
        <p><strong>{best_memory['model_config']}</strong> with <strong>{best_memory['optimization']}</strong></p>
        <p>Memory: {best_memory['peak_memory_gb']:.1f}GB</p>
        <p>Throughput: {best_memory['tokens_per_second']:.0f} tokens/s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Best throughput
        best_speed = df.loc[df['tokens_per_second'].idxmax()]
        st.markdown(f"""
        <div class="metric-card">
        <h4>⚡ Fastest Training</h4>
        <p><strong>{best_speed['model_config']}</strong> with <strong>{best_speed['optimization']}</strong></p>
        <p>Speed: {best_speed['tokens_per_second']:.0f} tokens/s</p>
        <p>Memory: {best_speed['peak_memory_gb']:.1f}GB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Best quality
        best_quality = df.loc[df['val_loss'].idxmin()]
        st.markdown(f"""
        <div class="metric-card">
        <h4>🎯 Best Quality</h4>
        <p><strong>{best_quality['model_config']}</strong> with <strong>{best_quality['optimization']}</strong></p>
        <p>Val Loss: {best_quality['val_loss']:.4f}</p>
        <p>Perplexity: {best_quality['val_perplexity']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall insights
    st.markdown("### 📈 Overall Findings")
    
    # Calculate average improvements
    baseline_df = df[df['optimization'] == 'Baseline']
    optimized_df = df[df['optimization'] != 'Baseline']
    
    if len(baseline_df) > 0 and len(optimized_df) > 0:
        avg_memory_reduction = (baseline_df['peak_memory_gb'].mean() - optimized_df['peak_memory_gb'].mean()) / baseline_df['peak_memory_gb'].mean() * 100
        avg_speed_improvement = (optimized_df['tokens_per_second'].mean() - baseline_df['tokens_per_second'].mean()) / baseline_df['tokens_per_second'].mean() * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Memory Reduction", f"{avg_memory_reduction:.1f}%")
        
        with col2:
            st.metric("Average Speed Improvement", f"{avg_speed_improvement:.1f}%")
        
        with col3:
            total_experiments = len(df)
            st.metric("Total Experiments", total_experiments)

def show_experiment_design_explanation(config: Optional[Dict]):
    """Show comprehensive experiment design explanation with visuals"""
    st.markdown("## 🧪 Experiment Design Overview")
    
    # Experiment matrix visualization
    st.markdown("### 📊 Experiment Matrix")
    
    if config:
        model_configs = config['model_configs']
        model_names = [f"{cfg['name']}\n({cfg['d_model']}d-{cfg['n_layers']}L)" for cfg in model_configs]
    else:
        model_names = ['Small\n(512d-8L)', 'Medium\n(768d-12L)', 'Large\n(1024d-16L)', 'XLarge\n(1280d-20L)']
    
    optimizations = ['Baseline', 'GQA Only', 'FP4 Only', 'GQA + FP4']
    
    # Create experiment matrix heatmap
    matrix_data = []
    for i, model in enumerate(model_names):
        for j, opt in enumerate(optimizations):
            matrix_data.append({
                'Model': model,
                'Optimization': opt,
                'Experiment_ID': f"Exp_{i+1}_{j+1}",
                'Value': 1  # All experiments planned
            })
    
    matrix_df = pd.DataFrame(matrix_data)
    matrix_pivot = matrix_df.pivot(index='Model', columns='Optimization', values='Value')
    
    fig = px.imshow(matrix_pivot, 
                    title=f'Experiment Matrix: {len(model_names)} Models × {len(optimizations)} Optimizations = {len(model_names) * len(optimizations)} Total Experiments',
                    color_continuous_scale='Blues',
                    aspect='auto')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model configurations table
    st.markdown("### 🏗️ Model Configurations")
    
    if config:
        config_data = []
        for cfg in config['model_configs']:
            params = estimate_parameters(cfg['d_model'], cfg['n_layers'], cfg['n_heads'], cfg['d_ff'])
            config_data.append({
                'Name': cfg['name'],
                'Dimensions': cfg['d_model'],
                'Layers': cfg['n_layers'],
                'Heads': cfg['n_heads'],
                'Feed Forward': cfg['d_ff'],
                'Est. Parameters': f"{params/1e6:.1f}M",
                'Batch Size': cfg['batch_size'],
                'Use Case': get_use_case_description(cfg['name'])
            })
    else:
        config_data = [
            {'Name': 'Small', 'Dimensions': 512, 'Layers': 8, 'Heads': 8, 'Feed Forward': 2048, 'Est. Parameters': '50M', 'Batch Size': 32, 'Use Case': 'Baseline efficiency'},
            {'Name': 'Medium', 'Dimensions': 768, 'Layers': 12, 'Heads': 12, 'Feed Forward': 3072, 'Est. Parameters': '150M', 'Batch Size': 24, 'Use Case': 'Balanced performance'},
            {'Name': 'Large', 'Dimensions': 1024, 'Layers': 16, 'Heads': 16, 'Feed Forward': 4096, 'Est. Parameters': '500M', 'Batch Size': 16, 'Use Case': 'High capacity'},
            {'Name': 'XLarge', 'Dimensions': 1280, 'Layers': 20, 'Heads': 20, 'Feed Forward': 5120, 'Est. Parameters': '1B', 'Batch Size': 12, 'Use Case': 'Maximum scale'}
        ]
    
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)
    
    # Expected results visualization
    st.markdown("### 📈 Expected Results")
    
    create_expected_results_charts()
    
    # Methodology explanation
    st.markdown("### 🔬 Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="optimization-explanation">
        <h4>🎯 Experimental Controls</h4>
        <ul>
            <li><strong>Fixed Training Time:</strong> 30 minutes per experiment</li>
            <li><strong>Same Dataset:</strong> SmolLM corpus (2000 documents)</li>
            <li><strong>Consistent Hardware:</strong> RTX 5090 (24GB VRAM)</li>
            <li><strong>Memory Safety:</strong> 85% utilization limit</li>
            <li><strong>Automatic Batch Sizing:</strong> Optimal batch size per config</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="optimization-explanation">
        <h4>📊 Metrics Collected</h4>
        <ul>
            <li><strong>Memory:</strong> Peak GPU usage, utilization %</li>
            <li><strong>Performance:</strong> Tokens/second, steps/second</li>
            <li><strong>Quality:</strong> Validation loss, perplexity, accuracy</li>
            <li><strong>Efficiency:</strong> Parameter count, quantization ratio</li>
            <li><strong>Training:</strong> Steps completed, convergence rate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Research background
    st.markdown("### 📚 Research Background")
    
    st.markdown("""
    <div class="success-box">
    <h4>🔬 Scientific Foundation</h4>
    <p>These experiments are based on cutting-edge research:</p>
    <ul>
        <li><strong>FP4 Quantization:</strong> "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"</li>
        <li><strong>Grouped Query Attention:</strong> "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"</li>
        <li><strong>Memory Optimization:</strong> "Training Deep Nets with Sublinear Memory Cost" (Gradient Checkpointing)</li>
        <li><strong>Mixed Precision:</strong> NVIDIA's Automatic Mixed Precision (AMP) techniques</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def estimate_parameters(d_model: int, n_layers: int, n_heads: int, d_ff: int) -> int:
    """Estimate total parameters for a transformer model"""
    # Embedding layer
    vocab_size = 50000  # Approximate for SmolLM tokenizer
    embed_params = vocab_size * d_model
    
    # Each transformer layer
    # Attention: 4 * d_model * d_model (Q, K, V, O projections)
    # Feed forward: 2 * d_model * d_ff (up and down projections)
    # Layer norms: 2 * d_model (attention and FF layer norms)
    layer_params = (4 * d_model * d_model) + (2 * d_model * d_ff) + (2 * d_model)
    
    # Output layer
    output_params = d_model * vocab_size
    
    total = embed_params + (n_layers * layer_params) + output_params
    return total

def get_use_case_description(name: str) -> str:
    """Get use case description for model configuration"""
    use_cases = {
        'Small': 'Baseline efficiency testing',
        'Medium': 'Balanced performance evaluation', 
        'Large': 'High capacity experiments',
        'XLarge': 'Maximum scale optimization'
    }
    return use_cases.get(name, 'General purpose')

def create_expected_results_charts():
    """Create charts showing expected experimental results"""
    
    # Expected memory usage across configurations
    st.markdown("#### 💾 Expected Memory Usage")
    
    models = ['Small', 'Medium', 'Large', 'XLarge']
    baseline_mem = [3.2, 7.8, 15.6, 22.4]
    gqa_mem = [2.4, 5.9, 11.7, 16.8]
    fp4_mem = [1.3, 3.1, 6.2, 8.9]
    combined_mem = [0.8, 2.0, 4.0, 5.7]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=models, y=baseline_mem, mode='lines+markers', name='Baseline', line=dict(color='#ff7f0e', width=3)))
    fig.add_trace(go.Scatter(x=models, y=gqa_mem, mode='lines+markers', name='GQA Only', line=dict(color='#2ca02c', width=3)))
    fig.add_trace(go.Scatter(x=models, y=fp4_mem, mode='lines+markers', name='FP4 Only', line=dict(color='#d62728', width=3)))
    fig.add_trace(go.Scatter(x=models, y=combined_mem, mode='lines+markers', name='GQA + FP4', line=dict(color='#9467bd', width=3)))
    
    fig.add_hline(y=24, line_dash="dash", line_color="red", line_width=2, annotation_text="RTX 5090 Limit (24GB)")
    
    fig.update_layout(
        title='Expected Memory Usage by Model Size and Optimization',
        xaxis_title='Model Configuration',
        yaxis_title='Memory Usage (GB)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Expected performance trade-offs
    st.markdown("#### ⚡ Expected Performance Trade-offs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Speed improvement expectations
        optimizations = ['Baseline', 'GQA Only', 'FP4 Only', 'GQA + FP4']
        speed_improvement = [0, 15, 10, 25]  # Percentage improvement
        
        fig_speed = px.bar(x=optimizations, y=speed_improvement,
                          title='Expected Speed Improvement (%)',
                          color=speed_improvement,
                          color_continuous_scale='viridis')
        fig_speed.update_layout(height=300)
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with col2:
        # Quality impact expectations
        quality_impact = [0, -1, -3, -5]  # Percentage degradation
        
        fig_quality = px.bar(x=optimizations, y=quality_impact,
                            title='Expected Quality Impact (%)',
                            color=quality_impact,
                            color_continuous_scale='RdYlBu_r')
        fig_quality.update_layout(height=300)
        st.plotly_chart(fig_quality, use_container_width=True)

def show_recommendations(df: pd.DataFrame):
    """Show recommendations based on results"""
    st.markdown("## 💡 Recommendations")
    
    if len(df) == 0:
        st.warning("No data available for recommendations.")
        return
    
    # Find best overall configuration (balance of memory, speed, quality)
    df_copy = df.copy()
    
    # Normalize metrics (0-1 scale)
    df_copy['memory_score'] = 1 - (df_copy['peak_memory_gb'] - df_copy['peak_memory_gb'].min()) / (df_copy['peak_memory_gb'].max() - df_copy['peak_memory_gb'].min())
    df_copy['speed_score'] = (df_copy['tokens_per_second'] - df_copy['tokens_per_second'].min()) / (df_copy['tokens_per_second'].max() - df_copy['tokens_per_second'].min())
    df_copy['quality_score'] = 1 - (df_copy['val_loss'] - df_copy['val_loss'].min()) / (df_copy['val_loss'].max() - df_copy['val_loss'].min())
    
    # Combined score (equal weights)
    df_copy['combined_score'] = (df_copy['memory_score'] + df_copy['speed_score'] + df_copy['quality_score']) / 3
    
    best_overall = df_copy.loc[df_copy['combined_score'].idxmax()]
    
    st.markdown(f"""
    <div class="success-box">
    <h3>🏆 Recommended Configuration</h3>
    <p><strong>Model:</strong> {best_overall['model_config']}</p>
    <p><strong>Optimization:</strong> {best_overall['optimization']}</p>
    <p><strong>Why this is recommended:</strong></p>
    <ul>
        <li>Memory Usage: {best_overall['peak_memory_gb']:.1f}GB ({best_overall['peak_memory_gb']/24*100:.1f}% of RTX 5090)</li>
        <li>Training Speed: {best_overall['tokens_per_second']:.0f} tokens/second</li>
        <li>Model Quality: {best_overall['val_loss']:.4f} validation loss</li>
        <li>Parameters: {best_overall['total_parameters']:,}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Specific use case recommendations
    st.markdown("### 🎯 Use Case Specific Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🚀 For Maximum Performance:**
        - Use GQA + FP4 combined optimizations
        - Choose largest model that fits in memory
        - Expect 20-35% speed improvement
        - Accept 3-7% quality degradation
        """)
    
    with col2:
        st.markdown("""
        **🎯 For Best Quality:**
        - Use baseline or GQA-only optimization
        - Choose model size based on quality requirements
        - Minimal quality degradation
        - May require more memory
        """)

def show_getting_started_guide():
    """Show getting started guide for running experiments"""
    st.markdown("## 🚀 Getting Started")
    
    st.markdown("""
    <div class="success-box">
    <h3>🎯 Quick Start</h3>
    <p>Ready to run the optimization experiments? Follow these steps:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step-by-step guide
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### 📋 Prerequisites
        
        **Hardware:**
        - RTX 5090 (24GB VRAM)
        - 32GB+ system RAM
        - CUDA 12.1+ support
        
        **Software:**
        - Python 3.8+
        - PyTorch with CUDA
        - Required packages
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Run Experiments
        
        **Option 1: Full Pipeline (Recommended)**
        ```bash
        python run_experiments.py
        ```
        
        **Option 2: Step by Step**
        ```bash
        # Check system compatibility
        python check_system.py
        
        # Run experiments only
        python experiment_pipeline.py
        
        # Launch dashboard
        streamlit run streamlit_dashboard.py
        ```
        """)
    
    # Installation guide
    st.markdown("### 📦 Installation")
    
    with st.expander("🔧 Detailed Installation Instructions"):
        st.markdown("""
        **1. Install PyTorch with CUDA support:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
        
        **2. Install required packages:**
        ```bash
        pip install transformers datasets tqdm numpy bitsandbytes streamlit plotly pandas
        ```
        
        **3. Verify installation:**
        ```bash
        python check_system.py
        ```
        
        **4. Run experiments:**
        ```bash
        python run_experiments.py
        ```
        """)
    
    # Expected timeline
    st.markdown("### ⏱️ What to Expect")
    
    timeline_data = pd.DataFrame({
        'Phase': ['System Check', 'Small Models', 'Medium Models', 'Large Models', 'XLarge Models', 'Report Generation'],
        'Duration (min)': [1, 30, 30, 30, 30, 5],
        'Description': [
            'Verify GPU, CUDA, and dependencies',
            '4 experiments with 512d-8L models',
            '4 experiments with 768d-12L models', 
            '4 experiments with 1024d-16L models',
            '4 experiments with 1280d-20L models',
            'Generate visualizations and reports'
        ]
    })
    
    fig = px.timeline(timeline_data, x_start=[0, 1, 31, 61, 91, 121], 
                     x_end=[1, 31, 61, 91, 121, 126],
                     y='Phase', color='Duration (min)',
                     title='Experiment Timeline (Total: ~2 hours)')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Troubleshooting
    st.markdown("### 🔧 Troubleshooting")
    
    with st.expander("❓ Common Issues and Solutions"):
        st.markdown("""
        **GPU Memory Issues:**
        - Ensure no other processes are using GPU memory
        - Close other applications that might use VRAM
        - The pipeline automatically adjusts batch sizes
        
        **CUDA Compatibility:**
        - Verify CUDA version: `nvidia-smi`
        - Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
        - Reinstall PyTorch with correct CUDA version
        
        **Package Dependencies:**
        - Update pip: `pip install --upgrade pip`
        - Install in virtual environment to avoid conflicts
        - Check bitsandbytes compatibility with your system
        
        **Slow Performance:**
        - Ensure GPU is being used (check nvidia-smi during training)
        - Verify mixed precision is enabled
        - Check system temperature and throttling
        """)
    
    # Results preview
    st.markdown("### 📊 What You'll Get")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Interactive Dashboard**
        - Memory usage analysis
        - Performance comparisons
        - Efficiency scatter plots
        - Detailed results tables
        """)
    
    with col2:
        st.markdown("""
        **📋 Comprehensive Reports**
        - Markdown summary report
        - JSON data for further analysis
        - PNG visualization plots
        - Training logs and metrics
        """)
    
    with col3:
        st.markdown("""
        **💡 Actionable Insights**
        - Best configuration recommendations
        - Memory vs speed trade-offs
        - Quality impact analysis
        - Use case specific guidance
        """)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 RTX 5090 LLM Optimization Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard visualizes and explains the results of comprehensive LLM training optimization experiments 
    designed specifically for RTX 5090 GPUs. Explore how FP4 quantization and Grouped Query Attention (GQA) 
    affect memory usage, training speed, and model quality.
    """)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Load data
    results = load_experiment_results()
    config = load_experiment_config()
    
    # Show experiment explanation even without results
    if results is None:
        st.warning("""
        ⚠️ **No experiment results found yet!**
        
        This dashboard explains the optimization experiments even before running them.
        To run the experiments:
        ```bash
        python run_experiments.py
        ```
        """)
        
        # Show experiment design and expected results
        show_experiment_design_explanation(config)
        return
    
    if results is not None:
        df = create_results_dataframe(results)
        
        # Sidebar filters
        st.sidebar.markdown("### 🔍 Filters")
        
        selected_configs = st.sidebar.multiselect(
            "Model Configurations",
            options=df['model_config'].unique(),
            default=df['model_config'].unique()
        )
        
        selected_optimizations = st.sidebar.multiselect(
            "Optimization Types",
            options=df['optimization'].unique(),
            default=df['optimization'].unique()
        )
        
        # Filter dataframe
        filtered_df = df[
            (df['model_config'].isin(selected_configs)) & 
            (df['optimization'].isin(selected_optimizations))
        ]
        
        # Sidebar info
        st.sidebar.markdown("### 📊 Experiment Info")
        st.sidebar.metric("Total Experiments", len(df))
        st.sidebar.metric("Filtered Results", len(filtered_df))
    else:
        # No results available
        df = pd.DataFrame()
        filtered_df = pd.DataFrame()
        
        st.sidebar.markdown("### 🧪 Experiment Status")
        st.sidebar.warning("No experiments run yet")
        st.sidebar.markdown("Run experiments to see filters and results")
    
    # Show config info if available
    if config:
        st.sidebar.markdown("### ⚙️ Configuration")
        st.sidebar.markdown(f"**Training Time:** {config['training']['max_training_time_minutes']} min")
        st.sidebar.markdown(f"**Target GPU:** {config['hardware']['target_gpu']}")
        st.sidebar.markdown(f"**Memory Limit:** {config['hardware']['max_memory_gb']}GB")
        st.sidebar.markdown(f"**Model Configs:** {len(config['model_configs'])}")
        st.sidebar.markdown(f"**Total Experiments:** {len(config['model_configs']) * 4}")
    
    # Quick action buttons
    st.sidebar.markdown("### 🚀 Quick Actions")
    if st.sidebar.button("🔄 Refresh Results"):
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button("📖 View README"):
        st.sidebar.markdown("[📖 Open README.md](README.md)")
    
    if results is None:
        st.sidebar.markdown("""
        **To run experiments:**
        ```bash
        python run_experiments.py
        ```
        """)
    else:
        st.sidebar.success(f"✅ {len(df)} experiments loaded")
    
    # Main content tabs
    if results is not None:
        # Show full dashboard with results
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📚 Overview", "📊 Memory Analysis", "⚡ Performance", 
            "🎯 Efficiency", "📋 Detailed Results", "💡 Insights"
        ])
        
        with tab1:
            explain_optimization_techniques()
            
            if len(filtered_df) > 0:
                st.markdown("## 📈 Quick Stats")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Experiments", len(filtered_df))
                with col2:
                    avg_memory = filtered_df['peak_memory_gb'].mean()
                    st.metric("Avg Memory", f"{avg_memory:.1f}GB")
                with col3:
                    avg_speed = filtered_df['tokens_per_second'].mean()
                    st.metric("Avg Speed", f"{avg_speed:.0f} tok/s")
                with col4:
                    avg_loss = filtered_df['val_loss'].mean()
                    st.metric("Avg Val Loss", f"{avg_loss:.4f}")
        
        with tab2:
            if len(filtered_df) > 0:
                create_memory_comparison_chart(filtered_df)
            else:
                st.warning("No data available with current filters.")
        
        with tab3:
            if len(filtered_df) > 0:
                create_performance_analysis(filtered_df)
            else:
                st.warning("No data available with current filters.")
        
        with tab4:
            if len(filtered_df) > 0:
                create_efficiency_scatter(filtered_df)
            else:
                st.warning("No data available with current filters.")
        
        with tab5:
            if len(filtered_df) > 0:
                create_detailed_comparison_table(filtered_df)
            else:
                st.warning("No data available with current filters.")
        
        with tab6:
            if len(filtered_df) > 0:
                show_experiment_insights(filtered_df)
                show_recommendations(filtered_df)
            else:
                st.warning("No data available with current filters.")
    
    else:
        # Show explanation-only dashboard without results
        tab1, tab2, tab3 = st.tabs([
            "📚 Optimization Guide", "🧪 Experiment Design", "🚀 Getting Started"
        ])
        
        with tab1:
            explain_optimization_techniques()
        
        with tab2:
            show_experiment_design_explanation(config)
        
        with tab3:
            show_getting_started_guide()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🚀 RTX 5090 LLM Optimization Dashboard | Built with Streamlit</p>
    <p>For more information, see the <a href="README.md">README</a> or run <code>python check_system.py</code></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()