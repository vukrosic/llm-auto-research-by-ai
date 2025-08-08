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

def explain_optimization_techniques():
    """Explain the optimization techniques used"""
    st.markdown("## 🔬 Optimization Techniques Explained")
    
    col1, col2 = st.columns(2)
    
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
    
    if results is None:
        st.error("""
        ❌ **No experiment results found!**
        
        Please run the experiments first:
        ```bash
        python run_experiments.py
        ```
        
        Or if you want to run just the experiments:
        ```bash
        python experiment_pipeline.py
        ```
        """)
        st.stop()
    
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
    
    if config:
        st.sidebar.markdown(f"**Training Time Limit:** {config['training']['max_training_time_minutes']} minutes")
        st.sidebar.markdown(f"**Target GPU:** {config['hardware']['target_gpu']}")
        st.sidebar.markdown(f"**Memory Limit:** {config['hardware']['max_memory_gb']}GB")
    
    # Main content tabs
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