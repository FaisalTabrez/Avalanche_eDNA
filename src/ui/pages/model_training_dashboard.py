"""
Model Training Dashboard
Visualize continual learning progress, model evolution, and performance
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.model_registry import ModelRegistry
from src.models.checkpoint_manager import CheckpointManager


def main():
    st.title("🧬 Model Training Dashboard")
    st.markdown("Monitor continual learning progress and model evolution across datasets")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Select registry directory
    default_registry = Path("consolidated_data/results/model_registry")
    registry_dir = st.sidebar.text_input(
        "Model Registry Directory",
        value=str(default_registry)
    )
    
    default_checkpoint = Path("consolidated_data/results/checkpoints")
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint Directory",
        value=str(default_checkpoint)
    )
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Overview",
        "📈 Performance Trends",
        "🌳 Model Lineage",
        "🔍 Model Comparison",
        "💾 Checkpoints"
    ])
    
    # Initialize registry
    try:
        if not Path(registry_dir).exists():
            st.warning(f"Registry directory not found: {registry_dir}")
            st.info("The registry will be created when you train your first model with continual learning enabled.")
            return
        
        registry = ModelRegistry(registry_dir=registry_dir, backend='json')
        models = registry.list_models()
        
        if not models:
            st.info("No models registered yet. Train a model with `--fine-tune` to get started!")
            return
        
        # Tab 1: Model Overview
        with tab1:
            show_model_overview(registry, models)
        
        # Tab 2: Performance Trends
        with tab2:
            show_performance_trends(registry, models)
        
        # Tab 3: Model Lineage
        with tab3:
            show_model_lineage(registry, models)
        
        # Tab 4: Model Comparison
        with tab4:
            show_model_comparison(registry, models)
        
        # Tab 5: Checkpoints
        with tab5:
            show_checkpoints(checkpoint_dir)
    
    except Exception as e:
        st.error(f"Error loading registry: {e}")
        st.exception(e)


def show_model_overview(registry: ModelRegistry, models: List[Dict]):
    """Show overview of all models"""
    st.header("Model Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(models))
    
    with col2:
        active_models = len([m for m in models if m.get('status') == 'active'])
        st.metric("Active Models", active_models)
    
    with col3:
        all_datasets = set()
        for m in models:
            all_datasets.update(m.get('datasets', []))
        st.metric("Datasets Used", len(all_datasets))
    
    with col4:
        # Find model with best metric if available
        best_model = registry.get_best_model('val_loss', minimize=True)
        if best_model:
            st.metric("Best Val Loss", f"{best_model['metrics'].get('val_loss', 'N/A'):.4f}")
        else:
            st.metric("Best Val Loss", "N/A")
    
    # Models table
    st.subheader("All Models")
    
    # Create DataFrame for display
    model_data = []
    for model in models:
        model_data.append({
            'Version': model['version'],
            'Status': model['status'],
            'Datasets': ', '.join(model.get('datasets', [])),
            'Created': model.get('created_at', 'N/A'),
            'Val Loss': model.get('metrics', {}).get('val_loss', 'N/A'),
            'Val Accuracy': model.get('metrics', {}).get('val_accuracy', 'N/A'),
            'Parent': model.get('parent_version', 'None')
        })
    
    df = pd.DataFrame(model_data)
    st.dataframe(df, use_container_width=True)
    
    # Model details
    st.subheader("Model Details")
    selected_version = st.selectbox(
        "Select model to view details",
        options=[m['version'] for m in models]
    )
    
    if selected_version:
        model_info = registry.get_model(selected_version)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information**")
            st.json({
                'Version': model_info['version'],
                'Status': model_info['status'],
                'Base Model': model_info.get('base_model', 'N/A'),
                'Created': model_info.get('created_at', 'N/A'),
                'Description': model_info.get('description', 'N/A')
            })
        
        with col2:
            st.write("**Metrics**")
            metrics = model_info.get('metrics', {})
            if metrics:
                st.json(metrics)
            else:
                st.info("No metrics available")
        
        # Show datasets
        st.write("**Datasets**")
        datasets = model_info.get('datasets', [])
        if datasets:
            st.write(", ".join(datasets))
        else:
            st.info("No datasets recorded")
        
        # Show configuration
        with st.expander("View Training Configuration"):
            config = model_info.get('config', {})
            if config:
                st.json(config)
            else:
                st.info("No configuration available")


def show_performance_trends(registry: ModelRegistry, models: List[Dict]):
    """Show performance trends over time"""
    st.header("Performance Trends")
    
    # Extract metrics over time
    metric_data = []
    for model in sorted(models, key=lambda x: x.get('created_at', '')):
        metrics = model.get('metrics', {})
        if metrics:
            row = {
                'version': model['version'],
                'created_at': model.get('created_at', ''),
                'datasets': ', '.join(model.get('datasets', [])),
                **metrics
            }
            metric_data.append(row)
    
    if not metric_data:
        st.info("No metrics available to display trends")
        return
    
    df = pd.DataFrame(metric_data)
    
    # Select metrics to plot
    available_metrics = [col for col in df.columns if col not in ['version', 'created_at', 'datasets']]
    
    if not available_metrics:
        st.info("No numeric metrics found")
        return
    
    selected_metrics = st.multiselect(
        "Select metrics to visualize",
        options=available_metrics,
        default=available_metrics[:2] if len(available_metrics) >= 2 else available_metrics
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric")
        return
    
    # Plot trends
    for metric in selected_metrics:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df[metric],
            mode='lines+markers',
            name=metric,
            text=df['version'],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{metric}: %{{y:.4f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{metric} Over Model Versions',
            xaxis_title='Model Version',
            yaxis_title=metric,
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Dataset-wise comparison
    st.subheader("Dataset-wise Performance")
    
    # Group by dataset
    dataset_metrics = {}
    for model in models:
        datasets = model.get('datasets', [])
        metrics = model.get('metrics', {})
        
        for dataset in datasets:
            if dataset not in dataset_metrics:
                dataset_metrics[dataset] = []
            
            dataset_metrics[dataset].append({
                'version': model['version'],
                'metrics': metrics
            })
    
    if dataset_metrics:
        selected_dataset = st.selectbox(
            "Select dataset",
            options=list(dataset_metrics.keys())
        )
        
        if selected_dataset:
            dataset_models = dataset_metrics[selected_dataset]
            
            # Create comparison table
            comparison_data = []
            for item in dataset_models:
                row = {'Version': item['version']}
                row.update(item['metrics'])
                comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)


def show_model_lineage(registry: ModelRegistry, models: List[Dict]):
    """Show model lineage tree"""
    st.header("Model Lineage")
    
    # Select model to trace
    model_versions = [m['version'] for m in models]
    selected_version = st.selectbox(
        "Select model to view lineage",
        options=model_versions
    )
    
    if not selected_version:
        return
    
    # Get lineage
    lineage = registry.get_lineage(selected_version)
    
    st.subheader(f"Ancestry of {selected_version}")
    
    # Display lineage as a tree
    for i, model in enumerate(lineage):
        indent = "  " * i
        status_emoji = "✅" if model['status'] == 'active' else "📦"
        
        st.markdown(f"{indent}{status_emoji} **{model['version']}**")
        st.markdown(f"{indent}   - Created: {model.get('created_at', 'N/A')}")
        st.markdown(f"{indent}   - Datasets: {', '.join(model.get('datasets', []))}")
        
        if i < len(lineage) - 1:
            st.markdown(f"{indent}   ⬇️")
    
    # Show descendants
    st.subheader(f"Descendants of {selected_version}")
    children = registry.get_children(selected_version)
    
    if children:
        for child in children:
            st.markdown(f"- **{child['version']}** (Created: {child.get('created_at', 'N/A')})")
    else:
        st.info("No descendants found")
    
    # Visualize lineage graph
    st.subheader("Lineage Graph")
    
    # Build graph data
    nodes = []
    edges = []
    
    for model in models:
        nodes.append({
            'id': model['version'],
            'label': model['version'],
            'status': model['status']
        })
        
        parent = model.get('parent_version')
        if parent:
            edges.append({
                'source': parent,
                'target': model['version']
            })
    
    if nodes:
        # Create network visualization using plotly
        # For simplicity, show as a hierarchical list
        st.write("Model Evolution:")
        
        # Group by generation
        generations = {}
        for model in models:
            lineage_len = len(registry.get_lineage(model['version']))
            if lineage_len not in generations:
                generations[lineage_len] = []
            generations[lineage_len].append(model['version'])
        
        for gen in sorted(generations.keys()):
            st.markdown(f"**Generation {gen}:** {', '.join(generations[gen])}")


def show_model_comparison(registry: ModelRegistry, models: List[Dict]):
    """Compare two models"""
    st.header("Model Comparison")
    
    model_versions = [m['version'] for m in models]
    
    col1, col2 = st.columns(2)
    
    with col1:
        model1 = st.selectbox(
            "Select first model",
            options=model_versions,
            key="model1"
        )
    
    with col2:
        model2 = st.selectbox(
            "Select second model",
            options=model_versions,
            key="model2"
        )
    
    if model1 and model2 and model1 != model2:
        comparison = registry.compare_models(model1, model2)
        
        # Show comparison summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model 1",
                model1
            )
        
        with col2:
            st.metric(
                "Age Difference (days)",
                f"{comparison.get('created_at_diff_days', 0):.0f}"
            )
        
        with col3:
            st.metric(
                "Model 2",
                model2
            )
        
        # Metric differences
        st.subheader("Metric Differences")
        
        metric_diffs = comparison.get('metric_differences', {})
        if metric_diffs:
            diff_data = []
            for metric, values in metric_diffs.items():
                diff_data.append({
                    'Metric': metric,
                    f'{model1}': values['v1'],
                    f'{model2}': values['v2'],
                    'Change': values['change'],
                    'Change %': f"{values['percent_change']:.2f}%"
                })
            
            diff_df = pd.DataFrame(diff_data)
            st.dataframe(diff_df, use_container_width=True)
            
            # Visualize differences
            fig = go.Figure()
            
            metrics = [d['Metric'] for d in diff_data]
            model1_values = [d[model1] for d in diff_data]
            model2_values = [d[model2] for d in diff_data]
            
            fig.add_trace(go.Bar(
                name=model1,
                x=metrics,
                y=model1_values
            ))
            
            fig.add_trace(go.Bar(
                name=model2,
                x=metrics,
                y=model2_values
            ))
            
            fig.update_layout(
                title="Metric Comparison",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No common metrics to compare")
        
        # Dataset differences
        st.subheader("Dataset Differences")
        
        dataset_diffs = comparison.get('dataset_differences', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Only in {model1}:**")
            only_in_v1 = dataset_diffs.get('only_in_v1', [])
            if only_in_v1:
                for ds in only_in_v1:
                    st.write(f"- {ds}")
            else:
                st.info("None")
        
        with col2:
            st.write("**Common:**")
            common = dataset_diffs.get('common', [])
            if common:
                for ds in common:
                    st.write(f"- {ds}")
            else:
                st.info("None")
        
        with col3:
            st.write(f"**Only in {model2}:**")
            only_in_v2 = dataset_diffs.get('only_in_v2', [])
            if only_in_v2:
                for ds in only_in_v2:
                    st.write(f"- {ds}")
            else:
                st.info("None")


def show_checkpoints(checkpoint_dir: str):
    """Show available checkpoints"""
    st.header("Training Checkpoints")
    
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        st.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        st.info("Checkpoints will be saved here when you train with `--checkpoint-every` flag.")
        return
    
    try:
        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        history = checkpoint_manager.get_checkpoint_history()
        
        if not history:
            st.info("No checkpoints found")
            return
        
        # Display checkpoint history
        st.subheader("Checkpoint History")
        
        checkpoint_data = []
        for ckpt in history:
            checkpoint_data.append({
                'Timestamp': ckpt['timestamp'],
                'Epoch': ckpt['epoch'],
                'Val Loss': ckpt.get('metrics', {}).get('val_loss', 'N/A'),
                'Path': Path(ckpt['path']).name
            })
        
        df = pd.DataFrame(checkpoint_data)
        st.dataframe(df, use_container_width=True)
        
        # Show best checkpoint
        best_checkpoint = checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            st.success(f"**Best Checkpoint:** Epoch {best_checkpoint['epoch']} (Val Loss: {best_checkpoint.get('metrics', {}).get('val_loss', 'N/A')})")
        
        # Show latest checkpoint
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            st.info(f"**Latest Checkpoint:** Epoch {latest_checkpoint['epoch']} ({latest_checkpoint['timestamp']})")
        
        # Checkpoint details
        st.subheader("Checkpoint Details")
        
        selected_ckpt_name = st.selectbox(
            "Select checkpoint",
            options=[Path(ckpt['path']).name for ckpt in history]
        )
        
        if selected_ckpt_name:
            selected_ckpt = next(
                (ckpt for ckpt in history if Path(ckpt['path']).name == selected_ckpt_name),
                None
            )
            
            if selected_ckpt:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Information**")
                    st.json({
                        'Epoch': selected_ckpt['epoch'],
                        'Timestamp': selected_ckpt['timestamp'],
                        'Path': selected_ckpt['path']
                    })
                
                with col2:
                    st.write("**Metrics**")
                    metrics = selected_ckpt.get('metrics', {})
                    if metrics:
                        st.json(metrics)
                    else:
                        st.info("No metrics available")
                
                # Resume command
                st.code(f"python scripts/run_pipeline.py --resume {selected_ckpt['path']} ...", language="bash")
    
    except Exception as e:
        st.error(f"Error loading checkpoints: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
