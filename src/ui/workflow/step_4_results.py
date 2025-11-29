"""
Step 4: Results Dashboard Component
Tabbed results view integrating all analysis outputs
"""

import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ui.task_manager import get_task_manager


def render_results():
    """Render results dashboard"""
    
    # Get current task results
    current_task_id = st.session_state.workflow_current_task_id
    results = st.session_state.workflow_results
    
    if not current_task_id and not results:
        st.info("No results yet. Complete an analysis to view results here.")
        if st.button("← Start New Analysis"):
            st.session_state.workflow_step = 1
            st.rerun()
        return
    
    # Get task info
    task_manager = get_task_manager()
    task = task_manager.get_task(current_task_id) if current_task_id else None
    
    # Header with task info
    if task:
        st.markdown(f"## Results: {task.name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "✅ Completed" if task.status.value == "completed" else task.status.value.title())
        with col2:
            if task.completed_at:
                st.metric("Completed", task.completed_at.split('T')[0])
        with col3:
            elapsed = format_duration(task.elapsed_time)
            st.metric("Duration", elapsed)
    else:
        st.markdown("## Analysis Results")
    
    st.divider()
    
    # Quick summary (always visible)
    render_quick_summary(results)
    
    st.divider()
    
    # Tabbed results
    tabs = st.tabs([
        "📊 Overview",
        "🧬 Diversity",
        "🔬 Taxonomy",
        "📈 Quality",
        "🤖 Model",
        "⚡ Scaling"
    ])
    
    with tabs[0]:
        render_overview_tab(results)
    
    with tabs[1]:
        render_diversity_tab(results)
    
    with tabs[2]:
        render_taxonomy_tab(results)
    
    with tabs[3]:
        render_quality_tab(results)
    
    with tabs[4]:
        render_model_tab(results)
    
    with tabs[5]:
        render_scaling_tab(results)
    
    # Actions
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.workflow_step = 1
            st.session_state.workflow_dataset = None
            st.session_state.workflow_current_task_id = None
            st.session_state.workflow_results = None
            st.rerun()
    
    with col2:
        if st.button("📥 Export Report", use_container_width=True):
            st.info("Export feature coming soon!")
    
    with col3:
        if st.button("🔗 Share Results", use_container_width=True):
            st.info("Share feature coming soon!")


def render_quick_summary(results: Optional[Dict[str, Any]]):
    """Render quick summary box"""
    st.markdown("### Quick Summary")
    
    if not results:
        # Demo data
        results = {
            'sequences_analyzed': 45120,
            'clusters_found': 127,
            'novel_taxa': 23,
            'accuracy': 94.2,
            'mean_length': 324,
            'gc_content': 47.3,
            'quality_score': 8.5,
            'diversity_index': 3.47
        }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Sequences",
            f"{results.get('sequences_analyzed', 0):,}",
            help="Total sequences analyzed"
        )
    
    with col2:
        st.metric(
            "Clusters",
            f"{results.get('clusters_found', 0)}",
            help="Distinct clusters identified"
        )
    
    with col3:
        st.metric(
            "Novel Taxa",
            f"{results.get('novel_taxa', 0)}",
            help="Potentially novel taxonomic groups"
        )
    
    with col4:
        st.metric(
            "Accuracy",
            f"{results.get('accuracy', 0):.1f}%",
            help="Model classification accuracy"
        )


def render_overview_tab(results: Optional[Dict[str, Any]]):
    """Overview tab with executive summary"""
    st.markdown("### Executive Summary")
    
    summary_text = f"""
    Analyzed **{results.get('sequences_analyzed', 45120):,}** high-quality sequences from the dataset.
    Identified **{results.get('clusters_found', 127)}** distinct clusters with **{results.get('accuracy', 94.2):.1f}%** 
    classification confidence. Detected **{results.get('novel_taxa', 23)}** potentially novel taxa requiring 
    further investigation. Overall diversity index: **{results.get('diversity_index', 3.47):.2f}** (Shannon).
    """
    
    st.info(summary_text)
    
    st.divider()
    
    # Key Metrics Grid
    st.markdown("### Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Sequence Length", f"{results.get('mean_length', 324)} bp")
        st.metric("GC Content", f"{results.get('gc_content', 47.3):.1f}%")
    
    with col2:
        st.metric("Quality Score", f"{results.get('quality_score', 8.5):.1f}/10")
        st.metric("Diversity Index", f"{results.get('diversity_index', 3.47):.2f}")
    
    with col3:
        st.metric("Clusters", f"{results.get('clusters_found', 127)}")
        st.metric("Novel Taxa", f"{results.get('novel_taxa', 23)}")
    
    st.divider()
    
    # Top organisms (demo data)
    st.markdown("### Top Organisms")
    
    demo_organisms = pd.DataFrame({
        'Organism': [
            'Prochlorococcus marinus',
            'Synechococcus sp.',
            'Pelagibacter ubique',
            'Candidatus Actinomarina minuta',
            '[Novel Clade A]'
        ],
        'Count': [8234, 6891, 5432, 3221, 2109],
        'Percentage': [18.3, 15.3, 12.0, 7.1, 4.7]
    })
    
    st.dataframe(demo_organisms, use_container_width=True, hide_index=True)


def render_diversity_tab(results: Optional[Dict[str, Any]]):
    """Diversity analysis results"""
    st.markdown("### Diversity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Alpha Diversity")
        st.metric("Shannon Index", "3.47")
        st.metric("Simpson Index", "0.89")
        st.metric("Species Richness", "127")
    
    with col2:
        st.markdown("#### Beta Diversity")
        st.metric("Bray-Curtis", "0.45")
        st.metric("Jaccard Index", "0.62")
    
    st.divider()
    
    # Rarefaction curve (demo)
    st.markdown("#### Rarefaction Curve")
    
    demo_rarefaction = pd.DataFrame({
        'Sequences': list(range(0, 50000, 5000)),
        'Species': [0, 45, 75, 95, 108, 116, 121, 124, 126, 127]
    })
    
    fig = px.line(
        demo_rarefaction,
        x='Sequences',
        y='Species',
        title='Species Accumulation Curve',
        labels={'Sequences': 'Number of Sequences', 'Species': 'Number of Species'}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_taxonomy_tab(results: Optional[Dict[str, Any]]):
    """Taxonomy classification results"""
    st.markdown("### Taxonomy Classification")
    
    # Taxonomic distribution (demo)
    demo_taxonomy = pd.DataFrame({
        'Phylum': ['Proteobacteria', 'Cyanobacteria', 'Actinobacteria', 'Bacteroidetes', 'Other'],
        'Count': [15234, 12456, 8901, 5432, 3097],
        'Percentage': [33.8, 27.6, 19.7, 12.0, 6.9]
    })
    
    fig = px.pie(
        demo_taxonomy,
        values='Count',
        names='Phylum',
        title='Taxonomic Distribution by Phylum',
        hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.markdown("#### Classification Details")
    st.dataframe(demo_taxonomy, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.markdown("#### Novel Taxa Detected")
    st.info("23 potentially novel taxonomic groups identified requiring further investigation")


def render_quality_tab(results: Optional[Dict[str, Any]]):
    """Quality analysis results"""
    st.markdown("### Quality Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Quality", "8.5/10")
        st.metric("Min Quality", "6.2/10")
    
    with col2:
        st.metric("Max Quality", "9.8/10")
        st.metric("Std Dev", "0.85")
    
    with col3:
        st.metric("Sequences Filtered", "1,234")
        st.metric("Filter Rate", "2.7%")
    
    st.divider()
    
    # Quality distribution (demo)
    st.markdown("#### Quality Score Distribution")
    
    import numpy as np
    demo_quality = pd.DataFrame({
        'Quality Score': np.random.normal(8.5, 0.85, 1000)
    })
    
    fig = px.histogram(
        demo_quality,
        x='Quality Score',
        nbins=30,
        title='Distribution of Quality Scores',
        labels={'Quality Score': 'Quality Score', 'count': 'Frequency'}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_tab(results: Optional[Dict[str, Any]]):
    """Model training results"""
    st.markdown("### Model Training Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Validation Accuracy", "94.2%")
        st.metric("Training Loss", "0.15")
    
    with col2:
        st.metric("Epochs Completed", "50")
        st.metric("Best Epoch", "42")
    
    with col3:
        st.metric("Training Time", "12m 34s")
        st.metric("Model Size", "245 MB")
    
    st.divider()
    
    # Training curves (demo)
    st.markdown("#### Training Progress")
    
    epochs = list(range(1, 51))
    demo_training = pd.DataFrame({
        'Epoch': epochs,
        'Training Loss': [0.8 - (i * 0.013) + np.random.normal(0, 0.02) for i in epochs],
        'Validation Accuracy': [60 + (i * 0.7) + np.random.normal(0, 1.5) for i in epochs]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=demo_training['Epoch'],
        y=demo_training['Training Loss'],
        name='Training Loss',
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=demo_training['Epoch'],
        y=demo_training['Validation Accuracy'],
        name='Validation Accuracy',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Training Progress',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss', side='left'),
        yaxis2=dict(title='Accuracy (%)', side='right', overlaying='y')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_scaling_tab(results: Optional[Dict[str, Any]]):
    """Dynamic scaling results"""
    st.markdown("### Dynamic Scaling Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Clusters", "127")
        st.metric("Initial Clusters", "50")
    
    with col2:
        st.metric("Scaling Events", "3")
        st.metric("Peak Clusters", "187")
    
    with col3:
        st.metric("Memory Saved", "2.3 GB")
        st.metric("Buffer Size", "1,777 samples")
    
    st.divider()
    
    # Scaling timeline (demo)
    st.markdown("#### Cluster Evolution")
    
    demo_scaling = pd.DataFrame({
        'Time': list(range(0, 100, 5)),
        'Clusters': [50, 65, 78, 95, 110, 127, 145, 167, 187, 175, 162, 148, 135, 127, 127, 127, 127, 127, 127, 127]
    })
    
    fig = px.line(
        demo_scaling,
        x='Time',
        y='Clusters',
        title='Dynamic Cluster Scaling Over Time',
        labels={'Time': 'Processing Stage', 'Clusters': 'Number of Clusters'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.markdown("#### Configuration History")
    st.info("3 automatic scaling adaptations performed to optimize memory usage")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
