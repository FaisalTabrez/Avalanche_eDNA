"""
Dynamic Scaling Visualization Components
Reusable Streamlit components for visualizing dynamic scaling metrics
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def render_adaptation_timeline(adaptation_history: List[Dict]) -> None:
    """
    Render timeline of adaptation events
    
    Args:
        adaptation_history: List of adaptation events from DynamicHybridBuffer
    """
    if not adaptation_history:
        st.info("No adaptations occurred - configuration remained stable")
        return
    
    st.markdown("#### 📅 Adaptation Timeline")
    
    # Extract data
    cluster_ids = []
    old_clusters = []
    new_clusters = []
    triggers = []
    
    for event in adaptation_history:
        cluster_ids.append(event.get('cluster_id', 0))
        old_clusters.append(event.get('old_n_clusters', 0))
        new_clusters.append(event.get('new_n_clusters', 0))
        triggers.append(event.get('trigger', 'auto'))
    
    # Create timeline visualization
    fig = go.Figure()
    
    # Add line for cluster count evolution
    fig.add_trace(go.Scatter(
        x=cluster_ids,
        y=new_clusters,
        mode='lines+markers',
        name='Cluster Count',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=10, symbol='circle')
    ))
    
    # Add markers for adaptation points
    fig.add_trace(go.Scatter(
        x=cluster_ids,
        y=new_clusters,
        mode='markers',
        name='Adaptation Events',
        marker=dict(
            size=15,
            symbol='star',
            color='#ff7f0e',
            line=dict(width=2, color='white')
        ),
        text=[f"Adapted at cluster {cid}" for cid in cluster_ids],
        hovertemplate='<b>Cluster %{x}</b><br>New count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Configuration Adaptation Timeline",
        xaxis_title="Cluster ID",
        yaxis_title="Cluster Count",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("📋 Adaptation Details"):
        adapt_df = pd.DataFrame({
            'Cluster ID': cluster_ids,
            'Old Count': old_clusters,
            'New Count': new_clusters,
            'Trigger': triggers
        })
        st.dataframe(adapt_df, hide_index=True, use_container_width=True)


def render_buffer_composition(buffer_stats: Dict[str, Any]) -> None:
    """
    Render buffer composition pie chart
    
    Args:
        buffer_stats: Comprehensive stats from HybridMemoryBuffer
    """
    st.markdown("#### 🥧 Buffer Composition")
    
    # Extract sizes
    exemplar_size = buffer_stats.get('exemplar', {}).get('total_exemplars', 0)
    uncertainty_size = buffer_stats.get('uncertainty', {}).get('size', 0)
    recent_size = buffer_stats.get('recent', {}).get('size', 0)
    
    total_size = exemplar_size + uncertainty_size + recent_size
    
    if total_size == 0:
        st.warning("No samples in buffer yet")
        return
    
    # Create pie chart
    labels = ['Exemplar Buffer', 'Uncertainty Buffer', 'Recent Buffer']
    values = [exemplar_size, uncertainty_size, recent_size]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='inside',
        hovertemplate='<b>%{label}</b><br>%{value:,} samples<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"Total Buffer Size: {total_size:,} samples",
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Exemplar Buffer",
            f"{exemplar_size:,}",
            help="Diverse representatives from each cluster"
        )
    
    with col2:
        st.metric(
            "Uncertainty Buffer",
            f"{uncertainty_size:,}",
            help="High-uncertainty samples for challenging cases"
        )
    
    with col3:
        st.metric(
            "Recent Buffer",
            f"{recent_size:,}",
            help="Most recently seen samples"
        )


def render_memory_usage_gauge(memory_mb: float, budget_gb: float) -> None:
    """
    Render memory usage gauge
    
    Args:
        memory_mb: Current memory usage in MB
        budget_gb: Total memory budget in GB
    """
    st.markdown("#### 💾 Memory Usage")
    
    budget_mb = budget_gb * 1024
    usage_pct = 100 * memory_mb / budget_mb
    
    # Determine color based on usage
    if usage_pct < 50:
        color = '#2ecc71'  # Green
    elif usage_pct < 80:
        color = '#f39c12'  # Orange
    else:
        color = '#e74c3c'  # Red
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=memory_mb,
        delta={'reference': budget_mb * 0.5, 'increasing': {'color': 'red'}},
        gauge={
            'axis': {'range': [None, budget_mb]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, budget_mb * 0.5], 'color': 'lightgray'},
                {'range': [budget_mb * 0.5, budget_mb * 0.8], 'color': 'gray'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': budget_mb * 0.9
            }
        },
        title={'text': "Memory Usage (MB)"}
    ))
    
    fig.update_layout(
        height=300,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current", f"{memory_mb:.1f} MB")
    
    with col2:
        st.metric("Budget", f"{budget_gb:.1f} GB ({budget_mb:.0f} MB)")
    
    with col3:
        status = "✓ OK" if usage_pct < 80 else "⚠ High" if usage_pct < 95 else "🔴 Critical"
        st.metric("Status", status, f"{usage_pct:.1f}%")


def render_configuration_diff(old_config: Dict, new_config: Dict) -> None:
    """
    Render configuration difference viewer
    
    Args:
        old_config: Previous ScalingConfig as dict
        new_config: New ScalingConfig as dict
    """
    st.markdown("#### 🔄 Configuration Changes")
    
    # Key parameters to compare
    params = [
        ('exemplars_per_cluster', 'Exemplars/Cluster'),
        ('uncertainty_buffer_size', 'Uncertainty Buffer'),
        ('recent_buffer_size', 'Recent Buffer'),
        ('temperature', 'Temperature'),
        ('hidden_dims', 'Architecture'),
        ('batch_size', 'Batch Size'),
        ('replay_ratio', 'Replay Ratio'),
        ('dropout_rate', 'Dropout Rate')
    ]
    
    changes = []
    for key, label in params:
        old_val = old_config.get(key, 'N/A')
        new_val = new_config.get(key, 'N/A')
        
        if old_val != new_val:
            changes.append({
                'Parameter': label,
                'Old Value': str(old_val),
                'New Value': str(new_val),
                'Changed': '✓'
            })
        else:
            changes.append({
                'Parameter': label,
                'Old Value': str(old_val),
                'New Value': str(new_val),
                'Changed': ''
            })
    
    # Display as table with highlighting
    df = pd.DataFrame(changes)
    
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            'Changed': st.column_config.TextColumn(
                'Changed',
                width='small'
            )
        }
    )


def render_training_metrics_evolution(training_history: Dict[str, Any]) -> None:
    """
    Render training metrics evolution across clusters
    
    Args:
        training_history: Training history from pipeline
    """
    st.markdown("#### 📈 Training Metrics Evolution")
    
    if not training_history:
        st.info("No training history available")
        return
    
    # Extract metrics
    cluster_ids = []
    losses = []
    accuracies = []
    
    for cluster_key in sorted(training_history.keys()):
        if cluster_key.startswith('cluster_'):
            cluster_id = int(cluster_key.split('_')[1])
            metrics = training_history[cluster_key]
            
            cluster_ids.append(cluster_id)
            losses.append(metrics.get('final_loss', 0))
            accuracies.append(metrics.get('final_accuracy', 0))
    
    if not cluster_ids:
        st.warning("No cluster metrics found")
        return
    
    # Create dual-axis plot
    fig = go.Figure()
    
    # Loss line
    fig.add_trace(go.Scatter(
        x=cluster_ids,
        y=losses,
        mode='lines+markers',
        name='Loss',
        line=dict(color='#e74c3c', width=2),
        yaxis='y'
    ))
    
    # Accuracy line
    fig.add_trace(go.Scatter(
        x=cluster_ids,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy (%)',
        line=dict(color='#2ecc71', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Training Performance Across Clusters",
        xaxis=dict(title='Cluster ID'),
        yaxis=dict(title='Loss', side='left', color='#e74c3c'),
        yaxis2=dict(title='Accuracy (%)', side='right', overlaying='y', color='#2ecc71'),
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Loss", f"{np.mean(losses):.4f}")
    
    with col2:
        st.metric("Avg Accuracy", f"{np.mean(accuracies):.1f}%")
    
    with col3:
        st.metric("Best Accuracy", f"{max(accuracies):.1f}%")
    
    with col4:
        st.metric("Worst Accuracy", f"{min(accuracies):.1f}%")


def render_cluster_size_distribution(cluster_labels: np.ndarray) -> None:
    """
    Render cluster size distribution
    
    Args:
        cluster_labels: Numpy array of cluster assignments
    """
    st.markdown("#### 📊 Cluster Size Distribution")
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    # Create bar chart
    fig = px.bar(
        x=unique,
        y=counts,
        labels={'x': 'Cluster ID', 'y': 'Number of Sequences'},
        title=f"Distribution across {len(unique)} Clusters",
        color=counts,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis={'type': 'category'},
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clusters", len(unique))
    
    with col2:
        st.metric("Avg Size", f"{counts.mean():.1f}")
    
    with col3:
        st.metric("Largest Cluster", counts.max())
    
    with col4:
        st.metric("Smallest Cluster", counts.min())


def render_buffer_evolution_timeline(adaptation_history: List[Dict]) -> None:
    """
    Render buffer size evolution over time
    
    Args:
        adaptation_history: List of adaptation events
    """
    st.markdown("#### 📈 Buffer Size Evolution")
    
    if not adaptation_history:
        st.info("No adaptation history - buffers remained constant")
        return
    
    # Extract buffer sizes over time
    cluster_ids = []
    exemplar_sizes = []
    uncertainty_sizes = []
    recent_sizes = []
    
    for event in adaptation_history:
        cluster_id = event.get('cluster_id', 0)
        new_config = event.get('new_config', {})
        n_clusters = new_config.get('n_clusters', 1)
        
        cluster_ids.append(cluster_id)
        exemplar_sizes.append(new_config.get('exemplars_per_cluster', 0) * n_clusters)
        uncertainty_sizes.append(new_config.get('uncertainty_buffer_size', 0))
        recent_sizes.append(new_config.get('recent_buffer_size', 0))
    
    # Create stacked area chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cluster_ids,
        y=exemplar_sizes,
        mode='lines',
        name='Exemplar Buffer',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=0),
        fillcolor='rgba(46, 204, 113, 0.6)'
    ))
    
    fig.add_trace(go.Scatter(
        x=cluster_ids,
        y=uncertainty_sizes,
        mode='lines',
        name='Uncertainty Buffer',
        fill='tonexty',
        line=dict(color='#3498db', width=0),
        fillcolor='rgba(52, 152, 219, 0.6)'
    ))
    
    fig.add_trace(go.Scatter(
        x=cluster_ids,
        y=recent_sizes,
        mode='lines',
        name='Recent Buffer',
        fill='tonexty',
        line=dict(color='#e74c3c', width=0),
        fillcolor='rgba(231, 76, 60, 0.6)'
    ))
    
    fig.update_layout(
        title="Buffer Capacity Evolution",
        xaxis_title="Cluster ID",
        yaxis_title="Buffer Size (samples)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_dynamic_scaling_summary(pipeline) -> None:
    """
    Render comprehensive dynamic scaling summary
    
    Args:
        pipeline: TaxonomyClassificationPipeline instance with dynamic_buffer
    """
    if not hasattr(pipeline, 'dynamic_buffer') or pipeline.dynamic_buffer is None:
        st.warning("Dynamic scaling not enabled for this pipeline")
        return
    
    st.markdown("### 🚀 Dynamic Scaling Summary")
    
    dynamic_buffer = pipeline.dynamic_buffer
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📅 Adaptations",
        "💾 Memory",
        "⚙️ Configuration"
    ])
    
    with tab1:
        # Current configuration
        config = dynamic_buffer.current_config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Configuration**")
            st.write(f"• Clusters: {config.n_clusters}")
            st.write(f"• Exemplars/Cluster: {config.exemplars_per_cluster}")
            st.write(f"• Uncertainty Buffer: {config.uncertainty_buffer_size:,}")
            st.write(f"• Recent Buffer: {config.recent_buffer_size:,}")
        
        with col2:
            st.markdown("**Model Settings**")
            st.write(f"• Architecture: {config.hidden_dims}")
            st.write(f"• Batch Size: {config.batch_size}")
            st.write(f"• Temperature: {config.temperature:.2f}")
            st.write(f"• Dropout: {config.dropout_rate:.2f}")
        
        # Buffer composition
        if hasattr(dynamic_buffer, 'buffer'):
            stats = dynamic_buffer.buffer.get_comprehensive_stats()
            render_buffer_composition(stats)
    
    with tab2:
        # Adaptation timeline
        render_adaptation_timeline(dynamic_buffer.adaptation_history)
        
        # Buffer evolution
        if dynamic_buffer.adaptation_history:
            render_buffer_evolution_timeline(dynamic_buffer.adaptation_history)
    
    with tab3:
        # Memory usage
        if hasattr(dynamic_buffer, 'buffer'):
            stats = dynamic_buffer.buffer.get_comprehensive_stats()
            total_samples = (stats['exemplar']['total_exemplars'] + 
                            stats['uncertainty']['size'] + 
                            stats['recent']['size'])
            memory_mb = total_samples * 768 * 4 / (1024**2)
            
            render_memory_usage_gauge(memory_mb, config.memory_budget_gb)
    
    with tab4:
        # Configuration history
        st.markdown("#### Configuration Evolution")
        
        if len(dynamic_buffer.adaptation_history) > 0:
            # Show first and last configs
            first_event = dynamic_buffer.adaptation_history[0]
            last_event = dynamic_buffer.adaptation_history[-1]
            
            old_config = first_event.get('old_config', {})
            new_config = last_event.get('new_config', {})
            
            render_configuration_diff(old_config, new_config)
        else:
            st.info("No configuration changes - remained stable throughout training")
