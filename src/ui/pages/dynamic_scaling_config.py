"""
Dynamic Scaling Configuration Page
Advanced configuration and presets for dynamic scaling system
"""
import streamlit as st
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.models.dynamic_hybrid_buffer import ScalingConfig
    SCALING_CONFIG_AVAILABLE = True
except ImportError:
    SCALING_CONFIG_AVAILABLE = False


def render():
    """Render dynamic scaling configuration page"""
    
    st.title("⚙️ Dynamic Scaling Configuration")
    st.markdown("**Advanced configuration and presets for the dynamic scaling system**")
    
    if not SCALING_CONFIG_AVAILABLE:
        st.error("ScalingConfig not available. Please check installation.")
        return
    
    # Tabs for different modes
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Presets",
        "🔧 Custom Configuration",
        "📊 Configuration Viewer",
        "📖 Documentation"
    ])
    
    with tab1:
        render_presets()
    
    with tab2:
        render_custom_config()
    
    with tab3:
        render_config_viewer()
    
    with tab4:
        render_documentation()


def render_presets():
    """Render preset configurations"""
    
    st.markdown("### 🎯 Configuration Presets")
    st.markdown("Choose a preset configuration optimized for different dataset sizes")
    
    # Preset selection
    preset = st.selectbox(
        "Select Preset",
        [
            "Small Dataset (5-25 clusters, <5K sequences)",
            "Medium Dataset (25-100 clusters, 5K-50K sequences)",
            "Large Dataset (100-500 clusters, 50K-500K sequences)",
            "Very Large Dataset (500-1000 clusters, >500K sequences)",
            "Maximum Performance (1000+ clusters, millions of sequences)"
        ]
    )
    
    # Map preset to configuration
    preset_configs = {
        "Small Dataset": {
            'n_clusters': 10,
            'dataset_size': 2500,
            'memory_budget_gb': 1.0,
            'target_accuracy': 0.85,
            'description': "Optimized for small datasets with high accuracy targets"
        },
        "Medium Dataset": {
            'n_clusters': 50,
            'dataset_size': 25000,
            'memory_budget_gb': 2.0,
            'target_accuracy': 0.80,
            'description': "Balanced configuration for typical eDNA datasets"
        },
        "Large Dataset": {
            'n_clusters': 250,
            'dataset_size': 250000,
            'memory_budget_gb': 4.0,
            'target_accuracy': 0.75,
            'description': "Optimized for large-scale biodiversity assessments"
        },
        "Very Large Dataset": {
            'n_clusters': 750,
            'dataset_size': 750000,
            'memory_budget_gb': 6.0,
            'target_accuracy': 0.70,
            'description': "High-throughput configuration for massive datasets"
        },
        "Maximum Performance": {
            'n_clusters': 2000,
            'dataset_size': 2000000,
            'memory_budget_gb': 8.0,
            'target_accuracy': 0.70,
            'description': "Maximum capacity for extremely large datasets"
        }
    }
    
    # Get selected preset config
    preset_key = preset.split(" (")[0]
    preset_cfg = preset_configs[preset_key]
    
    # Display preset info
    st.info(f"ℹ️ {preset_cfg['description']}")
    
    # Generate configuration
    config = ScalingConfig.auto_scale(
        n_clusters=preset_cfg['n_clusters'],
        dataset_size=preset_cfg['dataset_size'],
        memory_budget_gb=preset_cfg['memory_budget_gb'],
        target_accuracy=preset_cfg['target_accuracy']
    )
    
    # Display generated configuration
    st.markdown("#### Generated Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Buffer Configuration:**")
        st.write(f"• Exemplars per Cluster: {config.exemplars_per_cluster}")
        st.write(f"• Uncertainty Buffer: {config.uncertainty_buffer_size:,}")
        st.write(f"• Recent Buffer: {config.recent_buffer_size:,}")
        st.write(f"• Temperature Scaling: {config.temperature:.2f}")
        st.write(f"• Replay Ratio: {config.replay_ratio:.2f}")
    
        with col2:
            st.markdown("**Model Configuration:**")
            st.write(f"• Architecture: {config.hidden_dims}")
            st.write(f"• Batch Size: {config.batch_size}")
            st.write(f"• Dropout Rate: {config.dropout_rate:.2f}")
            st.write(f"• EWC Lambda: {config.ewc_lambda}")
            st.write(f"• Use LoRA: {config.use_lora}")
    
    # Memory estimate
    total_samples = (config.exemplars_per_cluster * preset_cfg['n_clusters'] + 
                    config.uncertainty_buffer_size + 
                    config.recent_buffer_size)
    memory_mb = total_samples * 768 * 4 / (1024**2)
    
    st.markdown("#### 💾 Memory Estimate")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", f"{total_samples:,}")
    
    with col2:
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    with col3:
        usage_pct = 100 * memory_mb / (preset_cfg['memory_budget_gb'] * 1024)
        st.metric("Budget Usage", f"{usage_pct:.1f}%")
    
    # Export configuration
    if st.button("📥 Export Configuration", type="primary"):
        export_config(config, f"preset_{preset_key.lower().replace(' ', '_')}.json")


def render_custom_config():
    """Render custom configuration builder"""
    
    st.markdown("### 🔧 Custom Configuration Builder")
    st.markdown("Build a custom configuration with fine-grained control")
    
    # Input parameters
    st.markdown("#### Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.number_input(
            "Number of Clusters",
            min_value=5,
            max_value=10000,
            value=50,
            help="Expected number of taxonomy clusters"
        )
        
        dataset_size = st.number_input(
            "Dataset Size",
            min_value=100,
            max_value=10000000,
            value=10000,
            help="Total number of sequences"
        )
    
    with col2:
        memory_budget = st.slider(
            "Memory Budget (GB)",
            min_value=0.5,
            max_value=16.0,
            value=2.0,
            step=0.5,
            help="Maximum memory for buffers"
        )
        
        target_accuracy = st.slider(
            "Target Accuracy",
            min_value=0.60,
            max_value=0.95,
            value=0.80,
            step=0.05,
            help="Target retention accuracy"
        )
    
    # Generate base configuration
    config = ScalingConfig.auto_scale(
        n_clusters=n_clusters,
        dataset_size=dataset_size,
        memory_budget_gb=memory_budget,
        target_accuracy=target_accuracy
    )
    
    # Advanced customization
    st.markdown("---")
    st.markdown("#### Advanced Customization")
    
    with st.expander("🎛️ Fine-Tune Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Buffer Sizes:**")
            exemplars = st.number_input(
                "Exemplars/Cluster",
                min_value=1,
                max_value=200,
                value=config.exemplars_per_cluster
            )
            
            uncertainty_buffer = st.number_input(
                "Uncertainty Buffer",
                min_value=0,
                max_value=10000,
                value=config.uncertainty_buffer_size
            )
            
            recent_buffer = st.number_input(
                "Recent Buffer",
                min_value=0,
                max_value=5000,
                value=config.recent_buffer_size
            )
        
        with col2:
            st.markdown("**Training Parameters:**")
            batch_size = st.number_input(
                "Batch Size",
                min_value=8,
                max_value=256,
                value=config.batch_size
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=1.0,
                max_value=5.0,
                value=config.temperature,
                step=0.1
            )
            
            replay_ratio = st.slider(
                "Replay Ratio",
                min_value=0.0,
                max_value=1.0,
                value=config.replay_ratio,
                step=0.05
            )
        
        with col3:
            st.markdown("**Model Parameters:**")
            dropout = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=config.dropout_rate,
                step=0.05
            )
            
            ewc_lambda = st.number_input(
                "EWC Lambda",
                min_value=0.0,
                max_value=1000.0,
                value=config.ewc_lambda,
                step=10.0,
                help="Elastic Weight Consolidation regularization strength"
            )
            
            use_lora = st.checkbox(
                "Use LoRA Adaptation",
                value=config.use_lora,
                help="Enable Low-Rank Adaptation for efficient fine-tuning"
            )
        
        # Update configuration with custom values
        config.exemplars_per_cluster = exemplars
        config.uncertainty_buffer_size = uncertainty_buffer
        config.recent_buffer_size = recent_buffer
        config.batch_size = batch_size
        config.temperature = temperature
        config.replay_ratio = replay_ratio
        config.dropout_rate = dropout
        config.ewc_lambda = ewc_lambda
        config.use_lora = use_lora
    
    # Display final configuration
    st.markdown("---")
    st.markdown("#### Final Configuration")
    
    display_config_summary(config, n_clusters)
    
    # Export
    if st.button("💾 Save Custom Configuration", type="primary"):
        export_config(config, "custom_scaling_config.json")


def render_config_viewer():
    """Render configuration file viewer/importer"""
    
    st.markdown("### 📊 Configuration Viewer")
    st.markdown("View and import saved configurations")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Configuration JSON",
        type=['json'],
        help="Upload a previously exported configuration file"
    )
    
    if uploaded_file:
        try:
            config_dict = json.load(uploaded_file)
            
            st.success("✓ Configuration loaded successfully!")
            
            # Display configuration
            st.markdown("#### Configuration Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Buffer Configuration:**")
                for key in ['exemplars_per_cluster', 'uncertainty_buffer_size', 
                           'recent_buffer_size', 'temperature', 'replay_ratio']:
                    if key in config_dict:
                        st.write(f"• {key}: {config_dict[key]}")
            
            with col2:
                st.markdown("**Model Configuration:**")
                for key in ['hidden_dims', 'batch_size', 'dropout_rate', 
                           'ewc_lambda', 'use_lora']:
                    if key in config_dict:
                        st.write(f"• {key}: {config_dict[key]}")
            
            # Full JSON view
            with st.expander("📄 View Full JSON"):
                st.json(config_dict)
            
            # Use configuration button
            if st.button("✓ Use This Configuration"):
                st.session_state.imported_config = config_dict
                st.success("Configuration saved to session! Use it in the Analysis page.")
        
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
    
    else:
        st.info("Upload a configuration JSON file to view and use it")


def render_documentation():
    """Render dynamic scaling documentation"""
    
    st.markdown("### 📖 Dynamic Scaling Documentation")
    
    st.markdown("""
    ## Overview
    
    The **Dynamic Scaling System** automatically adapts buffer sizes and model architecture 
    based on dataset characteristics and memory constraints. It supports scaling from 
    **10 to 10,000+ clusters** without manual intervention.
    
    ## Key Features
    
    ### 1. **Hybrid Memory Buffers**
    - **Exemplar Buffer**: Diverse representatives from each cluster
    - **Uncertainty Buffer**: High-uncertainty samples for challenging cases
    - **Recent Buffer**: Most recently seen samples for temporal patterns
    
    ### 2. **Auto-Adaptation**
    - Monitors cluster count and dataset size
    - Automatically adjusts buffer capacities
    - Scales model architecture dynamically
    - Maintains memory budget constraints
    
    ### 3. **Advanced Refinements**
    
    #### Temperature Scaling (Confidence Detection)
    - Identifies high-uncertainty samples using temperature-scaled softmax
    - Higher temperature = more sensitive to confidence differences
    - Default: 2.0 (balanced sensitivity)
    
    #### Reservoir Sampling (Diverse Exemplars)
    - Ensures diverse representation within each cluster
    - Uses weighted reservoir sampling
    - Prevents redundant similar samples
    
    #### Mini-Retrieval (Smart Replay)
    - Retrieves most relevant samples for current batch
    - k-nearest neighbors in embedding space
    - Improves replay effectiveness
    
    #### Centroid Updates (Cluster Tracking)
    - Maintains running centroids for each cluster
    - Enables drift detection
    - Supports adaptive clustering
    
    #### LoRA Adaptation (Efficient Fine-Tuning)
    - Low-Rank Adaptation for model updates
    - Reduces parameter count
    - Faster adaptation to new clusters
    
    ## Configuration Parameters
    
    ### Memory Budget
    - **Range**: 0.5 - 16 GB
    - **Default**: Auto-detected (50% of available RAM)
    - **Impact**: Larger budget = more samples retained
    
    ### Target Accuracy
    - **Range**: 0.60 - 0.95
    - **Default**: 0.80 (80%)
    - **Impact**: Higher target = larger buffers
    
    ### Auto-Adapt
    - **Enabled**: System automatically adjusts as clusters grow
    - **Disabled**: Configuration remains fixed
    - **Recommendation**: Enable for datasets >50 clusters
    
    ## Usage Guidelines
    
    ### When to Use Dynamic Scaling
    
    ✅ **Use when:**
    - Dataset has >25 clusters
    - Cluster count unknown or variable
    - Memory is constrained
    - Need automatic optimization
    - Processing incremental data
    
    ❌ **Don't use when:**
    - Very small datasets (<1000 sequences)
    - Cluster count is fixed and small (<10)
    - Legacy compatibility required
    - Need full manual control
    
    ### Preset Recommendations
    
    | Dataset Size | Clusters | Preset | Memory | Target Acc |
    |--------------|----------|--------|--------|------------|
    | <5K sequences | 5-25 | Small | 1 GB | 85% |
    | 5K-50K | 25-100 | Medium | 2 GB | 80% |
    | 50K-500K | 100-500 | Large | 4 GB | 75% |
    | >500K | 500+ | Very Large | 6+ GB | 70% |
    
    ## Performance Expectations
    
    ### Memory Usage
    - Typical: 0.2-0.3% of memory budget per cluster
    - Example: 50 clusters = ~100 MB for 2 GB budget
    
    ### Accuracy
    - **10 clusters**: 95-96% retention
    - **25 clusters**: 89-92% retention
    - **50 clusters**: 85-88% retention
    - **100+ clusters**: 75-85% retention
    
    ### Adaptation Frequency
    - Adapts every ~10-20 clusters (typical)
    - More frequent with aggressive growth
    - Can be disabled for fixed configuration
    
    ## Troubleshooting
    
    ### High Memory Usage
    - Reduce target accuracy
    - Decrease memory budget
    - Lower exemplars per cluster
    
    ### Low Accuracy
    - Increase memory budget
    - Raise target accuracy
    - Increase exemplars per cluster
    - Enable auto-adapt
    
    ### Too Many Adaptations
    - Increase adaptation threshold
    - Use larger initial configuration
    - Disable auto-adapt for stable workloads
    
    ## References
    
    - **Implementation**: `src/models/dynamic_hybrid_buffer.py`
    - **Pipeline Integration**: `scripts/run_taxonomy_pipeline_v2.py`
    - **Configuration**: `src/models/hybrid_memory_buffer.py`
    """)


def display_config_summary(config: ScalingConfig, n_clusters: int):
    """Display configuration summary with metrics"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Buffer Settings:**")
        st.write(f"• Exemplars/Cluster: {config.exemplars_per_cluster}")
        st.write(f"• Uncertainty Buffer: {config.uncertainty_buffer_size:,}")
        st.write(f"• Recent Buffer: {config.recent_buffer_size:,}")
        st.write(f"• Temperature: {config.temperature:.2f}")
        st.write(f"• Replay Ratio: {config.replay_ratio:.2f}")
    
    with col2:
        st.markdown("**Model Settings:**")
        st.write(f"• Architecture: {config.hidden_dims}")
        st.write(f"• Batch Size: {config.batch_size}")
        st.write(f"• Dropout: {config.dropout_rate:.2f}")
        st.write(f"• EWC Lambda: {config.ewc_lambda}")
        st.write(f"• Use LoRA: {config.use_lora}")
    
    # Memory calculation
    total_samples = (config.exemplars_per_cluster * n_clusters + 
                    config.uncertainty_buffer_size + 
                    config.recent_buffer_size)
    memory_mb = total_samples * 768 * 4 / (1024**2)
    
    st.metric(
        "Estimated Memory Usage",
        f"{memory_mb:.1f} MB",
        f"{100 * memory_mb / (config.memory_budget_gb * 1024):.1f}% of budget"
    )


def export_config(config: ScalingConfig, filename: str):
    """Export configuration to JSON"""
    
    config_dict = config.to_dict()
    config_json = json.dumps(config_dict, indent=2)
    
    st.download_button(
        label=f"📥 Download {filename}",
        data=config_json,
        file_name=filename,
        mime="application/json",
        type="primary"
    )
    
    st.success(f"✓ Configuration exported: {filename}")
