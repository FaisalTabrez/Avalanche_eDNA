"""
Model Training Page with Integrated Dynamic Scaling
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
import tempfile
from pathlib import Path
from src.utils.config import config as app_config

try:
    from src.models.tokenizer import DNATokenizer
    from src.models.embeddings import DNAContrastiveModel, DNATransformerEmbedder, DNAAutoencoder
    from src.models.trainer import EmbeddingTrainer
except ImportError:
    pass

try:
    from src.models.dynamic_hybrid_buffer import ScalingConfig
    SCALING_CONFIG_AVAILABLE = True
except ImportError:
    SCALING_CONFIG_AVAILABLE = False

try:
    from src.utils.sra_integration import create_sra_data_source_selector
except ImportError:
    create_sra_data_source_selector = None

def render():
    """Display model training page with integrated dynamic scaling"""
    st.title("🤖 Model Training & Dynamic Scaling")
    st.markdown("""
    Train custom DNA embedding models with **dynamic scaling** for continual learning.
    
    **Model Architecture:** Choose between **Contrastive Learning** (recommended), Autoencoders, or Transformers.  
    **Dynamic Scaling:** Automatically adapts cluster counts and memory usage as your model trains on streaming data.
    """)
    
    tab1, tab2 = st.tabs(["🚀 Train New Model", "📁 Manage Models"])
    
    # --- Tab 1: Train New Model ---
    with tab1:
        render_training_interface()

    # --- Tab 2: Manage Models ---
    with tab2:
        show_model_management()


def render_training_interface():
    """Render the training interface with integrated scaling"""
    
    st.markdown("### 1️⃣ Data Selection")
    
    # Use SRA-integrated data source selector if available
    if create_sra_data_source_selector:
        source_type, sequences_path, metadata = create_sra_data_source_selector()
        
        if metadata:
            st.info(f"📊 Data source: {metadata.get('source', 'unknown').upper()}")
            if metadata.get('source') == 'sra':
                st.success(f"🧬 SRA Accession: {metadata.get('accession')}")
    else:
        # Fallback to original data source selection
        data_source = st.radio("Data Source", ["Upload New File", "Select Existing Dataset"])
        
        sequences_path = None
        metadata = None
        
        if data_source == "Upload New File":
            uploaded_file = st.file_uploader("Upload FASTA File", type=['fasta', 'fa'])
            if uploaded_file:
                # Save to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    sequences_path = tmp.name
        else:
            # List files in datasets dir
            datasets_dir = Path(app_config.get('storage.datasets_dir', 'data/datasets'))
            if datasets_dir.exists():
                files = list(datasets_dir.glob("*.fasta")) + list(datasets_dir.glob("*.fa"))
                if files:
                    selected_file = st.selectbox("Select Dataset", files, format_func=lambda x: x.name)
                    sequences_path = str(selected_file)
                else:
                    st.warning("No datasets found in storage.")
            else:
                st.warning("Datasets directory not found.")
    
    # Labels (Optional)
    st.markdown("#### 🏷️ Labels (Optional)")
    st.markdown("Upload labels for supervised training. Leave empty for unsupervised/contrastive learning.")
    labels_file = st.file_uploader("Upload Labels", type=['csv', 'txt'])
    labels_path = None
    if labels_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(labels_file.name).suffix) as tmp:
            tmp.write(labels_file.getvalue())
            labels_path = tmp.name
    
    st.markdown("---")
    st.markdown("### 2️⃣ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧠 Model Architecture")
        model_type = st.selectbox(
            "Architecture Type", 
            ["Contrastive Learning", "Transformer", "Autoencoder"],
            help="Contrastive Learning is recommended for best embedding quality."
        )
        
        epochs = st.number_input("Training Epochs", min_value=1, max_value=1000, value=50)
        batch_size = st.number_input("Batch Size", min_value=2, max_value=512, value=32)
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=1e-4, format="%.6f")
        
    with col2:
        st.markdown("#### 🔧 Model Parameters")
        embedding_dim = st.number_input("Embedding Dimension", min_value=32, max_value=2048, value=256)
        
        if model_type == "Contrastive Learning":
            projection_dim = st.number_input("Projection Dimension", min_value=32, max_value=1024, value=64)  # OPTIMIZED
            temperature = st.number_input("Temperature", min_value=0.01, max_value=1.0, value=0.1)
        else:
            projection_dim = None
            temperature = None
        
        device = st.selectbox("Compute Device", ["auto", "cpu", "cuda"], 
                             help="'auto' selects GPU if available, otherwise CPU")
    
    st.markdown("---")
    st.markdown("### 3️⃣ Dynamic Scaling Configuration")
    
    # Dynamic scaling toggle
    enable_scaling = st.checkbox(
        "⚡ Enable Dynamic Scaling",
        value=True,
        help="Enable continual learning with dynamic cluster scaling and memory management"
    )
    
    scaling_config = None
    
    if enable_scaling:
        if not SCALING_CONFIG_AVAILABLE:
            st.warning("⚠️ Dynamic scaling not available. Please check installation.")
            enable_scaling = False
        else:
            scaling_config = render_scaling_config()
    else:
        st.info("ℹ️ Training without dynamic scaling (standard static training)")
    
    st.markdown("---")
    st.markdown("### 4️⃣ Output Configuration")
    
    model_name = st.text_input(
        "Model Name", 
        value=f"model_{int(time.time())}",
        help="Unique identifier for this model"
    )
    
    st.markdown("---")
    
    # Training button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not sequences_path:
                st.error("❌ Please select a sequence file.")
            else:
                train_model_ui(
                    sequences_path=sequences_path,
                    labels_path=labels_path,
                    model_type=model_type,
                    model_name=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    embedding_dim=embedding_dim,
                    projection_dim=projection_dim,
                    temperature=temperature,
                    device=device,
                    scaling_config=scaling_config,
                    enable_scaling=enable_scaling
                )


def render_scaling_config():
    """Render dynamic scaling configuration options"""
    
    st.markdown("#### ⚙️ Scaling Settings")
    
    # Preset or custom
    config_mode = st.radio(
        "Configuration Mode",
        ["🎯 Use Preset", "🔧 Custom Settings"],
        horizontal=True
    )
    
    if config_mode == "🎯 Use Preset":
        preset = st.selectbox(
            "Select Preset",
            [
                "Small Dataset (<5K sequences)",
                "Medium Dataset (5K-50K sequences)",
                "Large Dataset (50K-500K sequences)",
                "Very Large Dataset (>500K sequences)"
            ]
        )
        
        # Map preset to configuration
        preset_configs = {
            "Small Dataset (<5K sequences)": {
                'n_clusters': 10,
                'dataset_size': 2500,
                'memory_budget_gb': 1.0,
                'target_accuracy': 0.85
            },
            "Medium Dataset (5K-50K sequences)": {
                'n_clusters': 50,
                'dataset_size': 25000,
                'memory_budget_gb': 2.0,
                'target_accuracy': 0.80
            },
            "Large Dataset (50K-500K sequences)": {
                'n_clusters': 250,
                'dataset_size': 250000,
                'memory_budget_gb': 4.0,
                'target_accuracy': 0.75
            },
            "Very Large Dataset (>500K sequences)": {
                'n_clusters': 750,
                'dataset_size': 750000,
                'memory_budget_gb': 6.0,
                'target_accuracy': 0.70
            }
        }
        
        preset_cfg = preset_configs[preset]
        
        # Show what will be configured
        with st.expander("📊 Preset Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Initial Clusters", preset_cfg['n_clusters'])
                st.metric("Target Accuracy", f"{preset_cfg['target_accuracy']*100:.0f}%")
            with col2:
                st.metric("Dataset Size", f"{preset_cfg['dataset_size']:,}")
                st.metric("Memory Budget", f"{preset_cfg['memory_budget_gb']:.1f} GB")
        
        # Generate configuration
        config = ScalingConfig.auto_scale(
            n_clusters=preset_cfg['n_clusters'],
            dataset_size=preset_cfg['dataset_size'],
            memory_budget_gb=preset_cfg['memory_budget_gb'],
            target_accuracy=preset_cfg['target_accuracy']
        )
        
    else:  # Custom Settings
        st.markdown("##### 🎛️ Advanced Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.number_input(
                "Initial Clusters",
                min_value=2,
                max_value=10000,
                value=50,
                help="Starting number of clusters"
            )
            
            exemplars_per_cluster = st.number_input(
                "Exemplars per Cluster",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of representative samples to keep per cluster"
            )
            
            uncertainty_buffer = st.number_input(
                "Uncertainty Buffer Size",
                min_value=0,
                max_value=10000,
                value=500,
                help="Buffer for high-uncertainty samples"
            )
        
        with col2:
            recent_buffer = st.number_input(
                "Recent Buffer Size",
                min_value=0,
                max_value=10000,
                value=300,
                help="Buffer for recent samples"
            )
            
            temperature_scaling = st.slider(
                "Temperature Scaling",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Temperature for probability calibration"
            )
            
            replay_ratio = st.slider(
                "Replay Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Proportion of old samples in each batch"
            )
        
        # Advanced options
        with st.expander("🔬 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size_scaling = st.number_input("Batch Size", min_value=8, max_value=512, value=32)
                dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
            
            with col2:
                ewc_lambda = st.number_input("EWC Lambda", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
                use_lora = st.checkbox("Use LoRA", value=False, help="Low-Rank Adaptation for efficient fine-tuning")
        
        # Create custom configuration
        config = ScalingConfig(
            exemplars_per_cluster=exemplars_per_cluster,
            uncertainty_buffer_size=uncertainty_buffer,
            recent_buffer_size=recent_buffer,
            temperature=temperature_scaling,
            replay_ratio=replay_ratio,
            batch_size=batch_size_scaling,
            dropout_rate=dropout_rate,
            ewc_lambda=ewc_lambda,
            use_lora=use_lora
        )
    
    # Display memory estimate
    total_samples = (config.exemplars_per_cluster * 50 +  # Assume 50 clusters for estimate
                    config.uncertainty_buffer_size + 
                    config.recent_buffer_size)
    memory_mb = total_samples * 768 * 4 / (1024**2)  # 768-dim embedding, 4 bytes per float
    
    st.info(f"💾 **Estimated Memory Usage:** {memory_mb:.1f} MB ({total_samples:,} samples in buffer)")
    
    return config

def train_model_ui(sequences_path, labels_path, model_type_ui, model_name,
                  epochs, batch_size, learning_rate, embedding_dim,
                  projection_dim, temperature, device, scaling_config=None, enable_scaling=False):
    """Execute training from UI with optional dynamic scaling"""
    
    # Map UI model type to internal name
    type_map = {
        "Contrastive Learning": "contrastive",
        "Transformer": "transformer",
        "Autoencoder": "autoencoder"
    }
    model_type = type_map[model_type_ui]
    
    # Output directory
    models_dir = Path("models")
    output_dir = models_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # UI containers
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Metrics display
    metrics_cols = st.columns(4)
    chart_placeholder = st.empty()
    
    # Dynamic scaling display
    if enable_scaling and scaling_config:
        st.markdown("---")
        st.markdown("#### ⚡ Dynamic Scaling Metrics")
        scaling_cols = st.columns(4)
        scaling_chart = st.empty()
    
    try:
        status_container.info("🔄 Initializing training...")
        
        # Load data
        from scripts.train_model import load_sequences, load_labels, create_model
        
        sequences = load_sequences(sequences_path)
        labels = load_labels(labels_path, sequences) if labels_path else None
        
        status_container.info(f"📊 Loaded {len(sequences)} sequences")
        
        # Create tokenizer
        tokenizer = DNATokenizer(encoding_type='kmer', kmer_size=4)  # OPTIMIZED
        
        # Create model
        config_dict = {
            'embedding': {'embedding_dim': embedding_dim},
            'training': {
                'projection_dim': projection_dim,
                'temperature': temperature
            }
        }
        
        if enable_scaling and scaling_config:
            # Add scaling config to model config
            config_dict['scaling'] = {
                'exemplars_per_cluster': scaling_config.exemplars_per_cluster,
                'uncertainty_buffer_size': scaling_config.uncertainty_buffer_size,
                'recent_buffer_size': scaling_config.recent_buffer_size,
                'temperature': scaling_config.temperature,
                'replay_ratio': scaling_config.replay_ratio,
                'use_lora': scaling_config.use_lora
            }
        
        model, _ = create_model(model_type, tokenizer.vocab_size, config_dict)
        trainer = EmbeddingTrainer(model, tokenizer, device=device)
        
        # Prepare data
        train_loader, val_loader = trainer.prepare_data(
            sequences=sequences,
            labels=labels,
            validation_split=0.2,
            batch_size=batch_size
        )
        
        # Training with or without dynamic scaling
        status_container.info(f"🚀 Training {model_type_ui} model...")
        
        if enable_scaling and scaling_config:
            # Dynamic scaling mode - use continual learning
            status_container.info("⚡ Dynamic scaling enabled - training with continual learning")
            
            history = trainer.train_with_dynamic_scaling(
                sequences=sequences,
                labels=labels,
                epochs_per_task=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_length=512,
                replay_ratio=scaling_config.replay_ratio if hasattr(scaling_config, 'replay_ratio') else 0.3,
                validation_split=0.2
            )
            
            # Update UI with final metrics
            progress_bar.progress(1.0)
            
            with metrics_cols[0]:
                st.metric("Tasks Completed", len(history['train_loss']))
            with metrics_cols[1]:
                st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
            with metrics_cols[2]:
                st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
            with metrics_cols[3]:
                st.metric("Buffer Size", f"{history['buffer_size'][-1]:,}")
        else:
            # Standard training mode
            history = {
                'train_loss': [],
                'val_loss': []
            }
            
            for epoch in range(epochs):
                # Train one epoch
                if model_type == 'autoencoder':
                    epoch_history = trainer.train_autoencoder(train_loader, val_loader, epochs=1, learning_rate=learning_rate)
                else:
                    epoch_history = trainer.train_contrastive(train_loader, val_loader, epochs=1, learning_rate=learning_rate)
                
                # Update history
                train_loss = epoch_history['train_loss'][0]
                val_loss = epoch_history['val_loss'][0]
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                
                # Update UI
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                
                # Update metrics
                with metrics_cols[0]:
                    st.metric("Epoch", f"{epoch+1}/{epochs}")
                with metrics_cols[1]:
                    st.metric("Train Loss", f"{train_loss:.4f}")
                with metrics_cols[2]:
                    st.metric("Val Loss", f"{val_loss:.4f}")
                with metrics_cols[3]:
                    delta = val_loss - history['val_loss'][-2] if len(history['val_loss']) > 1 else 0
                    st.metric("Δ Loss", f"{delta:+.4f}", delta=f"{delta:+.4f}")
                
                # Update chart
                chart_data = pd.DataFrame({
                    'Epoch': range(1, len(history['train_loss']) + 1),
                    'Train Loss': history['train_loss'],
                    'Val Loss': history['val_loss']
                })
                
                fig = px.line(
                    chart_data, 
                    x='Epoch', 
                    y=['Train Loss', 'Val Loss'],
                    title='📈 Training Progress',
                    labels={'value': 'Loss', 'variable': 'Metric'}
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Show charts for dynamic scaling
        if enable_scaling and scaling_config and 'buffer_size' in history:
            # Display scaling metrics
            with scaling_cols[0]:
                st.metric("Clusters", history['clusters'][-1])
            with scaling_cols[1]:
                st.metric("Buffer Size", f"{history['buffer_size'][-1]:,}")
            with scaling_cols[2]:
                st.metric("Memory", f"{history['memory_mb'][-1]:.1f} MB")
            with scaling_cols[3]:
                st.metric("Exemplars/Cluster", history['exemplars_per_cluster'][-1])
            
            # Scaling chart
            scaling_data = pd.DataFrame({
                'Task': range(1, len(history['clusters']) + 1),
                'Clusters': history['clusters'],
                'Buffer Size': history['buffer_size']
            })
            
            fig2 = px.line(
                scaling_data,
                x='Task',
                y=['Clusters', 'Buffer Size'],
                title='⚡ Dynamic Scaling Metrics',
                labels={'value': 'Count', 'variable': 'Metric'}
            )
            scaling_chart.plotly_chart(fig2, use_container_width=True)
            
        # Save model
        status_container.info("💾 Saving model...")
        trainer.save_model(str(output_dir / "model"), include_tokenizer=True)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'embedding_dim': embedding_dim,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dynamic_scaling_enabled': enable_scaling
        }
        
        if enable_scaling and scaling_config:
            metadata['scaling_config'] = {
                'exemplars_per_cluster': scaling_config.exemplars_per_cluster,
                'uncertainty_buffer_size': scaling_config.uncertainty_buffer_size,
                'recent_buffer_size': scaling_config.recent_buffer_size,
                'temperature': scaling_config.temperature
            }
            if 'buffer_size' in history:
                metadata['final_clusters'] = history['clusters'][-1]
                metadata['final_buffer_size'] = history['buffer_size'][-1]
                metadata['final_memory_mb'] = history['memory_mb'][-1]
                metadata['final_exemplars_per_cluster'] = history['exemplars_per_cluster'][-1]
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / "training_history.csv", index=False)
        
        status_container.success(f"✅ Training complete! Model saved to {output_dir}")
        st.balloons()
        
        # Show summary
        st.markdown("---")
        st.markdown("### 📊 Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
        with col2:
            st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
        with col3:
            total_epochs = len(history['train_loss']) if not enable_scaling else f"{len(history['train_loss'])} tasks"
            st.metric("Total Training", total_epochs)
        with col4:
            improvement = ((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100)
            st.metric("Loss Improvement", f"{improvement:.1f}%")
        
        if enable_scaling and scaling_config and 'buffer_size' in history:
            st.markdown("#### ⚡ Scaling Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Clusters", history['clusters'][-1])
            with col2:
                st.metric("Final Buffer Size", f"{history['buffer_size'][-1]:,}")
            with col3:
                st.metric("Final Memory", f"{history['memory_mb'][-1]:.1f} MB")
        
    except Exception as e:
        status_container.error(f"❌ Training failed: {str(e)}")
        st.exception(e)

def show_model_management():
    """Display model management interface with scaling info"""
    models_dir = Path("models")
    if not models_dir.exists():
        st.info("📁 No models found. Train your first model to get started!")
        return
        
    models = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not models:
        st.info("📁 No trained models found. Start training above!")
        return
        
    st.markdown("### 📚 Trained Models")
    st.markdown(f"Found **{len(models)}** trained model(s)")
    
    for model_dir in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True):
        with st.expander(f"🤖 {model_dir.name}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Load metadata if exists
                meta_path = model_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    st.markdown("#### 📊 Model Information")
                    
                    # Basic info
                    info_cols = st.columns(3)
                    with info_cols[0]:
                        st.metric("Architecture", meta.get('model_type', 'Unknown').title())
                    with info_cols[1]:
                        st.metric("Epochs", meta.get('epochs', 'N/A'))
                    with info_cols[2]:
                        st.metric("Batch Size", meta.get('batch_size', 'N/A'))
                    
                    # Loss metrics
                    loss_cols = st.columns(3)
                    with loss_cols[0]:
                        st.metric("Final Train Loss", f"{meta.get('final_train_loss', 0):.4f}")
                    with loss_cols[1]:
                        st.metric("Final Val Loss", f"{meta.get('final_val_loss', 0):.4f}")
                    with loss_cols[2]:
                        st.metric("Learning Rate", f"{meta.get('learning_rate', 0):.6f}")
                    
                    # Dynamic scaling info
                    if meta.get('dynamic_scaling_enabled'):
                        st.markdown("#### ⚡ Dynamic Scaling")
                        st.success("✓ Trained with dynamic scaling enabled")
                        
                        scaling_config = meta.get('scaling_config', {})
                        if scaling_config:
                            scale_cols = st.columns(4)
                            with scale_cols[0]:
                                st.metric("Exemplars/Cluster", scaling_config.get('exemplars_per_cluster', 'N/A'))
                            with scale_cols[1]:
                                st.metric("Uncertainty Buffer", scaling_config.get('uncertainty_buffer_size', 'N/A'))
                            with scale_cols[2]:
                                st.metric("Recent Buffer", scaling_config.get('recent_buffer_size', 'N/A'))
                            with scale_cols[3]:
                                st.metric("Replay Ratio", f"{scaling_config.get('replay_ratio', 0):.2f}")
                            
                            # Final state
                            if meta.get('final_clusters'):
                                final_cols = st.columns(3)
                                with final_cols[0]:
                                    st.metric("Final Clusters", meta.get('final_clusters'))
                                with final_cols[1]:
                                    st.metric("Final Buffer Size", f"{meta.get('final_buffer_size', 0):,}")
                                with final_cols[2]:
                                    memory_mb = meta.get('final_buffer_size', 0) * 768 * 4 / (1024**2)
                                    st.metric("Memory Used", f"{memory_mb:.1f} MB")
                    else:
                        st.info("ℹ️ Trained without dynamic scaling (standard training)")
                    
                    # Timestamp
                    st.caption(f"🕒 Trained: {meta.get('timestamp', 'Unknown')}")
                    
                    # Show training history chart if available
                    history_path = model_dir / "training_history.csv"
                    if history_path.exists():
                        st.markdown("#### 📈 Training History")
                        history_df = pd.read_csv(history_path)
                        
                        # Loss plot
                        fig = px.line(
                            history_df,
                            x='Epoch' if 'Epoch' in history_df.columns else history_df.index + 1,
                            y=['train_loss', 'val_loss'] if 'train_loss' in history_df.columns else ['Train Loss', 'Val Loss'],
                            title='Loss Curve',
                            labels={'value': 'Loss', 'variable': 'Metric'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Scaling metrics plot if available
                        if 'clusters' in history_df.columns and history_df['clusters'].notna().any():
                            fig2 = px.line(
                                history_df,
                                x=history_df.index + 1,
                                y=['clusters', 'buffer_size'],
                                title='Dynamic Scaling Evolution',
                                labels={'value': 'Count', 'variable': 'Metric', 'index': 'Epoch'}
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                    
                else:
                    st.warning("⚠️ No metadata available for this model")
                    
            with col2:
                st.markdown("#### ⚙️ Actions")
                
                # Model path
                st.text_input("Path", str(model_dir), disabled=True, label_visibility="collapsed")
                
                # Delete button
                if st.button("🗑️ Delete Model", key=f"del_{model_dir.name}", type="secondary", use_container_width=True):
                    import shutil
                    shutil.rmtree(model_dir)
                    st.success(f"Deleted {model_dir.name}")
                    st.rerun()
                
                # Export config button
                if st.button("📥 Export Config", key=f"export_{model_dir.name}", use_container_width=True):
                    meta_path = model_dir / "metadata.json"
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            config_json = json.dumps(json.load(f), indent=2)
                        st.download_button(
                            "Download JSON",
                            config_json,
                            file_name=f"{model_dir.name}_config.json",
                            mime="application/json",
                            key=f"download_{model_dir.name}"
                        )
                    else:
                        st.error("No config found")
