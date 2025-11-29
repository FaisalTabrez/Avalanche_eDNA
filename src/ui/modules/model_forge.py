import streamlit as st
import time
import tempfile
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.ui.core.state_manager import StateManager
from src.ui.core.task_manager import TaskManager

try:
    from src.models.tokenizer import DNATokenizer
    from src.models.embeddings import DNAContrastiveModel
    from src.models.trainer import EmbeddingTrainer
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

try:
    from src.models.dynamic_hybrid_buffer import ScalingConfig
    SCALING_AVAILABLE = True
except ImportError:
    SCALING_AVAILABLE = False

def render():
    st.title("⚡ Model Forge")
    st.markdown("Train, fine-tune, and scale your eDNA models with **DNABERT-2 on CPU**.")

    tab1, tab2 = st.tabs(["🏋️ Training Arena", "⚖️ Dynamic Scaling"])

    with tab1:
        render_training_tab()
    
    with tab2:
        render_scaling_tab()

def render_training_tab():
    """DNABERT-2 CPU Training Interface"""
    
    if not MODELS_AVAILABLE:
        st.error("❌ Model training components not available. Please check installation.")
        return
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Configuration")
        
        # Data Source
        st.markdown("#### 📁 Data")
        data_source = st.radio("Source", ["Upload File", "Select Dataset"], label_visibility="collapsed")
        
        sequences_path = None
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("FASTA File", type=['fasta', 'fa'])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    sequences_path = tmp.name
        else:
            datasets_dir = Path("data/datasets")
            if datasets_dir.exists():
                files = list(datasets_dir.glob("*.fasta")) + list(datasets_dir.glob("*.fa"))
                if files:
                    selected = st.selectbox("Dataset", files, format_func=lambda x: x.name)
                    sequences_path = str(selected)
                else:
                    st.warning("No datasets found.")
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown("#### 🧠 Model")
        model_name = st.text_input("Name", value=f"dnabert2_cpu_{int(time.time())}")
        
        st.markdown("#### 🔧 Training")
        epochs = st.slider("Epochs", 1, 100, 10)
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        learning_rate = st.number_input("Learning Rate", value=1e-4, format="%.6f")
        
        st.markdown("---")
        
        # Dynamic Scaling
        enable_scaling = st.checkbox("⚡ Enable Dynamic Scaling", value=False)
        
        if enable_scaling and SCALING_AVAILABLE:
            memory_budget = st.number_input("Memory Budget (GB)", value=2.0, min_value=0.5, max_value=32.0)
            target_accuracy = st.slider("Target Accuracy", 0.5, 0.99, 0.85, step=0.05)
        
        st.markdown("---")
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not sequences_path:
                st.error("❌ Please select a data source.")
            else:
                # Prepare scaling config if enabled
                scaling_config = None
                if enable_scaling and SCALING_AVAILABLE:
                    scaling_config = ScalingConfig.auto_scale(
                        n_clusters=5,  # Will be updated during training
                        dataset_size=1000,  # Estimated
                        memory_budget_gb=memory_budget,
                        target_accuracy=target_accuracy
                    )
                
                # Submit training task
                tm = TaskManager()
                task_id = tm.submit_task(
                    name=f"Train-{model_name}",
                    target_func=train_dnabert2_cpu,
                    kwargs={
                        "sequences_path": sequences_path,
                        "model_name": model_name,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "scaling_config": scaling_config,
                        "enable_scaling": enable_scaling
                    }
                )
                StateManager.set("current_training_id", task_id)
                st.toast(f"✅ Training started! Task ID: {task_id}")
                st.rerun()

    with col2:
        st.subheader("📊 Live Metrics")
        
        # Check for active training task
        task_id = StateManager.get("current_training_id")
        if task_id:
            tm = TaskManager()
            task = tm.get_task_status(task_id)
            
            if task and task['status'] == 'running':
                st.info(f"🔄 Training: {task['name']}")
                progress = st.progress(0)
                
                # In production, read from training metrics file
                st.caption("Monitoring training progress...")
                render_placeholder_chart()
                
            elif task and task['status'] == 'completed':
                st.success("✅ Training completed!")
                
                # Display results
                if task.get('result'):
                    st.json(task['result'])
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Status", "Complete")
                with col_b:
                    st.metric("Device", "CPU")
                with col_c:
                    st.metric("Model Type", "DNABERT-2")
                
            elif task and task['status'] == 'failed':
                st.error(f"❌ Training failed: {task.get('error', 'Unknown error')}")
            else:
                st.info("No active training session.")
                render_placeholder_chart()
        else:
            st.info("No training jobs yet. Configure and start training.")
            render_placeholder_chart()

def render_placeholder_chart():
    """Empty chart placeholder"""
    df = pd.DataFrame({"Epoch": [], "Loss": []})
    fig = px.line(df, x="Epoch", y="Loss", title="Training Loss")
    fig.update_layout(xaxis_range=[0, 10], yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def render_scaling_tab():
    """Dynamic Scaling Configuration"""
    st.header("⚖️ Dynamic Scaling Configuration")
    st.markdown("Configure adaptive memory and cluster scaling for continual learning.")
    
    if not SCALING_AVAILABLE:
        st.warning("⚠️ Dynamic scaling not available. Please check installation.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Configuration Presets")
        
        preset = st.selectbox(
            "Select Preset",
            ["Small (5-20 clusters)", "Medium (20-100 clusters)", "Large (100-500 clusters)", "Custom"]
        )
        
        if preset == "Small (5-20 clusters)":
            memory_budget = 1.0
            target_accuracy = 0.90
            exemplars = 50
        elif preset == "Medium (20-100 clusters)":
            memory_budget = 2.0
            target_accuracy = 0.85
            exemplars = 30
        elif preset == "Large (100-500 clusters)":
            memory_budget = 4.0
            target_accuracy = 0.80
            exemplars = 20
        else:
            memory_budget = st.number_input("Memory Budget (GB)", 0.5, 32.0, 2.0)
            target_accuracy = st.slider("Target Accuracy", 0.5, 0.99, 0.85)
            exemplars = st.number_input("Exemplars per Cluster", 10, 100, 25)
        
        st.markdown("---")
        st.markdown("### Buffers")
        uncertainty_buffer = st.number_input("Uncertainty Buffer Size", 100, 10000, 1000)
        recent_buffer = st.number_input("Recent Buffer Size", 100, 10000, 1000)
        
    with col2:
        st.subheader("📊 Estimated Configuration")
        
        # Generate example scaling config
        try:
            config = ScalingConfig.auto_scale(
                n_clusters=50,  # Example
                dataset_size=5000,  # Example
                memory_budget_gb=memory_budget,
                target_accuracy=target_accuracy
            )
            
            st.metric("Exemplars per Cluster", config.exemplars_per_cluster)
            st.metric("Uncertainty Buffer", f"{config.uncertainty_buffer_size:,}")
            st.metric("Recent Buffer", f"{config.recent_buffer_size:,}")
            st.metric("Replay Ratio", f"{config.replay_ratio:.2f}")
            
            st.markdown("---")
            st.markdown("### Advanced Settings")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Temperature", f"{config.temperature:.2f}")
                st.metric("Use LoRA", "✓" if config.use_lora else "✗")
            with col_b:
                st.metric("EWC Lambda", f"{config.ewc_lambda:.1f}")
                st.metric("Hierarchical", "✓" if config.use_hierarchical else "✗")
                
        except Exception as e:
            st.error(f"Configuration error: {e}")


def train_dnabert2_cpu(sequences_path, model_name, epochs, batch_size, 
                       learning_rate, scaling_config=None, enable_scaling=False):
    """
    Training function for DNABERT-2 on CPU
    This runs in a background worker thread
    """
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load sequences
        from scripts.train_model import load_sequences
        sequences = load_sequences(sequences_path)
        
        logger.info(f"Loaded {len(sequences)} sequences")
        
        # Create tokenizer
        tokenizer = DNATokenizer(encoding_type='kmer', kmer_size=4)  # OPTIMIZED
        
        # Create model config
        config_dict = {
            'embedding': {'embedding_dim': 256},
            'training': {
                'projection_dim': 128,
                'temperature': 0.1
            }
        }
        
        # Create contrastive model
        model = DNAContrastiveModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=256,
            projection_dim=64  # OPTIMIZED
        )
        
        # Initialize trainer
        trainer = EmbeddingTrainer(
            model=model,
            tokenizer=tokenizer,
            device='cpu',
            scaling_config=scaling_config if enable_scaling else None
        )
        
        # Train
        if enable_scaling and scaling_config:
            logger.info("Training with dynamic scaling enabled")
            history = trainer.train_with_dynamic_scaling(
                sequences=sequences,
                labels=None,
                epochs_per_task=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                validation_split=0.2
            )
        else:
            logger.info("Training without dynamic scaling")
            train_loader, val_loader = trainer.prepare_data(
                sequences=sequences,
                labels=None,
                validation_split=0.2,
                batch_size=batch_size
            )
            
            history = trainer.train_contrastive(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate
            )
        
        # Save model
        output_dir = Path("models") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "model.pt"
        trainer.save_model(str(model_path))
        
        logger.info(f"Model saved to {model_path}")
        
        # Return results
        return {
            "status": "success",
            "model_path": str(model_path),
            "final_train_loss": float(history['train_loss'][-1]) if history['train_loss'] else None,
            "final_val_loss": float(history['val_loss'][-1]) if history['val_loss'] else None,
            "epochs_completed": len(history['train_loss']),
            "scaling_enabled": enable_scaling
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
