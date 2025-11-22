"""
Model Training Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
import tempfile
from pathlib import Path
from src.utils.config import config as app_config
from src.security import (
    FileValidator,
    InputSanitizer,
    get_rate_limiter,
    log_security_event
)

try:
    from src.models.tokenizer import DNATokenizer
    from src.models.embeddings import DNAContrastiveModel, DNATransformerEmbedder, DNAAutoencoder
    from src.models.trainer import EmbeddingTrainer
except ImportError:
    pass

try:
    from src.utils.sra_integration import create_sra_data_source_selector
except ImportError:
    create_sra_data_source_selector = None

def render():
    """Display model training page"""
    st.title("Model Training")
    st.markdown("""
    Train custom DNA embedding models using your own data. 
    Choose between **Contrastive Learning** (recommended for best performance), 
    **Autoencoders**, or standard **Transformers**.
    """)
    
    tab1, tab2 = st.tabs(["Train New Model", "Manage Models"])
    
    # --- Tab 1: Train New Model ---
    with tab1:
        st.markdown("### 1. Data Selection")
        
        # Use SRA-integrated data source selector if available
        if create_sra_data_source_selector:
            source_type, sequences_path, metadata = create_sra_data_source_selector()
            
            if metadata:
                st.info(f"Data source: {metadata.get('source', 'unknown').upper()}")
                if metadata.get('source') == 'sra':
                    st.success(f"SRA Accession: {metadata.get('accession')}")
        else:
            # Fallback to original data source selection
            data_source = st.radio("Data Source", ["Upload New File", "Select Existing Dataset"])
            
            sequences_path = None
            metadata = None
            
            if data_source == "Upload New File":
                uploaded_file = st.file_uploader("Upload FASTA File", type=['fasta', 'fa'])
                if uploaded_file:
                    # Validate filename
                    valid, error = FileValidator.validate_filename(uploaded_file.name)
                    if not valid:
                        st.error(f"❌ Invalid filename: {error}")
                        log_security_event('invalid_training_filename', {
                            'filename': uploaded_file.name,
                            'error': error
                        })
                    else:
                        # Validate file size
                        file_size = uploaded_file.size
                        valid, error = FileValidator.validate_file_size(file_size)
                        if not valid:
                            st.error(f"❌ {error}")
                            log_security_event('training_file_too_large', {
                                'filename': uploaded_file.name,
                                'size_bytes': file_size
                            })
                        else:
                            # Save to temp location
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
                                tmp.write(uploaded_file.getvalue())
                                sequences_path = tmp.name
                                log_security_event('training_file_uploaded', {
                                    'filename': uploaded_file.name,
                                    'size_bytes': file_size
                                })
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
        st.markdown("#### Labels (Optional)")
        st.markdown("Upload a CSV/TXT file with labels corresponding to sequences for supervised training.")
        labels_file = st.file_uploader("Upload Labels", type=['csv', 'txt'])
        labels_path = None
        if labels_file:
            # Validate filename
            valid, error = FileValidator.validate_filename(labels_file.name)
            if not valid:
                st.error(f"❌ Invalid labels filename: {error}")
                log_security_event('invalid_labels_filename', {
                    'filename': labels_file.name,
                    'error': error
                })
            else:
                # Validate file size
                file_size = labels_file.size
                valid, error = FileValidator.validate_file_size(file_size)
                if not valid:
                    st.error(f"❌ {error}")
                    log_security_event('labels_file_too_large', {
                        'filename': labels_file.name,
                        'size_bytes': file_size
                    })
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(labels_file.name).suffix) as tmp:
                        tmp.write(labels_file.getvalue())
                        labels_path = tmp.name
                        log_security_event('labels_file_uploaded', {
                            'filename': labels_file.name,
                            'size_bytes': file_size
                        })
        
        st.markdown("---")
        st.markdown("### 2. Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Architecture", 
                ["Contrastive Learning", "Transformer", "Autoencoder"],
                help="Contrastive Learning is recommended for embedding generation."
            )
            
            epochs = st.number_input("Epochs", min_value=1, value=50)
            batch_size = st.number_input("Batch Size", min_value=2, value=32)
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, value=1e-4, format="%.6f")
            
        with col2:
            embedding_dim = st.number_input("Embedding Dimension", min_value=32, value=256)
            
            if model_type == "Contrastive Learning":
                projection_dim = st.number_input("Projection Dimension", min_value=32, value=128)
                temperature = st.number_input("Temperature", min_value=0.01, value=0.1)
            
            device = st.selectbox("Device", ["auto", "cpu", "cuda"])
            
        model_name = st.text_input("Model Name", value=f"model_{int(time.time())}")
        
        st.markdown("---")
        
        if st.button("Start Training", type="primary"):
            if not sequences_path:
                st.error("Please select a sequence file.")
            else:
                train_model_ui(
                    sequences_path, labels_path, model_type, model_name,
                    epochs, batch_size, learning_rate, embedding_dim,
                    projection_dim if model_type == "Contrastive Learning" else None,
                    temperature if model_type == "Contrastive Learning" else None,
                    device
                )

    # --- Tab 2: Manage Models ---
    with tab2:
        show_model_management()

def train_model_ui(sequences_path, labels_path, model_type_ui, model_name,
                  epochs, batch_size, learning_rate, embedding_dim,
                  projection_dim, temperature, device):
    """Execute training from UI"""
    
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
    
    status_container = st.empty()
    progress_bar = st.progress(0)
    metrics_col1, metrics_col2 = st.columns(2)
    chart_placeholder = st.empty()
    
    try:
        status_container.info("Initializing training...")
        
        # Load data
        from scripts.train_model import load_sequences, load_labels, create_model
        
        sequences = load_sequences(sequences_path)
        labels = load_labels(labels_path, sequences) if labels_path else None
        
        # Create tokenizer
        tokenizer = DNATokenizer(encoding_type='kmer', kmer_size=6)
        
        # Create model
        config_dict = {
            'embedding': {'embedding_dim': embedding_dim},
            'training': {
                'projection_dim': projection_dim,
                'temperature': temperature
            }
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
        
        # Training loop
        status_container.info(f"Training {model_type_ui} model for {epochs} epochs...")
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Train one epoch
            if model_type == 'autoencoder':
                # Custom single epoch training logic would be needed here to update UI per epoch
                # For now, we'll use the trainer's method but it runs all epochs
                # To support UI updates, we'd need to modify trainer or implement loop here
                # Let's implement a simple loop here using trainer's internal methods if possible
                # Or just run the whole thing and show final result (less ideal)
                
                # Better approach: Use the trainer's methods but for 1 epoch at a time
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
            
            with metrics_col1:
                st.metric("Epoch", f"{epoch+1}/{epochs}")
            with metrics_col2:
                st.metric("Train Loss", f"{train_loss:.4f}", delta=None)
                
            # Update chart
            chart_data = pd.DataFrame({
                'Epoch': range(1, len(history['train_loss']) + 1),
                'Train Loss': history['train_loss'],
                'Val Loss': history['val_loss']
            })
            
            fig = px.line(chart_data, x='Epoch', y=['Train Loss', 'Val Loss'], title='Training Progress')
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
        # Save model
        status_container.info("Saving model...")
        trainer.save_model(str(output_dir / "model"), include_tokenizer=True)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'epochs': epochs,
            'final_loss': history['train_loss'][-1],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
            
        status_container.success(f"Training complete! Model saved to {output_dir}")
        st.balloons()
        
    except Exception as e:
        status_container.error(f"Training failed: {str(e)}")
        st.exception(e)

def show_model_management():
    """Display model management interface"""
    models_dir = Path("models")
    if not models_dir.exists():
        st.info("No models found.")
        return
        
    models = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not models:
        st.info("No trained models found.")
        return
        
    st.markdown("### Trained Models")
    
    for model_dir in models:
        with st.expander(f"{model_dir.name}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Load metadata if exists
                meta_path = model_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    st.json(meta)
                else:
                    st.text("No metadata available")
                    
            with col2:
                if st.button("Delete", key=f"del_{model_dir.name}"):
                    import shutil
                    shutil.rmtree(model_dir)
                    st.rerun()
