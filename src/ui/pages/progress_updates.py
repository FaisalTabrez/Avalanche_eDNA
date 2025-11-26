"""
Pipeline Progress Updates Page
Real-time monitoring of pipeline execution with model loading, batch progress, and terminal output
"""
import streamlit as st
import pandas as pd
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
import sys
import io
import subprocess
import re

def render():
    """Display the pipeline progress monitoring page"""
    
    st.title("Pipeline Progress Monitor")
    st.markdown("""
    Monitor real-time progress of eDNA pipeline execution including model loading, 
    batch processing updates, and complete terminal output.
    """)
    
    # Initialize session state
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'progress_logs' not in st.session_state:
        st.session_state.progress_logs = []
    if 'pipeline_stats' not in st.session_state:
        st.session_state.pipeline_stats = {
            'total_sequences': 0,
            'embedded_sequences': 0,
            'current_step': 'Not Started',
            'model_loaded': False,
            'start_time': None,
            'end_time': None
        }
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Pipeline Configuration")
        
        # Dataset selection
        datasets_dir = Path("data/datasets")
        if datasets_dir.exists():
            dataset_files = list(datasets_dir.glob("*.fasta")) + list(datasets_dir.glob("*.fastq"))
            if dataset_files:
                dataset_names = [f.name for f in dataset_files]
                selected_dataset = st.selectbox("Select Dataset", dataset_names)
                dataset_path = datasets_dir / selected_dataset
            else:
                st.warning("No datasets found in data/datasets/")
                dataset_path = None
        else:
            st.warning("Datasets directory not found")
            dataset_path = None
        
        st.divider()
        
        # Pipeline steps
        st.subheader("Pipeline Steps")
        run_preprocessing = st.checkbox("Preprocessing", value=True)
        run_embedding = st.checkbox("Embedding Generation", value=True)
        run_clustering = st.checkbox("Clustering", value=True)
        run_taxonomy = st.checkbox("Taxonomy Assignment", value=True)
        run_novelty = st.checkbox("Novelty Detection", value=True)
        run_visualization = st.checkbox("Visualization", value=True)
        
        st.divider()
        
        # Advanced options
        with st.expander("Advanced Options"):
            batch_size = st.number_input("Embedding Batch Size", min_value=1, max_value=64, value=8)
            use_gpu = st.checkbox("Use GPU (if available)", value=True)
            max_sequences = st.number_input("Max Sequences (0 = all)", min_value=0, value=0)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Progress")
        
        # Progress metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Current Step", st.session_state.pipeline_stats['current_step'])
        with metric_cols[1]:
            if st.session_state.pipeline_stats['total_sequences'] > 0:
                progress_pct = (st.session_state.pipeline_stats['embedded_sequences'] / 
                               st.session_state.pipeline_stats['total_sequences'] * 100)
                st.metric("Embedding Progress", f"{progress_pct:.1f}%")
            else:
                st.metric("Embedding Progress", "0%")
        with metric_cols[2]:
            if st.session_state.pipeline_stats['start_time']:
                elapsed = (datetime.now() - st.session_state.pipeline_stats['start_time']).total_seconds()
                st.metric("Elapsed Time", f"{elapsed:.1f}s")
            else:
                st.metric("Elapsed Time", "0s")
        with metric_cols[3]:
            model_status = "✓ Loaded" if st.session_state.pipeline_stats['model_loaded'] else "⏳ Pending"
            st.metric("Model Status", model_status)
        
        # Progress bar
        if st.session_state.pipeline_stats['total_sequences'] > 0:
            progress = st.session_state.pipeline_stats['embedded_sequences'] / st.session_state.pipeline_stats['total_sequences']
            st.progress(progress)
        else:
            st.progress(0)
        
        # Control buttons
        button_cols = st.columns([1, 1, 2])
        with button_cols[0]:
            start_button = st.button(
                "▶️ Start Pipeline", 
                type="primary",
                disabled=st.session_state.pipeline_running or dataset_path is None,
                use_container_width=True
            )
        with button_cols[1]:
            stop_button = st.button(
                "⏹️ Stop Pipeline",
                disabled=not st.session_state.pipeline_running,
                use_container_width=True
            )
        
        if start_button and dataset_path:
            st.session_state.pipeline_running = True
            st.session_state.progress_logs = []
            st.session_state.pipeline_stats['start_time'] = datetime.now()
            st.session_state.pipeline_stats['current_step'] = 'Initializing...'
            st.rerun()
        
        if stop_button:
            st.session_state.pipeline_running = False
            st.session_state.pipeline_stats['current_step'] = 'Stopped'
            st.warning("Pipeline stopped by user")
            st.rerun()
    
    with col2:
        st.subheader("Pipeline Steps")
        
        steps = [
            ("1. Preprocessing", run_preprocessing, "quality_control"),
            ("2. Embedding", run_embedding, "model"),
            ("3. Clustering", run_clustering, "groups"),
            ("4. Taxonomy", run_taxonomy, "classification"),
            ("5. Novelty", run_novelty, "new_label"),
            ("6. Visualization", run_visualization, "bar_chart")
        ]
        
        for step_name, enabled, icon in steps:
            if enabled:
                status_icon = "✓" if st.session_state.pipeline_stats['current_step'].startswith(step_name.split('.')[0]) else "⏳"
                st.markdown(f"{status_icon} {step_name}")
            else:
                st.markdown(f"⊘ {step_name} (skipped)")
    
    # Terminal output section
    st.divider()
    st.subheader("📋 Terminal Output")
    
    # Output display tabs
    tab1, tab2 = st.tabs(["Live Output", "Filtered Progress"])
    
    with tab1:
        output_container = st.container()
        with output_container:
            if st.session_state.progress_logs:
                log_text = "\n".join(st.session_state.progress_logs[-100:])  # Show last 100 lines
                st.code(log_text, language="log")
            else:
                st.info("Pipeline output will appear here when running...")
    
    with tab2:
        if st.session_state.progress_logs:
            # Filter for important progress messages
            progress_logs = [log for log in st.session_state.progress_logs if any(
                keyword in log.lower() for keyword in [
                    'loading', 'embedded', 'step', 'complete', 'processing', 
                    'model', 'batch', 'progress', 'pca', 'cluster', 'taxonomy'
                ]
            )]
            if progress_logs:
                st.code("\n".join(progress_logs[-50:]), language="log")
            else:
                st.info("No progress messages yet...")
        else:
            st.info("Pipeline output will appear here when running...")
    
    # Run pipeline in background
    if st.session_state.pipeline_running and dataset_path:
        run_pipeline_background(
            dataset_path,
            run_preprocessing=run_preprocessing,
            run_embedding=run_embedding,
            run_clustering=run_clustering,
            run_taxonomy=run_taxonomy,
            run_novelty=run_novelty,
            run_visualization=run_visualization,
            batch_size=batch_size,
            use_gpu=use_gpu,
            max_sequences=max_sequences
        )
    
    # Auto-refresh while running
    if st.session_state.pipeline_running:
        time.sleep(2)
        st.rerun()


def run_pipeline_background(dataset_path, **kwargs):
    """Run the pipeline and capture output"""
    
    try:
        # Build command
        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            str(dataset_path),
            "--output-dir", f"analysis_outputs/runs/{dataset_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ]
        
        # Add step flags
        if not kwargs.get('run_preprocessing', True):
            cmd.append("--skip-preprocessing")
        if not kwargs.get('run_embedding', True):
            cmd.append("--skip-embedding")
        if not kwargs.get('run_clustering', True):
            cmd.append("--skip-clustering")
        if not kwargs.get('run_taxonomy', True):
            cmd.append("--skip-taxonomy")
        if not kwargs.get('run_novelty', True):
            cmd.append("--skip-novelty")
        if not kwargs.get('run_visualization', True):
            cmd.append("--skip-visualization")
        
        # Add batch size
        if kwargs.get('batch_size'):
            cmd.extend(["--batch-size", str(kwargs['batch_size'])])
        
        # Add max sequences
        if kwargs.get('max_sequences', 0) > 0:
            cmd.extend(["--max-sequences", str(kwargs['max_sequences'])])
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                st.session_state.progress_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                
                # Parse progress information
                parse_progress_line(line)
        
        # Wait for completion
        process.wait()
        
        if process.returncode == 0:
            st.session_state.pipeline_stats['current_step'] = '✓ Complete'
            st.session_state.progress_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Pipeline completed successfully!")
        else:
            st.session_state.pipeline_stats['current_step'] = '✗ Failed'
            st.session_state.progress_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Pipeline failed with code {process.returncode}")
        
        st.session_state.pipeline_running = False
        st.session_state.pipeline_stats['end_time'] = datetime.now()
        
    except Exception as e:
        st.session_state.progress_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {str(e)}")
        st.session_state.pipeline_running = False
        st.session_state.pipeline_stats['current_step'] = '✗ Error'


def parse_progress_line(line: str):
    """Parse log line and update statistics"""
    
    # Detect model loading
    if "Loading" in line and "model" in line.lower():
        st.session_state.pipeline_stats['model_loaded'] = True
        st.session_state.pipeline_stats['current_step'] = 'Loading Model...'
    
    # Detect preprocessing
    if "Step 1:" in line or "Preprocessing" in line:
        st.session_state.pipeline_stats['current_step'] = '1. Preprocessing'
    
    # Detect embedding start
    if "Step 2:" in line or "Generating sequence embeddings" in line:
        st.session_state.pipeline_stats['current_step'] = '2. Embedding Generation'
    
    # Parse embedding progress: "Embedded 24/100 sequences"
    match = re.search(r'Embedded\s+(\d+)/(\d+)\s+sequences', line)
    if match:
        st.session_state.pipeline_stats['embedded_sequences'] = int(match.group(1))
        st.session_state.pipeline_stats['total_sequences'] = int(match.group(2))
    
    # Detect clustering
    if "Step 3:" in line or "Clustering" in line:
        st.session_state.pipeline_stats['current_step'] = '3. Clustering'
    
    # Detect taxonomy
    if "Step 4:" in line or "Taxonomy" in line:
        st.session_state.pipeline_stats['current_step'] = '4. Taxonomy Assignment'
    
    # Detect novelty
    if "Step 5:" in line or "Novelty" in line:
        st.session_state.pipeline_stats['current_step'] = '5. Novelty Detection'
    
    # Detect visualization
    if "Step 6:" in line or "Visualization" in line:
        st.session_state.pipeline_stats['current_step'] = '6. Visualization'
    
    # Detect PCA
    if "Applying PCA" in line:
        match = re.search(r'shape\s+\((\d+),\s*(\d+)\)', line)
        if match:
            st.session_state.pipeline_stats['total_sequences'] = int(match.group(1))


if __name__ == "__main__":
    render()
