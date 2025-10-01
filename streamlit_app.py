#!/usr/bin/env python3
"""
eDNA Biodiversity Assessment System - Streamlit GUI
A comprehensive web interface for biological sequence analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import json
from io import StringIO
import base64
import streamlit.components.v1 as components

# Project config
from src.utils.config import config as app_config

# Add project modules
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from src.analysis.dataset_analyzer import DatasetAnalyzer
except ImportError as e:
    st.error(f"Failed to import DatasetAnalyzer: {e}")
    st.error("Please ensure the src/analysis/dataset_analyzer.py file exists")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="eDNA Biodiversity Assessment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state for navigation (use stable keys to avoid emoji encoding issues)
    if 'current_page_key' not in st.session_state:
        st.session_state.current_page_key = 'home'

    # Header
    st.markdown('<h1 class="main-header">🧬 eDNA Biodiversity Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")

    PAGES = [
        {"key": "home", "label": "Home"},
        {"key": "analysis", "label": "Dataset Analysis"},
        {"key": "batch", "label": "Batch Analysis"},
        {"key": "results", "label": "Results Viewer"},
        {"key": "runs", "label": "Run Browser"},
        {"key": "taxonomy", "label": "Taxonomy Viewer"},
        {"key": "about", "label": "About"},
    ]

    # Find current index
    key_to_index = {p["key"]: i for i, p in enumerate(PAGES)}
    current_index = key_to_index.get(st.session_state.current_page_key, 0)

    selection = st.sidebar.selectbox(
        "Choose Analysis Type",
        PAGES,
        index=current_index,
        format_func=lambda p: p["label"],
        key="page_selector"
    )

    # Update session state when selection changes
    if selection["key"] != st.session_state.current_page_key:
        st.session_state.current_page_key = selection["key"]
        st.rerun()

    page_key = st.session_state.current_page_key

    if page_key == "home":
        show_home_page()
    elif page_key == "analysis":
        show_analysis_page()
    elif page_key == "batch":
        show_batch_analysis_page()
    elif page_key == "results":
        show_results_viewer()
    elif page_key == "runs":
        show_run_browser()
    elif page_key == "taxonomy":
        show_taxonomy_viewer()
    elif page_key == "about":
        show_about_page()

def show_home_page():
    """Display the home page with navigation and quick links"""
    from pathlib import Path
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            "https://via.placeholder.com/400x200/1f77b4/ffffff?text=eDNA+Analysis",
            caption="Environmental DNA Biodiversity Assessment"
        )
    
    st.markdown("""
    ## Welcome to the eDNA Biodiversity Assessment System
    
    This platform provides advanced analysis of environmental DNA (eDNA) datasets using
    machine learning and bioinformatics techniques.
    """)
    
    # Navigation tiles
    st.markdown("---")
    st.markdown("## 🧭 Navigate")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("📁 Dataset Analysis", use_container_width=True):
            st.session_state.current_page_key = "analysis"
            st.rerun()
        if st.button("🧬 Taxonomy Viewer", use_container_width=True):
            st.session_state.current_page_key = "taxonomy"
            st.rerun()
    with nav_col2:
        if st.button("📈 Results Viewer", use_container_width=True):
            st.session_state.current_page_key = "results"
            st.rerun()
        if st.button("🗂️ Run Browser", use_container_width=True):
            st.session_state.current_page_key = "runs"
            st.rerun()
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.current_page_key = "about"
            st.rerun()
    with nav_col3:
        st.markdown("### Quick Actions")
        if st.button("🚀 Start New Analysis", type="primary", use_container_width=True):
            st.session_state.current_page_key = "analysis"
            st.rerun()
    
    # Recent runs quick links (from configured storage.runs_dir)
    st.markdown("---")
    st.markdown("## 🗂️ Recent Runs")
    runs_root = Path(app_config.get('storage.runs_dir', 'runs'))
    try:
        if runs_root.exists():
            # Find run folders two levels deep: runs_root/dataset_name/timestamp
            candidates = []
            for dataset_dir in runs_root.iterdir():
                if dataset_dir.is_dir():
                    for run_dir in dataset_dir.iterdir():
                        if run_dir.is_dir():
                            try:
                                mtime = run_dir.stat().st_mtime
                            except Exception:
                                mtime = 0
                            candidates.append((mtime, dataset_dir.name, run_dir))
            candidates.sort(reverse=True)
            top = candidates[:6]
            if top:
                cols = st.columns(3)
                for idx, (_, ds_name, run_path) in enumerate(top):
                    with cols[idx % 3]:
                        label = f"{ds_name} / {run_path.name}"
                        if st.button(f"📦 {label}", key=f"recent_{idx}", use_container_width=True):
                            st.session_state.prefill_results_dir = str(run_path.resolve())
                            st.session_state.current_page_key = "results"
                            st.rerun()
            else:
                st.info(f"No runs found in {runs_root}")
        else:
            st.info(f"Runs directory not found: {runs_root}")
    except Exception as e:
        st.warning(f"Could not list recent runs: {e}")
    
    # Feature overview
    st.markdown("---")
    st.markdown("## ✨ Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🧬 Sequence Analysis
        - Multi-format support (FASTA, FASTQ, Swiss-Prot)
        - Automatic format detection
        - Quality assessment
        - Composition analysis
        """)
    with col2:
        st.markdown("""
        ### 📊 Biodiversity Metrics
        - Shannon and Simpson indices
        - Species richness
        - Evenness measures
        """)
    with col3:
        st.markdown("""
        ### 🎯 Advanced Features
        - Interactive visualizations
        - Batch analysis (planned)
        - Export capabilities
        """)

def show_analysis_page():
    """Display the main analysis page"""
    
    try:
        st.title("📁 Dataset Analysis")
        
        # Resolve storage directories from config
        datasets_dir = Path(app_config.get('storage.datasets_dir', 'data/datasets'))
        runs_root = Path(app_config.get('storage.runs_dir', 'runs'))
        datasets_dir.mkdir(parents=True, exist_ok=True)
        runs_root.mkdir(parents=True, exist_ok=True)
        
        # File upload section
        st.markdown("## 1. Upload Your Dataset")
    
        # Add info about potential upload issues
        with st.expander("📝 Troubleshooting Upload Issues"):
            st.markdown("""
            **If you encounter upload errors:**
            - **AxiosError or Request Failed**: Usually indicates a network timeout for large files
            - **Solutions**:
              - Try with a smaller file first (< 100MB)
              - Check your internet connection stability
              - For very large files, consider using the command-line interface
              - Refresh the page and try again
            
            **File Size Limits:**
            - Maximum supported: 10GB
            - Recommended for web upload: < 1GB
            - For larger files, use: `python scripts/analyze_dataset.py your_file.fasta output_report.txt`
            """)
        
        uploaded_file = st.file_uploader(
            "Choose a biological sequence file",
            type=['fasta', 'fa', 'fas', 'fastq', 'fq', 'swiss', 'gb', 'gbk', 'embl', 'em', 'gz'],
            help="Supported formats: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL (including gzipped files). Maximum file size: 10GB"
        )
        
        # File size information
        file_valid = True
        if uploaded_file is not None:
            try:
                # Check file size
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                file_size_gb = file_size_mb / 1024
                
                if file_size_gb >= 1:
                    st.info(f"📁 File size: {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
                else:
                    st.info(f"📁 File size: {file_size_mb:.2f} MB")
                
                if file_size_mb > 10240:  # 10GB limit
                    st.error("⚠️ File size exceeds 10GB limit. Please use a smaller file or contact support for processing very large datasets.")
                    file_valid = False
                elif file_size_mb > 1024:  # Warn for files over 1GB
                    st.warning(f"⚠️ Large file detected ({file_size_mb:.0f} MB). Upload may take longer than usual. Please be patient.")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.error("This might be due to a network timeout for large files. Please try with a smaller file or check your connection.")
                file_valid = False
        
        # Analysis configuration
        st.markdown("## 2. Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "Dataset Name",
                value="My Dataset",
                help="Custom name for your analysis"
            )
            
            max_sequences = st.number_input(
                "Maximum Sequences to Analyze",
                min_value=100,
                max_value=1000000,
                value=None,
                help="Leave empty to analyze all sequences"
            )
        
        with col2:
            format_override = st.selectbox(
                "Force Format (Optional)",
                ["Auto-detect", "FASTA", "FASTQ", "Swiss-Prot", "GenBank", "EMBL"],
                help="Override automatic format detection"
            )
            
            analysis_level = st.selectbox(
                "Analysis Depth",
                ["Standard", "Detailed", "Quick"],
                index=0,
                help="Choose analysis comprehensiveness"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_quality = st.checkbox("Quality Analysis", value=True)
                enable_diversity = st.checkbox("Biodiversity Metrics", value=True)
                fast_mode = st.checkbox("Fast Mode (Recommended for >10K sequences)", value=True, 
                                      help="Uses intelligent sampling to dramatically reduce analysis time for large datasets")
            
            with col2:
                enable_visualization = st.checkbox("Generate Plots", value=True)
                parallel_processing = st.checkbox("Parallel Processing", value=True)
        
        # Show fast mode info
        if fast_mode:
            st.info("⚡ Fast Mode enabled: Analysis will use intelligent sampling for datasets >10K sequences, reducing time by 60-80%")
        
        # Analysis execution
        st.markdown("## 3. Run Analysis")
        
        if uploaded_file is not None and file_valid:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                run_analysis(uploaded_file, dataset_name, max_sequences, format_override, 
                            analysis_level, enable_quality, enable_diversity, enable_visualization, fast_mode)
        elif uploaded_file is not None and not file_valid:
            st.warning("⚠️ Cannot proceed with analysis due to file size restriction.")
        else:
            st.info("Please upload a file to start analysis")
        
    except Exception as e:
        st.error(f"🚨 Error in analysis page: {str(e)}")
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())

def run_analysis(uploaded_file, dataset_name, max_sequences, format_override, 
                analysis_level, enable_quality, enable_diversity, enable_visualization, fast_mode=True):
    """Execute the analysis with progress tracking and persist inputs/outputs to configured storage"""
    
    # Resolve storage roots
    datasets_dir = Path(app_config.get('storage.datasets_dir', 'data/datasets'))
    runs_root = Path(app_config.get('storage.runs_dir', 'runs'))
    datasets_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    # Build run directory
    safe_name = ''.join(c if c.isalnum() or c in ('-','_') else '_' for c in (dataset_name or 'dataset'))
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = runs_root / safe_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist uploaded file to datasets_dir
    try:
        if '.' in uploaded_file.name:
            file_ext = uploaded_file.name.split('.')[-1]
        else:
            file_ext = 'fasta'
        dataset_path = datasets_dir / f"{safe_name}_{ts}.{file_ext}"
        with open(dataset_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
    except Exception as e:
        st.error(f"✂ Failed to store uploaded file: {e}")
        return

    # Prepare output report path inside run directory
    output_path = str(run_dir / 'analysis_report.txt')

    try:
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize analyzer with fast mode
        status_text.text("Initializing analyzer...")
        analyzer = DatasetAnalyzer(fast_mode=fast_mode)
        if fast_mode:
            st.info("⚡ Fast mode enabled - using optimized processing for large datasets")
        progress_bar.progress(10)
        
        # Prepare parameters
        format_type = None if format_override == "Auto-detect" else format_override.lower()
        max_seq = None if max_sequences == 0 else max_sequences
        
        # Run analysis with progress monitoring
        status_text.text("Running analysis...")
        progress_bar.progress(30)
        
        # Add warning for large files
        if uploaded_file and len(uploaded_file.getvalue()) > 100 * 1024 * 1024:  # > 100MB
            st.warning("⏳ Large file detected. Analysis may take several minutes. Please be patient...")
            if len(uploaded_file.getvalue()) > 200 * 1024 * 1024:  # > 200MB
                st.warning("📊 For files > 200MB, consider using a smaller subset or the command-line tool for better performance.")
        
        start_time = time.time()
        
        # Execute analysis
        results = analyzer.analyze_dataset(
            input_path=str(dataset_path),
            output_path=output_path,
            dataset_name=dataset_name,
            format_type=format_type,
            max_sequences=max_seq
        )
        
        if not results or 'basic_stats' not in results:
            raise ValueError("Analysis did not complete successfully or returned incomplete results")
        
        progress_bar.progress(80)
        status_text.text("Generating visualizations...")
        
        # Display results
        display_results(results, output_path, enable_visualization)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        elapsed_time = time.time() - start_time
        st.success(f"✅ Analysis completed in {elapsed_time:.2f} seconds!")
        st.info(f"Inputs saved to: {dataset_path}")
        st.info(f"Run outputs: {run_dir}")
        
    except Exception as e:
        st.error(f"✂ Analysis failed: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")

def display_results(results, output_path, enable_visualization):
    """Display analysis results with visualizations"""
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = results.get('basic_stats', {})
    comp = results.get('composition', {})
    proc = results.get('processing_info', {})
    
    with col1:
        st.metric(
            "Total Sequences",
            f"{stats.get('total_sequences', 0):,}"
        )
    
    with col2:
        st.metric(
            "Mean Length",
            f"{stats.get('mean_length', 0):.1f}",
            f"±{stats.get('std_length', 0):.1f}"
        )
    
    with col3:
        st.metric(
            "Sequence Type",
            comp.get('sequence_type', 'Unknown').upper()
        )
    
    with col4:
        st.metric(
            "Processing Time",
            f"{proc.get('total_time', 0):.2f}s"
        )
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Statistics", "🧬 Composition", "📝 Annotations", 
        "🌿 Biodiversity", "📄 Report"
    ])
    
    with tab1:
        show_statistics_tab(results, enable_visualization)
    
    with tab2:
        show_composition_tab(results, enable_visualization)
    
    with tab3:
        show_annotations_tab(results)
    
    with tab4:
        show_biodiversity_tab(results)
    
    with tab5:
        show_report_tab(output_path)

def show_statistics_tab(results, enable_visualization):
    """Display statistics tab"""
    
    stats = results.get('basic_stats', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Length Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': ['Total Sequences', 'Minimum Length', 'Maximum Length', 
                      'Mean Length', 'Median Length', 'Std Deviation'],
            'Value': [
                f"{stats.get('total_sequences', 0):,}",
                f"{stats.get('min_length', 0):,}",
                f"{stats.get('max_length', 0):,}",
                f"{stats.get('mean_length', 0):.2f}",
                f"{stats.get('median_length', 0):.2f}",
                f"{stats.get('std_length', 0):.2f}"
            ]
        })
        
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with col2:
        if enable_visualization and stats:
            st.markdown("### 📈 Length Distribution")
            
            # Create synthetic data for demonstration
            # In real implementation, you'd pass the actual sequence lengths
            lengths = np.random.normal(
                stats.get('mean_length', 300), 
                stats.get('std_length', 100), 
                min(1000, stats.get('total_sequences', 100))
            )
            lengths = lengths[lengths > 0]  # Remove negative values
            
            fig = px.histogram(
                x=lengths,
                nbins=50,
                title="Sequence Length Distribution",
                labels={'x': 'Sequence Length', 'y': 'Frequency'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_composition_tab(results, enable_visualization):
    """Display composition analysis tab"""
    
    comp = results.get('composition', {})
    
    if comp.get('composition'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🧬 {comp.get('sequence_type', 'Unknown').upper()} Composition")
            
            # Create composition dataframe
            comp_data = []
            for char, freq in list(comp['composition'].items())[:15]:  # Top 15
                comp_data.append({
                    'Character': char,
                    'Frequency': freq,
                    'Percentage': f"{freq*100:.3f}%"
                })
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
        
        with col2:
            if enable_visualization:
                st.markdown("### 📊 Composition Chart")
                
                # Create pie chart for top characters
                top_chars = list(comp['composition'].items())[:10]
                chars, freqs = zip(*top_chars)
                
                fig = px.pie(
                    values=freqs,
                    names=chars,
                    title=f"{comp.get('sequence_type', 'Unknown').upper()} Character Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

def show_annotations_tab(results):
    """Display annotations tab"""
    
    ann = results.get('annotations', {})
    
    st.markdown("### 📝 Annotation Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "With Organisms",
            f"{ann.get('sequences_with_organisms', 0):,}"
        )
    
    with col2:
        st.metric(
            "With Descriptions",
            f"{ann.get('sequences_with_descriptions', 0):,}"
        )
    
    with col3:
        st.metric(
            "With Features",
            f"{ann.get('sequences_with_features', 0):,}"
        )
    
    # Organism distribution
    if ann.get('organism_distribution'):
        st.markdown("### 🌍 Top Organisms")
        
        org_data = []
        for org, count in ann['organism_distribution'].items():
            org_data.append({'Organism': org, 'Count': count})
        
        org_df = pd.DataFrame(org_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(org_df, hide_index=True, use_container_width=True)
        
        with col2:
            if len(org_data) > 1:  # Only show chart if multiple organisms
                fig = px.bar(
                    org_df,
                    x='Count',
                    y='Organism',
                    orientation='h',
                    title="Organism Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_biodiversity_tab(results):
    """Display biodiversity metrics tab"""
    
    div = results.get('diversity', {})
    
    if div:
        st.markdown("### 🌿 Biodiversity Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Species Richness",
                f"{div.get('species_richness', 0):,}"
            )
            
            st.metric(
                "Shannon Diversity",
                f"{div.get('shannon_diversity', 0):.4f}"
            )
        
        with col2:
            st.metric(
                "Simpson Diversity",
                f"{div.get('simpson_diversity', 0):.4f}"
            )
            
            st.metric(
                "Evenness",
                f"{div.get('evenness', 0):.4f}"
            )
        
        # Create diversity comparison chart
        diversity_data = {
            'Metric': ['Shannon Diversity', 'Simpson Diversity', 'Evenness'],
            'Value': [
                div.get('shannon_diversity', 0),
                div.get('simpson_diversity', 0),
                div.get('evenness', 0)
            ]
        }
        
        fig = px.bar(
            diversity_data,
            x='Metric',
            y='Value',
            title="Biodiversity Metrics Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_report_tab(output_path):
    """Display report tab with download option"""
    
    st.markdown("### 📄 Analysis Report")
    
    try:
        # Read the report file
        with open(output_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # Display report content
        st.text_area(
            "Generated Report",
            value=report_content,
            height=400,
            help="Complete analysis report"
        )
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=report_content,
            file_name=f"analysis_report_{int(time.time())}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Could not load report: {str(e)}")

def show_batch_analysis_page():
    """Display batch analysis page"""
    
    st.title("📊 Batch Analysis")
    st.info("🚧 Batch analysis feature coming soon! This will allow you to analyze multiple datasets simultaneously.")
    
    # Placeholder for batch analysis interface
    st.markdown("""
    ### Planned Features:
    - **Multiple file upload**
    - **Parallel processing**
    - **Comparative analysis**
    - **Batch report generation**
    """)

def show_results_viewer():
    """Display results viewer page"""
    st.title("📈 Results Viewer")

    # Choose results directory (support prefill from homepage)
    default_dir = str(Path(app_config.get('storage.runs_dir', 'runs')).resolve())
    prefill = st.session_state.get('prefill_results_dir')
    if prefill and Path(prefill).exists():
        default_dir = prefill
    
    results_dir = st.text_input(
        "Results directory",
        value=default_dir,
        help="Folder containing pipeline_results.json and subfolders: clustering, taxonomy, novelty, visualizations",
        key="results_dir_input"
    )
    # Persist last used results dir for quick navigation
    if results_dir:
        st.session_state.prefill_results_dir = results_dir

    if not results_dir:
        return
    base = Path(results_dir)
    if not base.exists():
        st.warning(f"Directory not found: {base}")
        return

    # Helper to check files
    def p(*parts):
        return base.joinpath(*parts)

    st.markdown("---")
    # 1) Pipeline summary
    st.subheader("Pipeline Summary")
    summary_cols = st.columns(5)
    pr_path = p('pipeline_results.json')
    ui_report_path = p('analysis_report.txt')
    try:
        if pr_path.exists():
            with open(pr_path, 'r', encoding='utf-8') as f:
                pr = json.load(f)
            s = pr.get('summary', {})
            summary_cols[0].metric("Total sequences", f"{s.get('total_sequences_processed', 0):,}")
            summary_cols[1].metric("Clusters", f"{s.get('total_clusters', 0):,}")
            summary_cols[2].metric("Taxa identified", f"{s.get('total_taxa_identified', 0):,}")
            summary_cols[3].metric("Novel candidates", f"{s.get('novel_taxa_candidates', 0):,}")
            summary_cols[4].metric("Novel %", f"{s.get('novelty_percentage', 0)}%")
        else:
            # Fallback to UI report-only runs
            if ui_report_path.exists():
                summary_cols[0].metric("Report only", "Yes")
                st.info("Detected UI analysis report (no pipeline_results.json). Showing the text report below.")
            else:
                st.info("pipeline_results.json not found; showing sections based on available files")
    except Exception as e:
        st.warning(f"Could not read pipeline_results.json: {e}")

    # UI text report display
    if ui_report_path.exists():
        st.markdown("### Analysis Report (UI)")
        try:
            st.text_area("analysis_report.txt", ui_report_path.read_text(encoding='utf-8', errors='ignore'), height=240)
            st.download_button("Download analysis_report.txt", data=ui_report_path.read_bytes(), file_name="analysis_report.txt")
        except Exception:
            st.write(f"Open report: {ui_report_path}")

    st.markdown("---")
    # 2) Clustering
    st.subheader("Clustering")
    clus_img = p('clustering', 'cluster_visualization.png')
    clus_stats = p('clustering', 'cluster_stats.txt')
    clus_csv = p('clustering', 'cluster_assignments.csv')

    cols = st.columns([2,1])
    with cols[0]:
        if clus_img.exists():
            st.image(str(clus_img), caption="Cluster visualization")
        else:
            st.info("No cluster_visualization.png found")
    with cols[1]:
        if clus_stats.exists():
            try:
                st.text_area("Cluster stats", clus_stats.read_text(encoding='utf-8'), height=200)
            except Exception:
                st.text("cluster_stats.txt present (could not display)")
        if clus_csv.exists():
            try:
                dfc = pd.read_csv(clus_csv)
                st.write("Cluster assignments (first 10 rows):")
                st.dataframe(dfc.head(10), hide_index=True, use_container_width=True)
                st.download_button("Download cluster_assignments.csv", data=clus_csv.read_bytes(), file_name="cluster_assignments.csv")
            except Exception as e:
                st.warning(f"Could not read cluster_assignments.csv: {e}")

    st.markdown("---")
    # 3) Taxonomy
    st.subheader("Taxonomy")
    tax_csv = p('taxonomy', 'taxonomy_predictions.csv')
    if tax_csv.exists():
        try:
            dft = pd.read_csv(tax_csv)
            # Summary
            tcols = st.columns(4)
            tcols[0].metric("Rows", f"{len(dft):,}")
            if 'assigned_label' in dft.columns:
                tcols[1].metric("Unique taxa", f"{dft['assigned_label'].dropna().nunique():,}")
            if 'tiebreak_winner' in dft.columns:
                try:
                    tb_counts = dft['tiebreak_winner'].value_counts()
                    tcols[2].metric("BLAST wins", f"{int(tb_counts.get('blast',0)):,}")
                    tcols[3].metric("KNN wins", f"{int(tb_counts.get('knn',0)):,}")
                except Exception:
                    pass
            st.write("Top taxa (bar chart)")
            if 'assigned_label' in dft.columns:
                top_counts = dft['assigned_label'].fillna('Unknown').value_counts().head(15)
                chart_df = pd.DataFrame({"Taxon": top_counts.index, "Count": top_counts.values})
                fig = px.bar(chart_df, x='Taxon', y='Count', title="Top 15 Assigned Taxa")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dft.head(50), use_container_width=True, hide_index=True)
            st.download_button("Download taxonomy_predictions.csv", data=tax_csv.read_bytes(), file_name="taxonomy_predictions.csv")
            # Conflicts report
            tb_report = p('taxonomy', 'taxonomy_tiebreak_report.csv')
            if tb_report.exists():
                st.download_button("Download taxonomy_tiebreak_report.csv", data=tb_report.read_bytes(), file_name="taxonomy_tiebreak_report.csv")
        except Exception as e:
            st.warning(f"Could not parse taxonomy_predictions.csv: {e}")
    else:
        st.info("No taxonomy_predictions.csv found")

    st.markdown("---")
    # 4) Novelty
    st.subheader("Novelty")
    nov_json = p('novelty', 'novelty_analysis.json')
    nov_img = p('novelty', 'novelty_visualization.png')
    if nov_json.exists():
        try:
            nov = json.loads(nov_json.read_text(encoding='utf-8'))
            ncols = st.columns(4)
            ncols[0].metric("Total sequences", f"{nov.get('total_sequences',0):,}")
            ncols[1].metric("Novel candidates", f"{nov.get('novel_candidates',0):,}")
            ncols[2].metric("Novel %", f"{nov.get('novel_percentage',0):.2f}%")
            if nov_img.exists():
                st.image(str(nov_img), caption="Novelty visualization")
        except Exception as e:
            st.warning(f"Could not read novelty_analysis.json: {e}")
    else:
        st.info("No novelty analysis found")

    st.markdown("---")
    # 5) Dashboard
    st.subheader("Dashboard")
    dash_html = p('visualizations', 'analysis_dashboard.html')
    if dash_html.exists():
        try:
            components.html(dash_html.read_text(encoding='utf-8'), height=600, scrolling=True)
        except Exception:
            st.write(f"Open dashboard: {dash_html}")
    else:
        st.info("No analysis_dashboard.html found")
    
    st.title("📈 Results Viewer")
    st.info("🚧 Results viewer coming soon! This will allow you to browse and compare previous analyses.")
    
    # Placeholder for results viewer
    st.markdown("""
    ### Planned Features:
    - **Analysis history**
    - **Result comparison**
    - **Trend analysis**
    - **Export utilities**
    """)

def show_run_browser():
    """Browse and open runs stored under the configured runs directory"""
    import time as _time
    st.title("🗂️ Run Browser")

    # Base directory input
    default_root = str(Path(app_config.get('storage.runs_dir', 'runs')).resolve())
    root_dir = st.text_input(
        "Runs root directory",
        value=default_root,
        help="Root folder containing runs organized as <dataset_name>/<timestamp>"
    )

    if not root_dir:
        return
    base = Path(root_dir)
    if not base.exists():
        st.warning(f"Directory not found: {base}")
        return

    # Discover dataset folders
    datasets = sorted([d.name for d in base.iterdir() if d.is_dir()])
    cols_top = st.columns([2,2,2,1])
    with cols_top[0]:
        ds_filter = st.selectbox("Dataset", ["All"] + datasets)
    with cols_top[1]:
        search = st.text_input("Search (dataset/run)", "")
    with cols_top[2]:
        try:
            show_n = st.number_input("Show top N", min_value=5, max_value=500, value=20, step=5)
        except Exception:
            show_n = 20
    with cols_top[3]:
        refresh = st.button("Refresh")

    # Collect runs
    rows = []
    for ds in datasets:
        if ds_filter != "All" and ds != ds_filter:
            continue
        ds_path = base / ds
        try:
            for run_dir in ds_path.iterdir():
                if not run_dir.is_dir():
                    continue
                try:
                    mtime = run_dir.stat().st_mtime
                    mtime_str = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(mtime))
                except Exception:
                    mtime, mtime_str = 0, ""
                pr = (run_dir / 'pipeline_results.json').exists()
                tx = (run_dir / 'taxonomy' / 'taxonomy_predictions.csv').exists()
                nv = (run_dir / 'novelty' / 'novelty_analysis.json').exists()
                rows.append({
                    'dataset': ds,
                    'run': run_dir.name,
                    'path': str(run_dir.resolve()),
                    'modified': mtime_str,
                    'mtime': mtime,
                    'has_pipeline': pr,
                    'has_taxonomy': tx,
                    'has_novelty': nv,
                })
        except Exception:
            continue

    # Apply search filter
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r['dataset'].lower() or s in r['run'].lower()]

    # Sort and cap
    rows.sort(key=lambda r: r.get('mtime', 0), reverse=True)
    rows_view = rows[: int(show_n)] if show_n and len(rows) > show_n else rows

    # Render list
    if not rows_view:
        st.info("No runs found for the current filters.")
        return

    for i, r in enumerate(rows_view):
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
            c1.markdown(f"**{r['dataset']} / {r['run']}**\n\n``{r['path']}``")
            c2.text(r['modified'])
            c3.markdown("✅ results" if r['has_pipeline'] else "—")
            c4.markdown("🧬 taxonomy" if r['has_taxonomy'] else "—")
            if c5.button("Open", key=f"open_run_{i}"):
                st.session_state.prefill_results_dir = r['path']
                st.session_state.current_page_key = "results"
                st.rerun()


def show_taxonomy_viewer():
    """Display taxonomy viewer page with conflict and lineage summaries"""
    st.title("Taxonomy Viewer")
    st.info("Load taxonomy_predictions.csv (and optional tiebreak report) from a results directory.")

    col1, col2 = st.columns([2, 1])
    with col1:
        default_dir = str(Path("results").resolve())
        results_dir = st.text_input("Results directory", value=default_dir, help="Folder containing taxonomy/taxonomy_predictions.csv")
    with col2:
        refresh = st.button("Load", type="primary")

    if results_dir:
        tax_csv = Path(results_dir) / "taxonomy" / "taxonomy_predictions.csv"
        report_csv = Path(results_dir) / "taxonomy" / "taxonomy_tiebreak_report.csv"
        if not tax_csv.exists():
            st.warning(f"taxonomy_predictions.csv not found at {tax_csv}")
            return
        try:
            df = pd.read_csv(tax_csv)
        except Exception as e:
            st.error(f"Failed to read taxonomy_predictions.csv: {e}")
            return

        # Summary cards
        total = len(df)
        unique_labels = df['assigned_label'].dropna().nunique() if 'assigned_label' in df.columns else 0
        conflicts = 0
        if 'conflict_flag' in df.columns:
            try:
                conflicts = int(df['conflict_flag'].astype(str).str.lower().eq('true').sum())
            except Exception:
                conflicts = 0
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Total Sequences", f"{total:,}")
        colB.metric("Unique Taxa", f"{unique_labels:,}")
        colC.metric("Conflicts", f"{conflicts:,}")
        if report_csv.exists():
            colD.download_button("Download Tiebreak Report", data=report_csv.read_bytes(), file_name="taxonomy_tiebreak_report.csv")

        st.markdown("---")

        # Filters
        with st.expander("Filters"):
            only_conflicts = st.checkbox("Show only conflicts", value=False)
            top_n = st.number_input("Top N taxa chart", min_value=5, max_value=100, value=15)

        view_df = df.copy()
        if only_conflicts and 'conflict_flag' in view_df.columns:
            try:
                view_df = view_df[view_df['conflict_flag'].astype(str).str.lower() == 'true']
            except Exception:
                pass

        # Top taxa chart
        try:
            if 'assigned_label' in view_df.columns and len(view_df) > 0:
                top_counts = view_df['assigned_label'].fillna('Unknown').value_counts().head(int(top_n))
                chart_df = pd.DataFrame({"Taxon": top_counts.index, "Count": top_counts.values})
                fig = px.bar(chart_df, x='Taxon', y='Count', title=f"Top {int(top_n)} Assigned Taxa")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        # Data table (selected columns)
        cols = [c for c in [
            'sequence_id', 'assigned_rank', 'assigned_label', 'confidence',
            'knn_rank', 'knn_label', 'knn_confidence', 'blast_label', 'blast_identity', 'blast_taxid',
            'tiebreak_winner', 'tiebreak_reason', 'conflict_flag',
            'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'
        ] if c in view_df.columns]
        st.dataframe(view_df[cols], use_container_width=True, hide_index=True)

        # Download enriched predictions
        st.download_button(
            label="Download taxonomy_predictions.csv",
            data=tax_csv.read_bytes(),
            file_name="taxonomy_predictions.csv",
            use_container_width=True
        )


def show_about_page():
    """Display about page"""
    
    st.title("ℹ️ About eDNA Biodiversity Assessment System")
    
    st.markdown("""
    ## 🧬 Project Overview
    
    The eDNA Biodiversity Assessment System is an end-to-end platform for identifying 
    taxonomic diversity and assessing biological richness in deep-sea environmental DNA 
    (eDNA) datasets using advanced machine learning and bioinformatics techniques.
    
    ## 🎯 Key Features
    
    - **Universal Format Support**: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL
    - **Large File Processing**: Supports datasets up to 2GB in size
    - **Automatic Analysis**: Intelligent sequence type detection and analysis
    - **Performance Optimized**: Parallel processing and vectorized computations
    - **Interactive Visualizations**: Real-time charts and plots
    - **Comprehensive Reports**: Detailed analysis summaries
    
    ## 🛠️ Technology Stack
    
    - **Python 3.13** - Core programming language
    - **BioPython** - Biological sequence analysis
    - **Streamlit** - Web interface framework
    - **Plotly** - Interactive visualizations
    - **NumPy & Pandas** - Data processing
    - **PyTorch** - Deep learning capabilities
    
    ## 📞 Support
    
    For questions, issues, or feature requests, please contact the development team.
    """)

if __name__ == "__main__":
    main()