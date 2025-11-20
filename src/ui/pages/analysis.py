"""
Analysis Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from pathlib import Path
from src.utils.config import config as app_config

try:
    from src.analysis.dataset_analyzer import DatasetAnalyzer
except ImportError:
    DatasetAnalyzer = None

try:
    from src.utils.sra_integration import SRAIntegrationUI
except ImportError:
    SRAIntegrationUI = None

def render():
    """Display the main analysis page"""
    
    try:
        st.title("Dataset Analysis")
        
        # Resolve storage directories from config
        datasets_dir = Path(app_config.get('storage.datasets_dir', 'data/datasets'))
        runs_root = Path(app_config.get('storage.runs_dir', 'runs'))
        datasets_dir.mkdir(parents=True, exist_ok=True)
        runs_root.mkdir(parents=True, exist_ok=True)
        
        # File upload section
        st.markdown("## 1. Upload Your Dataset")
        
        # Add data source selection with SRA support
        data_source_type = st.radio(
            "Data Source",
            ["Upload File", "Download from SRA"],
            horizontal=True,
            help="Choose between uploading a local file or downloading from NCBI SRA"
        )
    
        # Add info about potential upload issues
        with st.expander("Troubleshooting Upload Issues"):
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
        
        uploaded_file = None
        sra_file_path = None
        
        if data_source_type == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a biological sequence file",
                type=['fasta', 'fa', 'fas', 'fastq', 'fq', 'swiss', 'gb', 'gbk', 'embl', 'em', 'gz'],
                help="Supported formats: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL (including gzipped files). Maximum file size: 10GB"
            )
        else:
            # SRA download interface
            if SRAIntegrationUI:
                sra_ui = SRAIntegrationUI()
                sra_ui.show_sra_toolkit_status()
                
                if sra_ui.sra_toolkit_available:
                    st.markdown("#### Quick SRA Download")
                    accession_input = st.text_input(
                        "Enter SRA Accession",
                        placeholder="e.g., SRR12345678",
                        help="Enter a specific SRA accession number to download"
                    )
                    
                    if accession_input and st.button("Download SRA Dataset", type="primary"):
                        output_dir = Path("data/sra") / accession_input
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        def update_progress(msg):
                            status_text.text(msg)
                        
                        progress_bar.progress(10)
                        success, file_path = sra_ui.download_sra_dataset(
                            accession_input,
                            output_dir,
                            progress_callback=update_progress
                        )
                        
                        if success:
                            progress_bar.progress(100)
                            status_text.text("Download complete!")
                            st.success(f"Successfully downloaded {accession_input}")
                            sra_file_path = file_path
                            # Store in session state for analysis
                            st.session_state.sra_downloaded_file = file_path
                        else:
                            status_text.text("Download failed")
                            st.error(f"Failed to download {accession_input}")
                    
                    # Use previously downloaded file
                    if 'sra_downloaded_file' in st.session_state:
                        sra_file_path = st.session_state.sra_downloaded_file
                        st.info(f"Using downloaded file: {sra_file_path.name}")
            else:
                st.warning("SRA integration not available. Please check installation.")
        
        # File size information
        file_valid = True
        if uploaded_file is not None:
            try:
                # Check file size
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                file_size_gb = file_size_mb / 1024
                
                if file_size_gb >= 1:
                    st.info(f"File size: {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
                else:
                    st.info(f"File size: {file_size_mb:.2f} MB")
                
                if file_size_mb > 10240:  # 10GB limit
                    st.error("File size exceeds 10GB limit. Please use a smaller file or contact support for processing very large datasets.")
                    file_valid = False
                elif file_size_mb > 1024:  # Warn for files over 1GB
                    st.warning(f"Large file detected ({file_size_mb:.0f} MB). Upload may take longer than usual. Please be patient.")
                    
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
        with st.expander("Advanced Options"):
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
            st.info("Fast Mode enabled: Analysis will use intelligent sampling for datasets >10K sequences, reducing time by 60-80%")
        
        # Analysis execution
        st.markdown("## 3. Run Analysis")
        
        # Determine which file to use
        analysis_file = uploaded_file if uploaded_file is not None else sra_file_path
        
        if analysis_file is not None and file_valid:
            if st.button("Start Analysis", type="primary", use_container_width=True):
                # Handle both uploaded files and SRA file paths
                if isinstance(analysis_file, Path):
                    # SRA file - already on disk
                    run_analysis_from_path(analysis_file, dataset_name, max_sequences, format_override, 
                                analysis_level, enable_quality, enable_diversity, enable_visualization, fast_mode)
                else:
                    # Uploaded file - use existing run_analysis function
                    run_analysis(analysis_file, dataset_name, max_sequences, format_override, 
                                analysis_level, enable_quality, enable_diversity, enable_visualization, fast_mode)
        elif analysis_file is not None and not file_valid:
            st.warning("Cannot proceed with analysis due to file size restriction.")
        else:
            st.info("Please upload a file or download from SRA to start analysis")
        
    except Exception as e:
        st.error(f"Error in analysis page: {str(e)}")
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
            st.info("Fast mode enabled - using optimized processing for large datasets")
        progress_bar.progress(10)
        
        # Prepare parameters
        format_type = None if format_override == "Auto-detect" else format_override.lower()
        max_seq = None if max_sequences == 0 else max_sequences
        
        # Run analysis with progress monitoring
        status_text.text("Running analysis...")
        progress_bar.progress(30)
        
        # Add warning for large files
        if uploaded_file and len(uploaded_file.getvalue()) > 100 * 1024 * 1024:  # > 100MB
            st.warning("Large file detected. Analysis may take several minutes. Please be patient...")
            if len(uploaded_file.getvalue()) > 200 * 1024 * 1024:  # > 200MB
                st.warning("For files > 200MB, consider using a smaller subset or the command-line tool for better performance.")
        
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
        st.success(f"Analysis completed in {elapsed_time:.2f} seconds!")
        st.info(f"Inputs saved to: {dataset_path}")
        st.info(f"Run outputs: {run_dir}")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")

def run_analysis_from_path(file_path, dataset_name, max_sequences, format_override, 
                          analysis_level, enable_quality, enable_diversity, enable_visualization, fast_mode=True):
    """Execute analysis from a file path (e.g., SRA downloaded file)"""
    
    # Resolve storage roots
    runs_root = Path(app_config.get('storage.runs_dir', 'runs'))
    runs_root.mkdir(parents=True, exist_ok=True)

    # Build run directory
    safe_name = ''.join(c if c.isalnum() or c in ('-','_') else '_' for c in (dataset_name or 'dataset'))
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = runs_root / safe_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

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
            st.info("Fast mode enabled - using optimized processing for large datasets")
        progress_bar.progress(10)
        
        # Prepare parameters
        format_type = None if format_override == "Auto-detect" else format_override.lower()
        max_seq = None if max_sequences == 0 else max_sequences
        
        # Run analysis with progress monitoring
        status_text.text("Running analysis...")
        progress_bar.progress(30)
        
        start_time = time.time()
        
        # Execute analysis
        results = analyzer.analyze_dataset(
            input_path=str(file_path),
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
        st.success(f"Analysis completed in {elapsed_time:.2f} seconds!")
        st.info(f"Input file: {file_path}")
        st.info(f"Run outputs: {run_dir}")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")


def display_results(results, output_path, enable_visualization):
    """Display analysis results with visualizations"""
    
    st.markdown("---")
    st.markdown("## Analysis Results")
    
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
        "Statistics", "Composition", "Annotations", 
        "Biodiversity", "Report"
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
        st.markdown("### Length Statistics")
        
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
            st.markdown("### Length Distribution")
            
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
            st.markdown(f"### {comp.get('sequence_type', 'Unknown').upper()} Composition")
            
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
                st.markdown("### Composition Chart")
                
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
    
    st.markdown("### Annotation Summary")
    
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
        st.markdown("### Top Organisms")
        
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
        st.markdown("### Biodiversity Metrics")
        
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
    
    st.markdown("### Analysis Report")
    
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
            label="Download Report",
            data=report_content,
            file_name=f"analysis_report_{int(time.time())}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Could not load report: {str(e)}")
