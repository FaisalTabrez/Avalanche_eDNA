"""
Biodiversity Results Page - Unified view for all analysis runs and results
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import shutil
import zipfile
import tempfile
import time
from pathlib import Path
from datetime import datetime
from src.ui.data_manager import get_data_manager, RunInfo
from src.ui.components.run_selector import current_run_header


def render():
    """Display unified biodiversity results page"""
    st.title("🧬 Biodiversity Results")
    
    st.markdown("""
    Browse all stored analysis runs, upload external runs, and view detailed results in one place.
    """)
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["📁 Browse Outputs", "⬆️ Upload Run"])
    
    with tab1:
        render_browse_outputs()
    
    with tab2:
        render_upload_run()


def render_browse_outputs():
    """Browse and view stored analysis outputs"""
    dm = get_data_manager()
    
    # Discover all runs
    runs = dm.discover_runs()
    
    if not runs:
        st.info("No analysis outputs found. Upload a run or perform a new analysis.")
        return
    
    st.markdown(f"### Found **{len(runs)}** analysis runs")
    
    # Filters in sidebar
    with st.sidebar:
        st.markdown("### Filters")
        
        # Dataset filter
        datasets = dm.get_datasets()
        selected_datasets = st.multiselect(
            "Datasets",
            options=datasets,
            default=datasets,
            key="filter_datasets"
        )
        
        # Content filters
        st.markdown("**Show runs with:**")
        show_pipeline = st.checkbox("Pipeline results", value=True, key="filter_pipeline")
        show_taxonomy = st.checkbox("Taxonomy predictions", value=True, key="filter_taxonomy")
        show_clustering = st.checkbox("Clustering results", value=True, key="filter_clustering")
        show_novelty = st.checkbox("Novelty analysis", value=True, key="filter_novelty")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            options=["Newest first", "Oldest first", "Dataset name"],
            key="sort_by"
        )
    
    # Apply filters
    filtered_runs = [
        r for r in runs
        if r.dataset in selected_datasets
        and (not show_pipeline or r.has_pipeline)
        and (not show_taxonomy or r.has_taxonomy)
        and (not show_clustering or r.has_clustering)
        and (not show_novelty or r.has_novelty)
    ]
    
    # Apply sorting
    if sort_by == "Oldest first":
        filtered_runs = sorted(filtered_runs, key=lambda r: r.mtime)
    elif sort_by == "Dataset name":
        filtered_runs = sorted(filtered_runs, key=lambda r: (r.dataset, r.mtime), reverse=True)
    
    if not filtered_runs:
        st.warning("No runs match the current filters.")
        return
    
    st.markdown(f"Showing **{len(filtered_runs)}** runs")
    st.markdown("---")
    
    # Display runs as expandable cards
    for run in filtered_runs:
        render_run_card(run, dm)


def render_run_card(run: RunInfo, dm):
    """Render an expandable card for a single run"""
    
    # Card header
    with st.expander(
        f"🔬 **{run.dataset}** · `{run.run_id}` · {run.modified}",
        expanded=False
    ):
        cols = st.columns([2, 1])
        
        with cols[0]:
            # Run metadata
            st.markdown(f"""
            **Dataset:** {run.dataset}  
            **Run ID:** `{run.run_id}`  
            **Modified:** {run.modified}  
            **Path:** `{run.path}`
            """)
            
            # Content badges
            badges = []
            if run.has_pipeline:
                badges.append("✓ Pipeline")
            if run.has_taxonomy:
                badges.append("✓ Taxonomy")
            if run.has_clustering:
                badges.append("✓ Clustering")
            if run.has_novelty:
                badges.append("✓ Novelty")
            
            st.markdown(" · ".join(badges))
        
        with cols[1]:
            # Actions
            if st.button("📖 View Details", key=f"view_{run.dataset}_{run.run_id}"):
                dm.set_current_run(run.path)
                st.rerun()
            
            if st.button("🗑️ Delete", key=f"delete_{run.dataset}_{run.run_id}"):
                if st.session_state.get(f"confirm_delete_{run.dataset}_{run.run_id}"):
                    try:
                        shutil.rmtree(run.path)
                        st.success(f"Deleted run: {run.dataset}/{run.run_id}")
                        dm.discover_runs.clear()  # Clear cache
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")
                else:
                    st.session_state[f"confirm_delete_{run.dataset}_{run.run_id}"] = True
                    st.warning("Click again to confirm deletion")
        
        # Show results if selected
        current_run = dm.get_current_run()
        if current_run and current_run == run.path:
            st.markdown("---")
            st.markdown("### 📊 Results")
            render_run_results(run, dm)


def render_run_results(run: RunInfo, dm):
    """Render detailed results for a run"""
    base = run.path
    files = dm.get_run_files(base)
    
    # Helper to check files
    def p(*parts):
        return base.joinpath(*parts)
    
    # Pipeline summary
    pr_path = p('pipeline_results.json')
    if pr_path.exists():
        st.markdown("#### Pipeline Summary")
        try:
            with open(pr_path, 'r', encoding='utf-8') as f:
                pr = json.load(f)
            s = pr.get('summary', {})
            
            cols = st.columns(5)
            cols[0].metric("Sequences", f"{s.get('total_sequences_processed', 0):,}")
            cols[1].metric("Clusters", f"{s.get('total_clusters', 0):,}")
            cols[2].metric("Taxa", f"{s.get('total_taxa_identified', 0):,}")
            cols[3].metric("Novel", f"{s.get('novel_taxa_candidates', 0):,}")
            cols[4].metric("Novel %", f"{s.get('novelty_percentage', 0)}%")
        except Exception as e:
            st.warning(f"Could not read pipeline summary: {e}")
    
    # Analysis report
    ui_report_path = p('analysis_report.txt')
    if ui_report_path.exists():
        with st.expander("📄 Analysis Report", expanded=False):
            try:
                report_text = ui_report_path.read_text(encoding='utf-8', errors='ignore')
                st.text_area("Report", report_text, height=200, key=f"report_{run.run_id}")
                st.download_button(
                    "Download Report",
                    data=ui_report_path.read_bytes(),
                    file_name=f"{run.dataset}_{run.run_id}_report.txt",
                    key=f"download_report_{run.run_id}"
                )
            except Exception as e:
                st.error(f"Could not read report: {e}")
    
    # Clustering results
    if run.has_clustering:
        with st.expander("🔗 Clustering Results", expanded=False):
            render_clustering_results(base, run.run_id)
    
    # Taxonomy results
    if run.has_taxonomy:
        with st.expander("🧬 Taxonomy Predictions", expanded=False):
            render_taxonomy_results(base, run.run_id)
    
    # Novelty results
    if run.has_novelty:
        with st.expander("✨ Novelty Analysis", expanded=False):
            render_novelty_results(base, run.run_id)


def render_clustering_results(base: Path, run_id: str):
    """Render clustering results section"""
    clus_img = base / 'clustering' / 'cluster_visualization.png'
    clus_stats = base / 'clustering' / 'cluster_stats.txt'
    clus_csv = base / 'clustering' / 'cluster_assignments.csv'
    
    cols = st.columns([2, 1])
    
    with cols[0]:
        if clus_img.exists():
            st.image(str(clus_img), caption="Cluster Visualization", use_container_width=True)
    
    with cols[1]:
        if clus_stats.exists():
            try:
                st.text_area("Stats", clus_stats.read_text(encoding='utf-8', errors='ignore'), height=300, key=f"clus_stats_{run_id}")
            except Exception:
                st.write(f"Stats at: {clus_stats}")
    
    if clus_csv.exists():
        try:
            df = pd.read_csv(clus_csv)
            st.dataframe(df.head(100), use_container_width=True)
            st.download_button(
                "Download cluster_assignments.csv",
                data=clus_csv.read_bytes(),
                file_name="cluster_assignments.csv",
                key=f"download_clus_{run_id}"
            )
        except Exception as e:
            st.error(f"Could not read clustering CSV: {e}")


def render_taxonomy_results(base: Path, run_id: str):
    """Render taxonomy results section"""
    tax_csv = base / 'taxonomy' / 'taxonomy_predictions.csv'
    
    if tax_csv.exists():
        try:
            df = pd.read_csv(tax_csv)
            st.dataframe(df.head(100), use_container_width=True)
            
            # Quick stats
            if 'phylum' in df.columns:
                st.markdown("**Top 5 Phyla:**")
                phylum_counts = df['phylum'].value_counts().head(5)
                fig = px.bar(
                    x=phylum_counts.values,
                    y=phylum_counts.index,
                    orientation='h',
                    labels={'x': 'Count', 'y': 'Phylum'},
                    title="Top 5 Phyla"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.download_button(
                "Download taxonomy_predictions.csv",
                data=tax_csv.read_bytes(),
                file_name="taxonomy_predictions.csv",
                key=f"download_tax_{run_id}"
            )
        except Exception as e:
            st.error(f"Could not read taxonomy CSV: {e}")


def render_novelty_results(base: Path, run_id: str):
    """Render novelty analysis section"""
    nov_json = base / 'novelty' / 'novelty_analysis.json'
    nov_img = base / 'novelty' / 'novelty_visualization.png'
    
    if nov_json.exists():
        try:
            with open(nov_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            st.json(data)
            
            if nov_img.exists():
                st.image(str(nov_img), caption="Novelty Visualization", use_container_width=True)
        except Exception as e:
            st.error(f"Could not read novelty data: {e}")


def render_upload_run():
    """Upload external run results"""
    st.markdown("### Upload External Run")
    
    st.markdown("""
    Upload analysis results from external sources. Supported formats:
    - **ZIP archive** containing the run folder structure
    - **Individual folder** dragged and dropped (browser permitting)
    
    Expected structure:
    ```
    run_folder/
    ├── pipeline_results.json (optional)
    ├── analysis_report.txt (optional)
    ├── clustering/
    │   ├── cluster_assignments.csv
    │   ├── cluster_visualization.png
    │   └── cluster_stats.txt
    ├── taxonomy/
    │   └── taxonomy_predictions.csv
    └── novelty/
        ├── novelty_analysis.json
        └── novelty_visualization.png
    ```
    """)
    
    # Upload form
    with st.form("upload_run_form"):
        dataset_name = st.text_input(
            "Dataset Name",
            placeholder="e.g., My_Dataset",
            help="Name of the dataset this run belongs to"
        )
        
        uploaded_file = st.file_uploader(
            "Upload ZIP archive",
            type=['zip'],
            help="ZIP file containing the run folder"
        )
        
        submitted = st.form_submit_button("Upload Run")
        
        if submitted:
            if not dataset_name:
                st.error("Please provide a dataset name")
            elif not uploaded_file:
                st.error("Please select a file to upload")
            else:
                process_upload(dataset_name, uploaded_file)


def process_upload(dataset_name: str, uploaded_file):
    """Process uploaded run archive"""
    dm = get_data_manager()
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Save uploaded file
            zip_path = tmpdir_path / uploaded_file.name
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir_path)
            
            # Find the run folder (should be the only directory)
            extracted_dirs = [d for d in tmpdir_path.iterdir() if d.is_dir()]
            
            if not extracted_dirs:
                st.error("No folder found in ZIP archive")
                return
            
            source_dir = extracted_dirs[0]
            
            # Generate run ID with timestamp
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Destination path
            dest_dir = dm.runs_root / dataset_name / run_id
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the run folder
            shutil.copytree(source_dir, dest_dir)
            
            st.success(f"✓ Successfully uploaded run: {dataset_name}/{run_id}")
            st.info(f"Location: `{dest_dir}`")
            
            # Clear cache
            dm.discover_runs.clear()
            
            # Auto-select the new run
            dm.set_current_run(dest_dir)
            
            time.sleep(1)
            st.rerun()
            
    except zipfile.BadZipFile:
        st.error("Invalid ZIP file")
    except Exception as e:
        st.error(f"Upload failed: {e}")
        st.exception(e)
