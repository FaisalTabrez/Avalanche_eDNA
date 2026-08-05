"""
Step 1: Dataset Selection Component
Handles upload, existing datasets, and SRA downloads
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.utils.config import config as app_config

try:
    from src.utils.sra_integration import SRAIntegrationUI
except ImportError:
    SRAIntegrationUI = None


def render_dataset_selection():
    """Render dataset selection interface"""

    # Dataset source tabs
    tab1, tab2, tab3 = st.tabs(
        ["📤 Upload File", "💾 Existing Datasets", "🌐 Download from SRA"]
    )

    with tab1:
        render_upload_tab()

    with tab2:
        render_existing_tab()

    with tab3:
        render_sra_tab()

    # Navigation
    st.divider()

    # Check if dataset is selected
    if st.session_state.workflow_dataset:
        st.success(
            f"✓ Selected: {st.session_state.workflow_dataset.get('name', 'Unknown')}"
        )
        if st.button("Next: Configure →", type="primary", use_container_width=True):
            st.session_state.workflow_step = 2
            st.rerun()
    else:
        st.info("Please select or upload a dataset to continue")


def render_upload_tab():
    """File upload interface"""
    st.markdown("### Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose a biological sequence file",
        type=[
            "fasta",
            "fa",
            "fas",
            "fastq",
            "fq",
            "swiss",
            "gb",
            "gbk",
            "embl",
            "em",
            "gz",
        ],
        help="Supported: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL (including .gz). Max: 10GB",
        key="workflow_file_upload",
    )

    if uploaded_file is not None:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        file_size_gb = file_size_mb / 1024

        if file_size_gb >= 1:
            st.info(f"📊 File size: {file_size_gb:.2f} GB")
        else:
            st.info(f"📊 File size: {file_size_mb:.2f} MB")

        # Size warnings
        if file_size_mb > 10240:  # 10GB limit
            st.error("⚠️ File exceeds 10GB limit. Please use a smaller file.")
            return
        elif file_size_mb > 1024:  # Warn for files over 1GB
            st.warning(
                f"⚠️ Large file ({file_size_mb:.0f} MB). Upload may take longer."
            )

        # Dataset name
        dataset_name = st.text_input(
            "Dataset Name",
            value=uploaded_file.name.rsplit(".", 1)[0],
            key="workflow_dataset_name",
        )

        if st.button("Use This File", type="primary"):
            # Save file to datasets directory
            datasets_dir = Path(app_config.get("storage.datasets_dir", "data/datasets"))
            datasets_dir.mkdir(parents=True, exist_ok=True)

            file_path = datasets_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Store in session state
            st.session_state.workflow_dataset = {
                "name": dataset_name,
                "file_path": str(file_path),
                "file_name": uploaded_file.name,
                "size_mb": file_size_mb,
                "source": "upload",
                "uploaded_at": datetime.now().isoformat(),
            }
            st.session_state.workflow_dataset_type = "upload"
            st.success(f"✓ File saved and ready: {dataset_name}")
            st.rerun()

    # Upload tips
    with st.expander("💡 Upload Tips"):
        st.markdown("""
        **Supported Formats:**
        - FASTA (.fasta, .fa, .fas)
        - FASTQ (.fastq, .fq)
        - Swiss-Prot (.swiss)
        - GenBank (.gb, .gbk)
        - EMBL (.embl, .em)
        - Compressed (.gz)
        
        **Troubleshooting:**
        - For files >1GB, ensure stable internet connection
        - If upload fails, try smaller file or use command-line
        - Maximum file size: 10GB
        """)


def render_existing_tab():
    """Browse existing datasets"""
    st.markdown("### Existing Datasets")

    datasets_dir = Path(app_config.get("storage.datasets_dir", "data/datasets"))

    if not datasets_dir.exists():
        st.info("No datasets directory found. Upload a file to create it.")
        return

    # List existing files
    files = list(datasets_dir.glob("*"))
    sequence_files = [
        f
        for f in files
        if f.suffix.lower()
        in [
            ".fasta",
            ".fa",
            ".fas",
            ".fastq",
            ".fq",
            ".swiss",
            ".gb",
            ".gbk",
            ".embl",
            ".em",
            ".gz",
        ]
    ]

    if not sequence_files:
        st.info("No datasets found. Upload a file to get started.")
        return

    st.markdown(f"Found {len(sequence_files)} dataset(s)")

    # Display as table
    dataset_data = []
    for file_path in sorted(
        sequence_files, key=lambda x: x.stat().st_mtime, reverse=True
    ):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(file_path.stat().st_mtime)

        dataset_data.append(
            {
                "Name": file_path.stem,
                "File": file_path.name,
                "Size (MB)": f"{size_mb:.2f}",
                "Modified": modified.strftime("%Y-%m-%d %H:%M"),
                "Path": str(file_path),
            }
        )

    if dataset_data:
        df = pd.DataFrame(dataset_data)

        # Display table (without path column)
        st.dataframe(
            df[["Name", "File", "Size (MB)", "Modified"]],
            use_container_width=True,
            hide_index=True,
        )

        # Selection
        selected_file = st.selectbox(
            "Select Dataset",
            options=df["Name"].tolist(),
            key="workflow_existing_dataset",
        )

        if selected_file:
            selected_row = df[df["Name"] == selected_file].iloc[0]

            st.info(f"📂 {selected_row['File']} ({selected_row['Size (MB)']} MB)")

            if st.button("Use This Dataset", type="primary"):
                st.session_state.workflow_dataset = {
                    "name": selected_file,
                    "file_path": selected_row["Path"],
                    "file_name": selected_row["File"],
                    "size_mb": float(selected_row["Size (MB)"]),
                    "source": "existing",
                    "selected_at": datetime.now().isoformat(),
                }
                st.session_state.workflow_dataset_type = "existing"
                st.success(f"✓ Dataset ready: {selected_file}")
                st.rerun()


def render_sra_tab():
    """SRA download interface"""
    st.markdown("### Download from NCBI SRA")

    if not SRAIntegrationUI:
        st.warning("⚠️ SRA integration not available. Please check installation.")
        return

    sra_ui = SRAIntegrationUI()

    # Show toolkit status
    sra_ui.show_sra_toolkit_status()

    if not sra_ui.sra_toolkit_available:
        st.info("💡 SRA Toolkit not detected, but ENA fallback download is available.")

    # Quick download
    st.markdown("#### Quick Download")

    col1, col2 = st.columns([3, 1])

    with col1:
        accession_input = st.text_input(
            "SRA Accession",
            placeholder="e.g., SRR12345678, SRX31101225",
            help="Enter SRA run (SRR), experiment (SRX), or project (SRP) accession",
            key="workflow_sra_accession",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        download_button = st.button("📥 Download", type="primary")

    if download_button and accession_input:
        output_dir = Path("data/sra") / accession_input
        output_dir.mkdir(parents=True, exist_ok=True)

        status_container = st.empty()
        progress_bar = st.progress(0)

        def update_progress(msg):
            status_container.text(msg)

        progress_bar.progress(10)

        try:
            success, file_path = sra_ui.download_sra_dataset(
                accession_input, output_dir, progress_callback=update_progress
            )

            if success and file_path:
                progress_bar.progress(100)
                status_container.success("✓ Download complete!")

                # Get file size
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                # Store in session state
                st.session_state.workflow_dataset = {
                    "name": f"{accession_input}_dataset",
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "size_mb": file_size_mb,
                    "source": "sra",
                    "accession": accession_input,
                    "downloaded_at": datetime.now().isoformat(),
                }
                st.session_state.workflow_dataset_type = "sra"

                st.success(f"✓ Ready: {file_path.name} ({file_size_mb:.2f} MB)")
                st.rerun()
            else:
                progress_bar.empty()
                status_container.error(f"❌ Failed to download {accession_input}")

        except Exception as e:
            progress_bar.empty()
            status_container.error(f"❌ Error: {str(e)}")

    # Recently downloaded (from session state)
    if "sra_downloaded_file" in st.session_state:
        st.divider()
        st.markdown("#### Recent Downloads")
        recent_file = Path(st.session_state.sra_downloaded_file)
        if recent_file.exists():
            file_size_mb = recent_file.stat().st_size / (1024 * 1024)
            st.info(f"📂 {recent_file.name} ({file_size_mb:.2f} MB)")

            if st.button("Use Recent Download", type="primary"):
                st.session_state.workflow_dataset = {
                    "name": recent_file.stem,
                    "file_path": str(recent_file),
                    "file_name": recent_file.name,
                    "size_mb": file_size_mb,
                    "source": "sra",
                    "selected_at": datetime.now().isoformat(),
                }
                st.session_state.workflow_dataset_type = "sra"
                st.success(f"✓ Dataset ready: {recent_file.stem}")
                st.rerun()

    # Tips
    with st.expander("💡 SRA Download Tips"):
        st.markdown("""
        **Supported Accessions:**
        - SRR (Run): Single sequencing run
        - SRX (Experiment): Converts to SRR automatically
        - SRP (Project): Lists available runs
        
        **Download Methods:**
        1. SRA Toolkit (if installed)
        2. ENA Fallback (automatic, no toolkit needed)
        
        **Notes:**
        - Very recent datasets may not be on ENA yet
        - Downloads are cached for 7 days
        - Large files may take several minutes
        """)
