#!/usr/bin/env python3
"""
SRA Integration Utilities for Streamlit UI

This module provides reusable components for integrating SRA Toolkit
functionality into the Streamlit web interface.
"""

import json
import logging
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from src.utils.config import config

logger = logging.getLogger(__name__)


class SRAIntegrationUI:
    """Streamlit UI components for SRA integration"""

    def __init__(self):
        """Initialize SRA integration UI"""
        self.config = config
        self.sra_config = config.get("databases", {}).get("sra", {})

        # SRA URLs
        self.sra_search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        )
        self.sra_summary_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        )

        # Check SRA Toolkit availability
        self.sra_toolkit_available = self._check_sra_toolkit()

    def _check_sra_toolkit(self) -> bool:
        """Check if SRA Toolkit is available"""
        try:
            prefetch_path = self.sra_config.get("sra_tools", {}).get(
                "prefetch_path", "prefetch"
            )
            result = subprocess.run(
                [prefetch_path, "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def show_sra_toolkit_status(self):
        """Display SRA Toolkit status banner"""
        if self.sra_toolkit_available:
            st.success("✓ SRA Toolkit is installed and configured")
        else:
            st.info("""
            ℹ️ **SRA Toolkit not detected** - Using alternative download method
            
            Downloads will use ENA (European Nucleotide Archive) mirror, which works without SRA Toolkit.
            
            **To install SRA Toolkit for faster downloads:**
            1. Run: `python install_sra_toolkit.py`
            2. Or download from: https://github.com/ncbi/sra-tools
            
            Note: ENA downloads work fine for most use cases!
            """)

    def search_sra_datasets(
        self, keywords: List[str], max_results: int = 50
    ) -> List[Dict]:
        """
        Search NCBI SRA for datasets

        Args:
            keywords: Search keywords
            max_results: Maximum results to return

        Returns:
            List of dataset metadata
        """
        # Build search query
        search_terms = (
            keywords if keywords else ["eDNA", "environmental DNA", "metabarcoding"]
        )
        query = " OR ".join(f'"{term}"' for term in search_terms)

        logger.info(f"Searching SRA with query: {query}")

        params = {
            "db": "sra",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
        }

        url = f"{self.sra_search_url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())

            study_ids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(study_ids)} SRA studies")

            return self._get_study_details(study_ids[:max_results])

        except Exception as e:
            logger.error(f"Error searching SRA: {e}")
            st.error(f"Failed to search SRA: {e}")
            return []

    def _get_study_details(self, study_ids: List[str]) -> List[Dict]:
        """Get detailed information about studies"""
        if not study_ids:
            return []

        studies = []
        batch_size = 20

        for i in range(0, len(study_ids), batch_size):
            batch_ids = study_ids[i : i + batch_size]

            params = {"db": "sra", "id": ",".join(batch_ids), "retmode": "xml"}

            url = f"{self.sra_summary_url}?{urllib.parse.urlencode(params)}"

            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    root = ET.fromstring(response.read())

                for docsum in root.findall(".//DocSum"):
                    study_info = self._parse_study_xml(docsum)
                    if study_info:
                        studies.append(study_info)

            except Exception as e:
                logger.error(f"Error getting study details for batch: {e}")
                continue

        return studies

    def _parse_study_xml(self, docsum) -> Optional[Dict]:
        """Parse study XML to extract information"""
        try:
            study_info = {}

            for item in docsum.findall(".//Item"):
                name = item.get("Name")
                value = item.text if item.text else ""

                if name in [
                    "Accession",
                    "Title",
                    "Organism",
                    "Platform",
                    "Spots",
                    "Bases",
                ]:
                    study_info[name.lower()] = value

            # Only return if has accession
            if "accession" in study_info:
                return study_info

        except Exception as e:
            logger.error(f"Error parsing study XML: {e}")

        return None

    def show_sra_browser(self, on_select_callback=None) -> Optional[str]:
        """
        Display SRA dataset browser UI

        Args:
            on_select_callback: Callback function when dataset is selected

        Returns:
            Selected accession number or None
        """
        st.markdown("### NCBI SRA Dataset Browser")

        # Search interface
        col1, col2 = st.columns([3, 1])

        with col1:
            search_terms = st.text_input(
                "Search Keywords",
                placeholder="e.g., eDNA, 18S rRNA, marine",
                help="Enter keywords to search NCBI SRA database",
            )

        with col2:
            max_results = st.number_input(
                "Max Results", min_value=10, max_value=100, value=30
            )

        search_button = st.button("Search SRA", type="primary")

        selected_accession = None

        if search_button or "sra_search_results" in st.session_state:
            if search_button:
                keywords = [
                    term.strip() for term in search_terms.split(",") if term.strip()
                ]

                with st.spinner("Searching NCBI SRA database..."):
                    results = self.search_sra_datasets(keywords, max_results)
                    st.session_state.sra_search_results = results

            results = st.session_state.get("sra_search_results", [])

            if results:
                st.success(f"Found {len(results)} datasets")

                # Display results in a table
                st.markdown("#### Search Results")

                for idx, study in enumerate(results):
                    with st.expander(
                        f"**{study.get('accession', 'Unknown')}** - {study.get('title', 'No title')[:100]}..."
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(
                                f"**Accession:** {study.get('accession', 'N/A')}"
                            )
                            st.markdown(f"**Organism:** {study.get('organism', 'N/A')}")
                            st.markdown(f"**Platform:** {study.get('platform', 'N/A')}")
                            st.markdown(f"**Title:** {study.get('title', 'N/A')}")

                        with col2:
                            spots = study.get("spots", "0")
                            bases = study.get("bases", "0")
                            st.metric(
                                "Spots", f"{int(spots):,}" if spots.isdigit() else spots
                            )
                            st.metric(
                                "Bases", f"{int(bases):,}" if bases.isdigit() else bases
                            )

                        if st.button(
                            f"Select {study.get('accession')}", key=f"select_{idx}"
                        ):
                            selected_accession = study.get("accession")
                            if on_select_callback:
                                on_select_callback(selected_accession, study)
                            st.session_state.selected_sra_accession = selected_accession
                            st.session_state.selected_sra_metadata = study
                            st.success(f"Selected: {selected_accession}")
                            st.rerun()

            else:
                st.info(
                    "No results found. Try different keywords or check your internet connection."
                )

        return selected_accession

    def _convert_srx_to_srr(self, accession: str) -> Optional[str]:
        """
        Convert SRX (experiment) accession to SRR (run) accession

        Args:
            accession: SRX accession

        Returns:
            SRR accession or None
        """
        if accession.startswith("SRR"):
            return accession

        if not accession.startswith("SRX"):
            logger.warning(f"Accession {accession} is not SRX or SRR format")
            return None

        try:
            # Query NCBI to get run accessions for this experiment
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id={accession}&retmode=xml"

            with urllib.request.urlopen(url, timeout=30) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)

                # Find RUN accession
                run_elem = root.find(".//RUN")
                if run_elem is not None:
                    srr_accession = run_elem.get("accession")
                    logger.info(f"Converted {accession} to {srr_accession}")
                    return srr_accession
                else:
                    logger.error(f"No RUN found for {accession}")
                    return None

        except Exception as e:
            logger.error(f"Error converting {accession} to SRR: {e}")
            return None

    def _download_from_ena(self, accession: str, output_dir: Path) -> Optional[Path]:
        """
        Download FASTQ files directly from ENA (European Nucleotide Archive)
        This method works without SRA Toolkit

        Args:
            accession: SRR accession
            output_dir: Output directory

        Returns:
            Path to downloaded file or None
        """
        try:
            # Get download URLs from ENA API
            ena_api_url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={accession}&result=read_run&fields=fastq_ftp,fastq_aspera,fastq_galaxy"

            logger.info(f"Querying ENA for {accession}...")
            with urllib.request.urlopen(ena_api_url, timeout=30) as response:
                data = response.read().decode("utf-8")
                lines = data.strip().split("\n")

                if len(lines) < 2:
                    logger.error(f"No data found for {accession} in ENA")
                    return None

                # Parse FTP URLs (second line contains the data)
                # Format: run_accession\tfastq_ftp\tfastq_aspera\tfastq_galaxy
                fields = lines[1].split("\t")

                if len(fields) < 2:
                    logger.error(f"Unexpected ENA API response format for {accession}")
                    return None

                ftp_urls = fields[1]  # Second field is fastq_ftp

                if not ftp_urls or ftp_urls == "":
                    logger.error(f"No FASTQ files available for {accession} in ENA")
                    logger.info(f"This dataset may require SRA Toolkit for download")
                    return None

                # Get first FASTQ file URL (for paired-end, there might be multiple separated by ;)
                ftp_url = ftp_urls.split(";")[0]

                # Convert FTP to HTTP
                if ftp_url.startswith("ftp.sra.ebi.ac.uk"):
                    http_url = f"http://{ftp_url}"
                elif ftp_url.startswith("ftp://"):
                    http_url = ftp_url.replace("ftp://", "http://")
                else:
                    http_url = f"http://{ftp_url}"

                logger.info(f"Downloading from ENA: {http_url}")

                # Download file
                filename = Path(http_url).name
                output_file = output_dir / filename

                # Create output directory
                output_dir.mkdir(parents=True, exist_ok=True)

                def download_progress(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(
                            100, int(block_num * block_size * 100 / total_size)
                        )
                        if percent % 10 == 0:  # Log every 10%
                            logger.info(f"Download progress: {percent}%")

                urllib.request.urlretrieve(
                    http_url, output_file, reporthook=download_progress
                )

                if output_file.exists():
                    logger.info(
                        f"Successfully downloaded {accession} from ENA to {output_file}"
                    )
                    return output_file

        except urllib.error.URLError as e:
            logger.error(f"URL error downloading from ENA: {e}")
        except Exception as e:
            logger.error(f"ENA download failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())

        return None

    def download_sra_dataset(
        self, accession: str, output_dir: Path, progress_callback=None
    ) -> Tuple[bool, Optional[Path]]:
        """
        Download SRA dataset using SRA Toolkit or fallback to ENA

        Args:
            accession: SRA accession number (SRR or SRX)
            output_dir: Output directory for downloaded files
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success, output_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert SRX to SRR if needed
        if accession.startswith("SRX"):
            logger.info(f"Converting experiment {accession} to run accession...")
            if progress_callback:
                progress_callback(f"Converting {accession} to run accession...")

            srr_accession = self._convert_srx_to_srr(accession)
            if not srr_accession:
                logger.error(f"Failed to convert {accession} to SRR")
                return False, None

            logger.info(f"Using run accession: {srr_accession}")
            accession = srr_accession

        # Try SRA Toolkit first if available
        if self.sra_toolkit_available:
            try:
                prefetch_path = self.sra_config.get("sra_tools", {}).get(
                    "prefetch_path", "prefetch"
                )
                fastq_dump_path = self.sra_config.get("sra_tools", {}).get(
                    "fastq_dump_path", "fastq-dump"
                )

                # Prefetch SRA file
                if progress_callback:
                    progress_callback("Downloading SRA file with SRA Toolkit...")

                logger.info(f"Prefetching {accession}...")
                result = subprocess.run(
                    [prefetch_path, accession, "-O", str(output_dir)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if result.returncode != 0:
                    logger.warning(f"Prefetch failed: {result.stderr}")
                    logger.info("Falling back to ENA download...")
                else:
                    # Convert to FASTQ
                    if progress_callback:
                        progress_callback("Converting to FASTQ format...")

                    sra_file = output_dir / accession / f"{accession}.sra"
                    if not sra_file.exists():
                        sra_file = output_dir / f"{accession}.sra"

                    if sra_file.exists():
                        logger.info(f"Converting {accession} to FASTQ...")
                        fastq_result = subprocess.run(
                            [
                                fastq_dump_path,
                                "--outdir",
                                str(output_dir),
                                "--gzip",
                                "--skip-technical",
                                "--readids",
                                "--read-filter",
                                "pass",
                                "--dumpbase",
                                "--split-3",
                                "--clip",
                                str(sra_file),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=1200,
                        )

                        if fastq_result.returncode == 0:
                            fastq_files = list(
                                output_dir.glob(f"{accession}*.fastq.gz")
                            )
                            if fastq_files:
                                logger.info(
                                    f"Successfully downloaded and converted {accession}"
                                )
                                return True, fastq_files[0]
                        else:
                            logger.error(
                                f"FASTQ conversion failed: {fastq_result.stderr}"
                            )

            except subprocess.TimeoutExpired:
                logger.warning("SRA Toolkit download timeout, falling back to ENA...")
            except Exception as e:
                logger.warning(
                    f"SRA Toolkit method failed: {e}, falling back to ENA..."
                )

        # Fallback to ENA download (works without SRA Toolkit)
        logger.info("Using ENA direct download method...")
        if progress_callback:
            progress_callback(
                "Downloading from ENA mirror (no SRA Toolkit required)..."
            )

        ena_file = self._download_from_ena(accession, output_dir)

        if ena_file and ena_file.exists():
            logger.info(f"Successfully downloaded {accession} from ENA")
            return True, ena_file
        else:
            logger.error(f"All download methods failed for {accession}")
            return False, None

    def show_sra_download_ui(self, accession: str, output_dir: Path) -> Optional[Path]:
        """
        Show download UI for a specific accession

        Args:
            accession: SRA accession to download
            output_dir: Output directory

        Returns:
            Path to downloaded file or None
        """
        st.markdown(f"### Download {accession}")

        # Show method info
        if self.sra_toolkit_available:
            st.info("Using SRA Toolkit for download (faster)")
        else:
            st.info("Using ENA mirror for download (no SRA Toolkit required)")

        if st.button(f"Download {accession}", type="primary"):
            status_text = st.empty()
            progress_bar = st.progress(0)

            def update_progress(message):
                status_text.text(message)

            progress_bar.progress(10)
            success, file_path = self.download_sra_dataset(
                accession, output_dir, progress_callback=update_progress
            )

            if success:
                progress_bar.progress(100)
                status_text.text("Download complete!")
                st.success(f"Successfully downloaded {accession}")
                st.info(f"File location: {file_path}")
                return file_path
            else:
                status_text.text("Download failed")
                st.error(f"Failed to download {accession}")
                return None

        return None


def create_sra_data_source_selector() -> Tuple[str, Optional[Path], Optional[Dict]]:
    """
    Create a unified data source selector supporting local files and SRA downloads

    Returns:
        Tuple of (source_type, file_path, metadata)
        source_type: "local" or "sra"
        file_path: Path to data file
        metadata: Optional metadata dict
    """
    sra_ui = SRAIntegrationUI()

    st.markdown("### Data Source Selection")

    source_type = st.radio(
        "Choose Data Source",
        ["Upload Local File", "Select from Storage", "Download from NCBI SRA"],
        help="Select where your sequence data comes from",
    )

    file_path = None
    metadata = None

    if source_type == "Upload Local File":
        uploaded_file = st.file_uploader(
            "Upload FASTA/FASTQ File", type=["fasta", "fa", "fastq", "fq", "gz"]
        )
        if uploaded_file:
            import tempfile

            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                file_path = Path(tmp.name)
                metadata = {"source": "upload", "filename": uploaded_file.name}

    elif source_type == "Select from Storage":
        datasets_dir = Path(config.get("storage.datasets_dir", "data/datasets"))
        if datasets_dir.exists():
            files = (
                list(datasets_dir.glob("*.fasta"))
                + list(datasets_dir.glob("*.fa"))
                + list(datasets_dir.glob("*.fastq"))
                + list(datasets_dir.glob("*.fq"))
                + list(datasets_dir.glob("*.fastq.gz"))
                + list(datasets_dir.glob("*.fq.gz"))
            )

            if files:
                selected_file = st.selectbox(
                    "Select Dataset", files, format_func=lambda x: x.name
                )
                file_path = selected_file
                metadata = {"source": "storage", "filename": selected_file.name}
            else:
                st.warning("No datasets found in storage.")
        else:
            st.warning("Datasets directory not found.")

    elif source_type == "Download from NCBI SRA":
        sra_ui.show_sra_toolkit_status()

        if sra_ui.sra_toolkit_available:
            # Show SRA browser
            selected_accession = sra_ui.show_sra_browser()

            # If accession was selected (stored in session state)
            if "selected_sra_accession" in st.session_state:
                accession = st.session_state.selected_sra_accession
                sra_metadata = st.session_state.get("selected_sra_metadata", {})

                st.markdown(f"**Selected:** {accession}")

                # Download interface
                output_dir = Path("data/sra") / accession
                downloaded_file = sra_ui.show_sra_download_ui(accession, output_dir)

                if downloaded_file:
                    file_path = downloaded_file
                    metadata = {
                        "source": "sra",
                        "accession": accession,
                        "sra_metadata": sra_metadata,
                    }

    return source_type, file_path, metadata
