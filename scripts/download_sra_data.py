#!/usr/bin/env python3
"""
NCBI SRA Data Download and Integration Module

This script provides functionality to:
1. Search NCBI SRA for eDNA-relevant datasets
2. Download SRA runs using SRA Toolkit or direct FTP
3. Convert SRA files to FASTQ format
4. Integrate with the eDNA analysis pipeline
"""

import argparse
import json
import logging
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import urllib.request
import urllib.parse
import gzip
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SRADownloader:
    """NCBI SRA data downloader and processor"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize SRA downloader with configuration"""
        self.config = config
        self.sra_config = config.get('databases', {}).get('sra', {})

        # SRA URLs
        self.sra_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.sra_summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self.sra_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # Output directories
        self.sra_data_dir = Path("data/sra")
        self.sra_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SRA Downloader initialized")

    def search_edna_studies(self, keywords: Optional[List[str]] = None,
                          max_results: int = 100) -> List[Dict]:
        """
        Search NCBI SRA for eDNA-relevant studies

        Args:
            keywords: List of search keywords
            max_results: Maximum number of results to return

        Returns:
            List of study metadata dictionaries
        """
        if keywords is None:
            keywords = self.sra_config.get('search', {}).get('edna_keywords', [])

        # Build search query
        search_terms = ['eDNA', 'environmental DNA', 'metabarcoding']
        search_terms.extend(keywords)

        query = ' OR '.join(f'"{term}"' for term in search_terms)
        query += ' AND "environmental"[filter]'

        logger.info(f"Searching SRA for eDNA studies with query: {query}")

        # Search parameters
        params = {
            'db': 'sra',
            'term': query,
            'retmax': str(max_results),
            'retmode': 'json'
        }

        url = f"{self.sra_search_url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())

            study_ids = data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(study_ids)} SRA studies")

            return self._get_study_details(study_ids)

        except Exception as e:
            logger.error(f"Error searching SRA: {e}")
            return []

    def _get_study_details(self, study_ids: List[str]) -> List[Dict]:
        """Get detailed information about studies"""
        studies = []

        # Process in batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(study_ids), batch_size):
            batch_ids = study_ids[i:i+batch_size]

            params = {
                'db': 'sra',
                'id': ','.join(batch_ids),
                'retmode': 'xml'
            }

            url = f"{self.sra_summary_url}?{urllib.parse.urlencode(params)}"

            try:
                with urllib.request.urlopen(url) as response:
                    root = ET.fromstring(response.read())

                for docsum in root.findall('.//DocSum'):
                    study_info = self._parse_study_xml(docsum)
                    if study_info:
                        studies.append(study_info)

            except Exception as e:
                logger.error(f"Error getting study details: {e}")
                continue

        return studies

    def _parse_study_xml(self, docsum) -> Optional[Dict]:
        """Parse study XML to extract relevant information"""
        try:
            study_info = {}

            # Extract basic information
            for item in docsum.findall('.//Item'):
                name = item.get('Name')
                value = item.text

                if name in ['Accession', 'Title', 'Organism', 'Platform', 'Spots', 'Bases']:
                    study_info[name.lower()] = value

            # Filter for eDNA-relevant studies
            title = study_info.get('title', '').lower()
            edna_keywords = self.sra_config.get('search', {}).get('edna_keywords', [])

            is_edna_relevant = any(keyword.lower() in title for keyword in edna_keywords)

            if is_edna_relevant and int(study_info.get('spots', 0)) >= self.sra_config.get('search', {}).get('min_spots', 1000000):
                return study_info

        except Exception as e:
            logger.error(f"Error parsing study XML: {e}")

        return None

    def download_sra_run(self, accession: str, output_dir: Optional[Path] = None) -> bool:
        """
        Download SRA run using SRA Toolkit or direct download

        Args:
            accession: SRA accession number
            output_dir: Output directory for downloaded files

        Returns:
            True if download successful
        """
        if output_dir is None:
            output_dir = self.sra_data_dir / accession

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading SRA run: {accession}")

        # Try using SRA Toolkit first
        if self._download_with_sra_tools(accession, output_dir):
            return True

        # Fallback to direct download
        return self._download_direct(accession, output_dir)

    def _download_with_sra_tools(self, accession: str, output_dir: Path) -> bool:
        """Download using SRA Toolkit"""
        try:
            prefetch_path = self.sra_config.get('sra_tools', {}).get('prefetch_path', 'prefetch')
            fastq_dump_path = self.sra_config.get('sra_tools', {}).get('fastq_dump_path', 'fastq-dump')

            # Prefetch SRA file
            logger.info(f"Prefetching {accession}...")
            result = subprocess.run(
                [prefetch_path, accession, '-O', str(output_dir)],
                capture_output=True, text=True, timeout=600
            )

            if result.returncode != 0:
                logger.warning(f"Prefetch failed: {result.stderr}")
                return False

            # Convert to FASTQ
            sra_file = output_dir / f"{accession}.sra"
            if sra_file.exists():
                logger.info(f"Converting {accession} to FASTQ...")
                fastq_result = subprocess.run(
                    [fastq_dump_path, '--outdir', str(output_dir),
                     '--gzip', '--skip-technical', '--readids',
                     '--read-filter', 'pass', '--dumpbase',
                     '--split-3', '--clip', str(sra_file)],
                    capture_output=True, text=True, timeout=1200
                )

                if fastq_result.returncode == 0:
                    logger.info(f"Successfully downloaded and converted {accession}")
                    return True
                else:
                    logger.error(f"FASTQ conversion failed: {fastq_result.stderr}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"SRA Toolkit method failed: {e}")

        return False

    def _download_direct(self, accession: str, output_dir: Path) -> bool:
        """Direct download from NCBI FTP"""
        try:
            # Construct FTP URL
            ftp_url = f"ftp://ftp.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByRun/sra/{accession[:3]}/{accession[:6]}/{accession}/{accession}.sra"

            logger.info(f"Attempting direct download: {ftp_url}")

            # Download SRA file
            sra_file = output_dir / f"{accession}.sra"
            urllib.request.urlretrieve(ftp_url, sra_file)

            if sra_file.exists():
                logger.info(f"Downloaded SRA file: {sra_file}")
                # Note: Would need SRA Toolkit for conversion to FASTQ
                # This is a limitation of the direct download approach
                return True

        except Exception as e:
            logger.error(f"Direct download failed: {e}")

        return False

    def convert_sra_to_fastq(self, sra_file: Path, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Convert SRA file to FASTQ format

        Args:
            sra_file: Path to SRA file
            output_dir: Output directory for FASTQ files

        Returns:
            List of generated FASTQ file paths
        """
        if output_dir is None:
            output_dir = sra_file.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        fastq_files = []

        try:
            fastq_dump_path = self.sra_config.get('sra_tools', {}).get('fastq_dump_path', 'fastq-dump')

            logger.info(f"Converting {sra_file} to FASTQ...")

            result = subprocess.run(
                [fastq_dump_path, '--outdir', str(output_dir),
                 '--gzip', '--skip-technical', '--readids',
                 '--read-filter', 'pass', '--dumpbase',
                 '--split-3', '--clip', str(sra_file)],
                capture_output=True, text=True, timeout=1800
            )

            if result.returncode == 0:
                # Find generated FASTQ files
                for fastq_file in output_dir.glob("*.fastq.gz"):
                    fastq_files.append(fastq_file)

                logger.info(f"Generated {len(fastq_files)} FASTQ files")
            else:
                logger.error(f"Conversion failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Error converting SRA to FASTQ: {e}")

        return fastq_files

    def download_edna_datasets(self, study_type: str = "marine_sediment",
                              max_downloads: int = 10) -> List[Path]:
        """
        Download eDNA datasets of specified type

        Args:
            study_type: Type of eDNA study ('marine_sediment', 'deep_sea', 'plankton')
            max_downloads: Maximum number of datasets to download

        Returns:
            List of downloaded FASTQ file paths
        """
        studies = self.sra_config.get('edna_studies', {}).get(study_type, [])

        if not studies:
            logger.warning(f"No predefined studies for type: {study_type}")
            return []

        downloaded_files = []
        download_count = 0

        for accession in studies:
            if download_count >= max_downloads:
                break

            logger.info(f"Processing study {accession} ({download_count + 1}/{max_downloads})")

            if self.download_sra_run(accession, self.sra_data_dir / accession):
                # Convert to FASTQ
                sra_file = self.sra_data_dir / accession / f"{accession}.sra"
                if sra_file.exists():
                    fastq_files = self.convert_sra_to_fastq(sra_file)
                    downloaded_files.extend(fastq_files)
                    download_count += 1

            time.sleep(2)  # Rate limiting

        logger.info(f"Downloaded {len(downloaded_files)} FASTQ files from {download_count} SRA runs")
        return downloaded_files

    def search_and_download_relevant_studies(self, max_results: int = 20) -> List[Path]:
        """
        Search for and download relevant eDNA studies

        Args:
            max_results: Maximum number of studies to process

        Returns:
            List of downloaded FASTQ file paths
        """
        logger.info("Searching for relevant eDNA studies...")

        studies = self.search_edna_studies(max_results=max_results)
        downloaded_files = []

        logger.info(f"Found {len(studies)} relevant studies")

        for i, study in enumerate(studies[:max_results]):
            accession = study.get('accession')
            if not accession:
                continue

            logger.info(f"Downloading study {i+1}/{min(max_results, len(studies))}: {accession}")

            if self.download_sra_run(accession, self.sra_data_dir / accession):
                sra_file = self.sra_data_dir / accession / f"{accession}.sra"
                if sra_file.exists():
                    fastq_files = self.convert_sra_to_fastq(sra_file)
                    downloaded_files.extend(fastq_files)

            time.sleep(2)  # Rate limiting

        return downloaded_files

    def create_sra_metadata_report(self, output_path: str) -> None:
        """Create a report of downloaded SRA data"""
        metadata = {
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_studies': 0,
            'total_runs': 0,
            'total_files': 0,
            'studies': []
        }

        # Scan downloaded data
        for study_dir in self.sra_data_dir.iterdir():
            if study_dir.is_dir():
                study_info = {
                    'accession': study_dir.name,
                    'files': [],
                    'total_size': 0
                }

                for file_path in study_dir.rglob('*'):
                    if file_path.is_file():
                        study_info['files'].append({
                            'name': file_path.name,
                            'size_mb': file_path.stat().st_size / (1024 * 1024),
                            'type': file_path.suffix
                        })
                        study_info['total_size'] += file_path.stat().st_size

                if study_info['files']:
                    metadata['studies'].append(study_info)
                    metadata['total_studies'] += 1
                    metadata['total_files'] += len(study_info['files'])

        # Save metadata
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"SRA metadata report saved to: {output_path}")

def main():
    """Main function for SRA data downloading"""
    parser = argparse.ArgumentParser(description="Download eDNA data from NCBI SRA")

    parser.add_argument('--search', action='store_true',
                       help="Search for eDNA-relevant studies")
    parser.add_argument('--download-type', type=str,
                       choices=['marine_sediment', 'deep_sea', 'plankton'],
                       help="Type of eDNA studies to download")
    parser.add_argument('--max-results', type=int, default=10,
                       help="Maximum number of studies to process")
    parser.add_argument('--accession', type=str,
                       help="Download specific SRA accession")
    parser.add_argument('--output-dir', type=str,
                       help="Output directory for downloads")
    parser.add_argument('--report', type=str,
                       help="Generate metadata report")

    args = parser.parse_args()

    downloader = SRADownloader()

    try:
        if args.search:
            logger.info("Searching and downloading relevant eDNA studies...")
            downloaded_files = downloader.search_and_download_relevant_studies(args.max_results)
            logger.info(f"Downloaded {len(downloaded_files)} files")

        elif args.download_type:
            logger.info(f"Downloading {args.download_type} eDNA datasets...")
            downloaded_files = downloader.download_edna_datasets(args.download_type, args.max_results)
            logger.info(f"Downloaded {len(downloaded_files)} files")

        elif args.accession:
            logger.info(f"Downloading SRA accession: {args.accession}")
            success = downloader.download_sra_run(args.accession)
            if success:
                logger.info("Download completed successfully")
            else:
                logger.error("Download failed")
                return 1

        if args.report:
            downloader.create_sra_metadata_report(args.report)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
