"""
Data acquisition utilities for downloading reference databases and sample eDNA datasets
"""

import os
import sys
import requests
import gzip
import shutil
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin
import subprocess
import logging
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Handles downloading of reference databases and sample datasets"""
    
    def __init__(self):
        self.reference_dir = Path(config.get('data.reference_dir', 'data/reference'))
        self.raw_dir = Path(config.get('data.raw_dir', 'data/raw'))
        
        # Create directories
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, destination: Path, extract: bool = False) -> bool:
        """
        Download a file with progress bar
        
        Args:
            url: URL to download from
            destination: Local path to save file
            extract: Whether to extract if it's a compressed file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading {url} to {destination}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract if needed
            if extract and destination.suffix in ['.gz', '.bz2']:
                self._extract_file(destination)
            
            logger.info(f"Successfully downloaded {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    def _extract_file(self, file_path: Path) -> None:
        """Extract compressed files"""
        logger.info(f"Extracting {file_path}")
        
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rb') as f_in:
                with open(file_path.with_suffix(''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_path.unlink()  # Remove compressed file
    
    def download_blast_db(self, db_name: str) -> bool:
        """
        Download BLAST database using update_blastdb.pl
        
        Args:
            db_name: Name of BLAST database (e.g., 'nt', '16S_ribosomal_RNA')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading BLAST database: {db_name}")
            
            # Check if update_blastdb.pl is available
            result = subprocess.run(['which', 'update_blastdb.pl'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning("update_blastdb.pl not found. Using direct download method.")
                return self._download_blast_db_direct(db_name)
            
            # Use update_blastdb.pl
            cmd = [
                'update_blastdb.pl',
                '--decompress',
                '--source', 'ncbi',
                '--timeout', '300',
                db_name
            ]
            
            result = subprocess.run(cmd, cwd=self.reference_dir, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded BLAST database: {db_name}")
                return True
            else:
                logger.error(f"Error downloading BLAST database: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading BLAST database {db_name}: {e}")
            return False
    
    def _download_blast_db_direct(self, db_name: str) -> bool:
        """
        Direct download of BLAST database files
        
        Args:
            db_name: Name of BLAST database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            base_url = config.get('databases.ncbi_ftp')
            
            # Get list of files for this database
            files_url = urljoin(base_url, f"{db_name}.tar.gz")
            
            destination = self.reference_dir / f"{db_name}.tar.gz"
            
            if self.download_file(files_url, destination):
                # Extract tar.gz
                subprocess.run(['tar', '-xzf', str(destination)], 
                             cwd=self.reference_dir)
                destination.unlink()  # Remove tar file
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in direct download of {db_name}: {e}")
            return False
    
    def download_silva_db(self) -> bool:
        """Download SILVA reference database"""
        try:
            silva_config = config.get('databases.silva')
            base_url = silva_config['url']
            
            for filename in silva_config['files']:
                url = urljoin(base_url, filename)
                destination = self.reference_dir / filename
                
                if not self.download_file(url, destination, extract=True):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading SILVA database: {e}")
            return False
    
    def download_greengenes_db(self) -> bool:
        """Download Greengenes reference database"""
        try:
            gg_config = config.get('databases.greengenes')
            base_url = gg_config['url']
            
            for filename in gg_config['files']:
                url = urljoin(base_url, filename)
                destination = self.reference_dir / filename
                
                if not self.download_file(url, destination, extract=True):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading Greengenes database: {e}")
            return False
    
    def download_sample_data(self) -> bool:
        """Download sample eDNA datasets for testing"""
        try:
            # Sample datasets from public repositories
            sample_datasets = [
                {
                    'name': 'marine_edna_sample_1.fastq.gz',
                    'url': 'https://www.ebi.ac.uk/ena/portal/api/filereport?accession=ERR3063408&result=read_run&fields=fastq_ftp',
                    'description': 'Marine eDNA sample from ENA'
                }
            ]
            
            for dataset in sample_datasets:
                logger.info(f"Downloading sample dataset: {dataset['name']}")
                destination = self.raw_dir / dataset['name']
                
                # Note: This is a placeholder - actual URLs would need to be determined
                # based on available public datasets
                logger.info(f"Sample dataset placeholder: {dataset['description']}")
            
            # Create a mock sample dataset for testing
            self._create_mock_sample()
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading sample data: {e}")
            return False
    
    def _create_mock_sample(self) -> None:
        """Create a mock eDNA dataset for testing purposes"""
        mock_sequences = [
            "@seq1\nACGTACGTACGTACGTACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "@seq2\nTGCATGCATGCATGCATGCATGCATGCATGCA\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
            "@seq3\nGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCC\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
        ]
        
        mock_file = self.raw_dir / "mock_edna_sample.fastq"
        with open(mock_file, 'w') as f:
            f.write('\n'.join(mock_sequences))
        
        logger.info(f"Created mock sample dataset: {mock_file}")

def main():
    """Main function for downloading data"""
    downloader = DataDownloader()
    
    logger.info("Starting data acquisition...")
    
    # Download reference databases
    blast_dbs = config.get('databases.blast_dbs', [])
    
    for db_name in blast_dbs:
        logger.info(f"Downloading BLAST database: {db_name}")
        downloader.download_blast_db(db_name)
    
    # Download SILVA database
    logger.info("Downloading SILVA database...")
    downloader.download_silva_db()
    
    # Download Greengenes database
    logger.info("Downloading Greengenes database...")
    downloader.download_greengenes_db()
    
    # Download sample data
    logger.info("Downloading sample datasets...")
    downloader.download_sample_data()
    
    logger.info("Data acquisition complete!")

if __name__ == "__main__":
    main()