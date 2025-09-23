"""
Data preprocessing pipeline for eDNA sequences
Handles quality filtering, adapter trimming, and chimera removal
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import subprocess
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceQualityFilter:
    """Handles quality filtering of DNA sequences"""
    
    def __init__(self, 
                 min_length: int = 50,
                 max_length: int = 500,
                 quality_threshold: int = 20,
                 max_n_bases: int = 5):
        """
        Initialize quality filter
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            quality_threshold: Minimum average quality score
            max_n_bases: Maximum number of N bases allowed
        """
        self.min_length = min_length
        self.max_length = max_length
        self.quality_threshold = quality_threshold
        self.max_n_bases = max_n_bases
    
    def filter_sequence(self, record) -> bool:
        """
        Filter a single sequence record
        
        Args:
            record: BioPython sequence record
            
        Returns:
            True if sequence passes filters, False otherwise
        """
        seq = str(record.seq)
        
        # Length filter
        if len(seq) < self.min_length or len(seq) > self.max_length:
            return False
        
        # N base filter
        if seq.count('N') > self.max_n_bases:
            return False
        
        # Quality filter (if quality scores available)
        if hasattr(record, 'letter_annotations') and 'phred_quality' in record.letter_annotations:
            avg_quality = np.mean(record.letter_annotations['phred_quality'])
            if avg_quality < self.quality_threshold:
                return False
        
        return True
    
    def filter_fastq(self, input_file: Path, output_file: Path) -> Dict[str, int]:
        """
        Filter FASTQ file
        
        Args:
            input_file: Input FASTQ file
            output_file: Output filtered FASTQ file
            
        Returns:
            Dictionary with filtering statistics
        """
        stats = {
            'total_sequences': 0,
            'passed_length': 0,
            'passed_quality': 0,
            'passed_n_filter': 0,
            'final_passed': 0
        }
        
        logger.info(f"Filtering {input_file} -> {output_file}")
        
        with open(output_file, 'w') as out_handle:
            for record in tqdm(SeqIO.parse(input_file, 'fastq'), desc="Filtering sequences"):
                stats['total_sequences'] += 1
                
                seq = str(record.seq)
                
                # Track individual filter results
                length_pass = self.min_length <= len(seq) <= self.max_length
                n_pass = seq.count('N') <= self.max_n_bases
                quality_pass = True
                
                if hasattr(record, 'letter_annotations') and 'phred_quality' in record.letter_annotations:
                    avg_quality = np.mean(record.letter_annotations['phred_quality'])
                    quality_pass = avg_quality >= self.quality_threshold
                
                if length_pass:
                    stats['passed_length'] += 1
                if quality_pass:
                    stats['passed_quality'] += 1
                if n_pass:
                    stats['passed_n_filter'] += 1
                
                # Final filter decision
                if self.filter_sequence(record):
                    SeqIO.write(record, out_handle, 'fastq')
                    stats['final_passed'] += 1
        
        logger.info(f"Filtering complete. {stats['final_passed']}/{stats['total_sequences']} sequences passed")
        return stats

class AdapterTrimmer:
    """Handles adapter trimming using cutadapt"""
    
    def __init__(self, adapters: List[str], min_length: int = 50):
        """
        Initialize adapter trimmer
        
        Args:
            adapters: List of adapter sequences
            min_length: Minimum length after trimming
        """
        self.adapters = adapters
        self.min_length = min_length
    
    def trim_adapters(self, input_file: Path, output_file: Path) -> bool:
        """
        Trim adapters from sequences
        
        Args:
            input_file: Input FASTQ file
            output_file: Output trimmed FASTQ file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Trimming adapters from {input_file}")
            
            # Build cutadapt command
            cmd = [
                'cutadapt',
                '--minimum-length', str(self.min_length),
                '--quality-cutoff', '20',
                '--times', '2',  # Remove adapters up to 2 times
                '--output', str(output_file)
            ]
            
            # Add adapter sequences
            for adapter in self.adapters:
                cmd.extend(['-a', adapter])
            
            cmd.append(str(input_file))
            
            # Run cutadapt
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Adapter trimming successful: {output_file}")
                return True
            else:
                logger.error(f"Cutadapt error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error trimming adapters: {e}")
            return False

class ChimeraDetector:
    """Handles chimera detection and removal"""
    
    def __init__(self, reference_db: Optional[Path] = None, method: str = "vsearch"):
        """
        Initialize chimera detector
        
        Args:
            reference_db: Path to reference database for chimera detection
            method: Method to use ('vsearch', 'uchime')
        """
        self.reference_db = reference_db
        self.method = method
    
    def detect_chimeras(self, input_file: Path, output_file: Path) -> bool:
        """
        Detect and remove chimeric sequences
        
        Args:
            input_file: Input FASTA file
            output_file: Output non-chimeric FASTA file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Detecting chimeras in {input_file}")
            
            if self.method == "vsearch":
                return self._vsearch_chimera_detection(input_file, output_file)
            else:
                logger.error(f"Unsupported chimera detection method: {self.method}")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting chimeras: {e}")
            return False
    
    def _vsearch_chimera_detection(self, input_file: Path, output_file: Path) -> bool:
        """Use VSEARCH for chimera detection"""
        try:
            chimera_file = input_file.parent / f"{input_file.stem}_chimeras.fasta"
            
            cmd = [
                'vsearch',
                '--uchime_denovo', str(input_file),
                '--chimeras', str(chimera_file),
                '--nonchimeras', str(output_file),
                '--threads', str(mp.cpu_count())
            ]
            
            # Add reference database if available
            if self.reference_db and self.reference_db.exists():
                cmd.extend(['--db', str(self.reference_db)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Chimera detection successful: {output_file}")
                # Clean up chimera file
                if chimera_file.exists():
                    chimera_file.unlink()
                return True
            else:
                logger.error(f"VSEARCH error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error in VSEARCH chimera detection: {e}")
            return False

class PreprocessingPipeline:
    """Main preprocessing pipeline for eDNA sequences"""
    
    def __init__(self):
        """Initialize preprocessing pipeline with configuration"""
        self.config = config
        
        # Initialize components
        preprocessing_config = self.config.get('preprocessing', {})
        
        self.quality_filter = SequenceQualityFilter(
            min_length=preprocessing_config.get('min_length', 50),
            max_length=preprocessing_config.get('max_length', 500),
            quality_threshold=preprocessing_config.get('quality_threshold', 20),
            max_n_bases=preprocessing_config.get('quality_filter', {}).get('max_n_bases', 5)
        )
        
        self.adapter_trimmer = AdapterTrimmer(
            adapters=preprocessing_config.get('adapter_sequences', []),
            min_length=preprocessing_config.get('min_length', 50)
        )
        
        chimera_config = preprocessing_config.get('chimera_detection', {})
        reference_db = chimera_config.get('reference_db')
        
        self.chimera_detector = ChimeraDetector(
            reference_db=Path(reference_db) if reference_db else None,
            method=chimera_config.get('method', 'vsearch')
        )
        
        # Setup directories
        self.raw_dir = Path(self.config.get('data.raw_dir', 'data/raw'))
        self.processed_dir = Path(self.config.get('data.processed_dir', 'data/processed'))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, input_file: Path, output_prefix: str) -> Dict[str, Any]:
        """
        Process a single eDNA file through the complete pipeline
        
        Args:
            input_file: Input FASTQ file
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {input_file}")
        
        stats = {
            'input_file': str(input_file),
            'processing_steps': []
        }
        
        # Step 1: Adapter trimming
        trimmed_file = self.processed_dir / f"{output_prefix}_trimmed.fastq"
        if self.adapter_trimmer.trim_adapters(input_file, trimmed_file):
            stats['processing_steps'].append('adapter_trimming')
        else:
            logger.error(f"Adapter trimming failed for {input_file}")
            return stats
        
        # Step 2: Quality filtering
        filtered_file = self.processed_dir / f"{output_prefix}_filtered.fastq"
        filter_stats = self.quality_filter.filter_fastq(trimmed_file, filtered_file)
        stats['filtering_stats'] = filter_stats
        stats['processing_steps'].append('quality_filtering')
        
        # Step 3: Convert to FASTA for chimera detection
        fasta_file = self.processed_dir / f"{output_prefix}_filtered.fasta"
        self._fastq_to_fasta(filtered_file, fasta_file)
        
        # Step 4: Chimera detection
        final_file = self.processed_dir / f"{output_prefix}_final.fasta"
        if self.chimera_detector.detect_chimeras(fasta_file, final_file):
            stats['processing_steps'].append('chimera_detection')
        else:
            logger.warning(f"Chimera detection failed for {input_file}, using filtered sequences")
            final_file = fasta_file
        
        # Clean up intermediate files
        if trimmed_file.exists():
            trimmed_file.unlink()
        if fasta_file != final_file and fasta_file.exists():
            fasta_file.unlink()
        
        stats['final_file'] = str(final_file)
        stats['completed'] = True
        
        logger.info(f"Processing complete for {input_file} -> {final_file}")
        return stats
    
    def _fastq_to_fasta(self, fastq_file: Path, fasta_file: Path) -> None:
        """Convert FASTQ to FASTA format"""
        with open(fasta_file, 'w') as out_handle:
            for record in SeqIO.parse(fastq_file, 'fastq'):
                SeqIO.write(record, out_handle, 'fasta')
    
    def process_directory(self, input_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Process all FASTQ files in a directory
        
        Args:
            input_dir: Directory containing FASTQ files. If None, uses configured raw_dir
            
        Returns:
            List of processing statistics for each file
        """
        if input_dir is None:
            input_dir = self.raw_dir
        
        input_dir = Path(input_dir)
        
        # Find all FASTQ files
        fastq_files = list(input_dir.glob("*.fastq")) + list(input_dir.glob("*.fastq.gz"))
        
        if not fastq_files:
            logger.warning(f"No FASTQ files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(fastq_files)} FASTQ files to process")
        
        results = []
        for fastq_file in fastq_files:
            output_prefix = fastq_file.stem.replace('.fastq', '')
            
            try:
                stats = self.process_file(fastq_file, output_prefix)
                results.append(stats)
            except Exception as e:
                logger.error(f"Error processing {fastq_file}: {e}")
                results.append({
                    'input_file': str(fastq_file),
                    'error': str(e),
                    'completed': False
                })
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate processing report
        
        Args:
            results: List of processing results
            
        Returns:
            DataFrame with processing summary
        """
        report_data = []
        
        for result in results:
            row = {
                'file': Path(result['input_file']).name,
                'completed': result.get('completed', False),
                'steps_completed': len(result.get('processing_steps', [])),
                'final_file': result.get('final_file', 'N/A')
            }
            
            # Add filtering statistics if available
            if 'filtering_stats' in result:
                stats = result['filtering_stats']
                row.update({
                    'total_sequences': stats.get('total_sequences', 0),
                    'final_sequences': stats.get('final_passed', 0),
                    'pass_rate': stats.get('final_passed', 0) / max(stats.get('total_sequences', 1), 1)
                })
            
            if 'error' in result:
                row['error'] = result['error']
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)

def main():
    """Main function for running preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="eDNA Sequence Preprocessing Pipeline")
    parser.add_argument('--input', type=str, help="Input directory containing FASTQ files")
    parser.add_argument('--output', type=str, help="Output directory for processed files")
    parser.add_argument('--config', type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline()
    
    # Override directories if specified
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = None
    
    if args.output:
        pipeline.processed_dir = Path(args.output)
        pipeline.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    logger.info("Starting eDNA preprocessing pipeline...")
    results = pipeline.process_directory(input_dir)
    
    # Generate report
    report = pipeline.generate_report(results)
    
    # Save report
    report_file = pipeline.processed_dir / "preprocessing_report.csv"
    report.to_csv(report_file, index=False)
    
    logger.info(f"Preprocessing complete! Report saved to {report_file}")
    print("\nProcessing Summary:")
    print(report.to_string(index=False))

if __name__ == "__main__":
    main()