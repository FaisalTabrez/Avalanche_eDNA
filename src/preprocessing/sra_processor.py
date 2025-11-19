"""
SRA Data Processing Module

This module handles the integration of NCBI SRA data with the eDNA analysis pipeline.
It provides functionality to:
1. Process downloaded SRA data
2. Extract eDNA-relevant sequences
3. Integrate with existing preprocessing pipeline
"""

import logging
import gzip
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import config
from .pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

class SRAProcessor:
    """Process NCBI SRA data for eDNA analysis"""

    def __init__(self):
        """Initialize SRA processor"""
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.sra_config = config.get('databases', {}).get('sra', {})

        logger.info("SRA Processor initialized")

    def process_sra_fastq(self, fastq_path: Path, output_dir: Optional[Path] = None) -> List[SeqRecord]:
        """
        Process FASTQ file from SRA for eDNA analysis

        Args:
            fastq_path: Path to FASTQ file
            output_dir: Output directory for processed data

        Returns:
            List of processed sequence records
        """
        if output_dir is None:
            output_dir = fastq_path.parent / "processed"

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing SRA FASTQ: {fastq_path}")

        # Load sequences
        sequences = []
        try:
            if fastq_path.suffix == '.gz':
                with gzip.open(fastq_path, 'rt') as handle:
                    for record in SeqIO.parse(handle, 'fastq'):
                        sequences.append(record)
            else:
                for record in SeqIO.parse(fastq_path, 'fastq'):
                    sequences.append(record)

            logger.info(f"Loaded {len(sequences)} sequences from {fastq_path}")

        except Exception as e:
            logger.error(f"Error loading FASTQ file: {e}")
            return []

        # Apply eDNA-specific filtering
        filtered_sequences = self._filter_edna_sequences(sequences)

        # Apply standard preprocessing
        processed_records = []
        for i, seq in enumerate(filtered_sequences):
            # Create new SeqRecord with eDNA-specific metadata
            metadata = {
                'source': 'SRA',
                'original_file': str(fastq_path),
                'sra_accession': fastq_path.parent.name,
                'sequence_id': f"SRA_{fastq_path.parent.name}_{i}"
            }

            # Add quality information if available
            if hasattr(seq, 'letter_annotations') and 'phred_quality' in seq.letter_annotations:
                metadata['mean_quality'] = float(np.mean(seq.letter_annotations['phred_quality']))

            processed_record = SeqRecord(
                seq=seq.seq,
                id=metadata['sequence_id'],
                description=f"SRA eDNA sequence from {fastq_path.parent.name}",
                annotations=metadata
            )

            processed_records.append(processed_record)

        # Save processed sequences
        output_file = output_dir / f"processed_{fastq_path.stem}.fasta"
        SeqIO.write(processed_records, output_file, 'fasta')

        logger.info(f"Processed {len(processed_records)} eDNA sequences, saved to {output_file}")

        return processed_records

    def _filter_edna_sequences(self, sequences: List[SeqRecord]) -> List[SeqRecord]:
        """
        Apply eDNA-specific filtering to sequences

        Args:
            sequences: List of sequence records

        Returns:
            Filtered list of sequences
        """
        filtered = []

        for seq in sequences:
            # Length filtering
            seq_len = len(seq)
            min_length = self.config.get('preprocessing', {}).get('min_length', 50)
            max_length = self.config.get('preprocessing', {}).get('max_length', 500)

            if seq_len < min_length or seq_len > max_length:
                continue

            # Quality filtering (if quality scores available)
            if hasattr(seq, 'letter_annotations') and 'phred_quality' in seq.letter_annotations:
                quality_scores = seq.letter_annotations['phred_quality']
                mean_quality = np.mean(quality_scores)

                if mean_quality < 20:  # Low quality threshold
                    continue

            # Ambiguity filtering
            seq_str = str(seq.seq).upper()
            n_count = seq_str.count('N')

            if n_count / seq_len > 0.1:  # More than 10% ambiguous bases
                continue

            filtered.append(seq)

        logger.info(f"Filtered {len(sequences)} sequences to {len(filtered)} eDNA-relevant sequences")
        return filtered

    def extract_edna_markers(self, sequences: List[SeqRecord],
                           marker_genes: Optional[List[str]] = None) -> Dict[str, List[SeqRecord]]:
        """
        Extract sequences containing eDNA marker genes

        Args:
            sequences: List of sequence records
            marker_genes: List of marker gene names to search for

        Returns:
            Dictionary mapping marker genes to matching sequences
        """
        if marker_genes is None:
            marker_genes = ['18S', '16S', 'COI', '12S', 'ITS']

        marker_sequences = {gene: [] for gene in marker_genes}

        for seq in sequences:
            seq_str = str(seq.seq).upper()
            description = str(seq.description).upper()

            for gene in marker_genes:
                if gene.upper() in description or gene.upper() in seq_str[:100]:
                    marker_sequences[gene].append(seq)
                    break

        # Log results
        for gene, seqs in marker_sequences.items():
            if seqs:
                logger.info(f"Found {len(seqs)} sequences containing {gene} marker")

        return marker_sequences

    def create_sra_analysis_report(self, sra_accession: str,
                                  processed_sequences: List[SeqRecord],
                                  output_dir: Path) -> None:
        """
        Create comprehensive analysis report for SRA data

        Args:
            sra_accession: SRA accession number
            processed_sequences: List of processed sequences
            output_dir: Output directory for report
        """
        report = {
            'sra_accession': sra_accession,
            'total_sequences': len(processed_sequences),
            'processing_date': pd.Timestamp.now().isoformat(),
            'statistics': {}
        }

        if processed_sequences:
            # Basic statistics
            lengths = [len(seq) for seq in processed_sequences]
            report['statistics'] = {
                'min_length': int(np.min(lengths)),
                'max_length': int(np.max(lengths)),
                'mean_length': float(np.mean(lengths)),
                'median_length': float(np.median(lengths)),
                'total_bases': sum(lengths)
            }

            # Sequence composition
            all_bases = ''.join(str(seq.seq) for seq in processed_sequences)
            base_counts = {}
            for base in 'ATCGN':
                base_counts[base] = all_bases.count(base)

            report['composition'] = base_counts

            # Quality statistics (if available)
            quality_scores = []
            for seq in processed_sequences:
                if hasattr(seq, 'annotations') and 'mean_quality' in seq.annotations:
                    quality_scores.append(seq.annotations['mean_quality'])

            if quality_scores:
                report['quality_statistics'] = {
                    'mean_quality': float(np.mean(quality_scores)),
                    'min_quality': float(np.min(quality_scores)),
                    'max_quality': float(np.max(quality_scores))
                }

        # Save report
        report_file = output_dir / f"sra_analysis_report_{sra_accession}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"SRA analysis report saved to: {report_file}")

    def integrate_with_pipeline(self, sra_fastq_files: List[Path],
                               output_dir: Optional[Path] = None) -> Dict[str, List[SeqRecord]]:
        """
        Integrate SRA data processing with main analysis pipeline

        Args:
            sra_fastq_files: List of SRA FASTQ file paths
            output_dir: Output directory for integrated results

        Returns:
            Dictionary containing processed sequences and metadata
        """
        if output_dir is None:
            output_dir = Path("results/sra_integration")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Integrating {len(sra_fastq_files)} SRA files with analysis pipeline")

        all_processed_sequences = []
        sra_metadata = {}

        for fastq_file in sra_fastq_files:
            sra_accession = fastq_file.parent.name

            # Process SRA FASTQ
            processed_seqs = self.process_sra_fastq(fastq_file, output_dir / sra_accession)

            if processed_seqs:
                all_processed_sequences.extend(processed_seqs)

                # Extract marker genes
                marker_sequences = self.extract_edna_markers(processed_seqs)

                # Create analysis report
                self.create_sra_analysis_report(sra_accession, processed_seqs, output_dir)

                sra_metadata[sra_accession] = {
                    'processed_sequences': len(processed_seqs),
                    'marker_genes': {gene: len(seqs) for gene, seqs in marker_sequences.items()},
                    'file_path': str(fastq_file)
                }

        # Save integrated results
        integrated_results = {
            'total_sra_files': len(sra_fastq_files),
            'total_processed_sequences': len(all_processed_sequences),
            'sra_metadata': sra_metadata,
            'output_directory': str(output_dir)
        }

        results_file = output_dir / "sra_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(integrated_results, f, indent=2, default=str)

        logger.info(f"SRA integration complete. Processed {len(all_processed_sequences)} sequences from {len(sra_fastq_files)} files")

        return {
            'sequences': all_processed_sequences,
            'metadata': sra_metadata,
            'results': integrated_results
        }

def main():
    """Command-line interface for SRA processing"""
    import argparse

    parser = argparse.ArgumentParser(description="Process SRA data for eDNA analysis")

    parser.add_argument('fastq_file', help="Path to SRA FASTQ file")
    parser.add_argument('--output-dir', type=str, help="Output directory")
    parser.add_argument('--batch-process', nargs='+', help="Process multiple FASTQ files")

    args = parser.parse_args()

    processor = SRAProcessor()

    try:
        if args.batch_process:
            fastq_files = [Path(f) for f in args.batch_process]
            results = processor.integrate_with_pipeline(fastq_files, Path(args.output_dir) if args.output_dir else None)
            print(f"Processed {results['results']['total_sra_files']} SRA files")
            print(f"Generated {results['results']['total_processed_sequences']} processed sequences")
        else:
            fastq_path = Path(args.fastq_file)
            sequences = processor.process_sra_fastq(fastq_path, Path(args.output_dir) if args.output_dir else None)
            print(f"Processed {len(sequences)} sequences from {fastq_path}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
