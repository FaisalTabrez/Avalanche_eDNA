#!/usr/bin/env python3
"""
Universal Dataset Analysis Script for eDNA Biodiversity Assessment System

This script provides a command-line interface for analyzing various biological
sequence datasets using the integrated analysis framework.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.dataset_analyzer import DatasetAnalyzer


def main():
    """Main entry point for dataset analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze biological sequence datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a FASTA file
  python analyze_dataset.py data/sequences.fasta results/analysis_report.txt
  
  # Analyze with custom name
  python analyze_dataset.py data/edna_samples.fastq.gz results/edna_analysis.txt --name "eDNA Samples"
  
  # Analyze protein sequences with format override
  python analyze_dataset.py data/proteins.gz results/protein_analysis.txt --format fasta
  
  # Quick analysis of subset
  python analyze_dataset.py data/large_dataset.fasta results/quick_test.txt --max 1000
  
Supported formats:
  - FASTA (.fasta, .fa, .fas)
  - FASTQ (.fastq, .fq) 
  - Swiss-Prot (.swiss, .sp)
  - GenBank (.gb, .gbk)
  - EMBL (.embl, .em)
  - Gzipped versions of any above
        """
    )
    
    parser.add_argument(
        'input_file', 
        help='Input dataset file (supports various biological sequence formats)'
    )
    parser.add_argument(
        'output_file', 
        help='Output analysis report file (text format)'
    )
    parser.add_argument(
        '--name', 
        help='Custom name for the dataset (default: filename)'
    )
    parser.add_argument(
        '--format', 
        choices=['fasta', 'fastq', 'swiss', 'genbank', 'embl'],
        help='Force specific format instead of auto-detection'
    )
    parser.add_argument(
        '--max', 
        type=int, 
        help='Maximum number of sequences to analyze (for testing large files)'
    )
    parser.add_argument(
        '--config', 
        help='Path to custom configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    try:
        # Initialize analyzer
        if args.verbose:
            print("Initializing dataset analyzer...")
        
        analyzer = DatasetAnalyzer(args.config)
        
        # Run analysis
        results = analyzer.analyze_dataset(
            input_path=args.input_file,
            output_path=args.output_file,
            dataset_name=args.name,
            format_type=args.format,
            max_sequences=args.max
        )
        
        # Print summary
        stats = results.get('basic_stats', {})
        comp = results.get('composition', {})
        proc = results.get('processing_info', {})
        
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Input file: {args.input_file}")
        print(f"Output report: {args.output_file}")
        print(f"Format detected: {proc.get('format', 'unknown')}")
        print(f"File size: {proc.get('file_size_mb', 0):.2f} MB")
        print(f"Sequences analyzed: {stats.get('total_sequences', 0):,}")
        print(f"Mean length: {stats.get('mean_length', 0):.1f}")
        print(f"Sequence type: {comp.get('sequence_type', 'unknown')}")
        print(f"Processing time: {proc.get('total_time', 0):.2f} seconds")
        print("="*50)
        print("Analysis completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())