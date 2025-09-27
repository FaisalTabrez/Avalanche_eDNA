"""
Universal Dataset Analyzer for eDNA Biodiversity Assessment System

This module provides a unified interface for analyzing various biological sequence datasets
including eDNA, protein sequences, and other biological data formats.
"""

import os
import gzip
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from collections import Counter, defaultdict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.utils.config import config
except ImportError:
    # Fallback configuration if config module not available
    class Config:
        def get(self, key, default=None):
            return default
    config = Config()

# Import enhanced analyzers
try:
    from .advanced_taxonomic_analyzer import AdvancedTaxonomicAnalyzer
    from .environmental_context_analyzer import EnvironmentalContextAnalyzer
    from .enhanced_diversity_analyzer import EnhancedDiversityAnalyzer
except ImportError:
    # Fallback if enhanced analyzers not available
    AdvancedTaxonomicAnalyzer = None
    EnvironmentalContextAnalyzer = None
    EnhancedDiversityAnalyzer = None


class DatasetAnalyzer:
    """
    Universal analyzer for biological sequence datasets.
    
    Supports multiple input formats and provides comprehensive analysis
    with optimized performance for large datasets.
    """
    
    def __init__(self, config_path: Optional[str] = None, fast_mode: bool = False):
        """
        Initialize the dataset analyzer.
        
        Args:
            config_path: Optional path to configuration file
            fast_mode: Enable fast analysis mode for large datasets
        """
        self.config = config
        self.fast_mode = fast_mode
        self.n_workers = min(cpu_count(), 8)  # Limit workers to avoid memory issues
        
        # Initialize enhanced analyzers
        self.advanced_taxonomic_analyzer = AdvancedTaxonomicAnalyzer() if AdvancedTaxonomicAnalyzer else None
        self.environmental_context_analyzer = EnvironmentalContextAnalyzer() if EnvironmentalContextAnalyzer else None
        self.enhanced_diversity_analyzer = EnhancedDiversityAnalyzer() if EnhancedDiversityAnalyzer else None
        
        # Optimize worker count based on system resources
        if fast_mode:
            self.n_workers = min(cpu_count(), 16)  # Use more workers in fast mode
        
        self.supported_formats = {
            'fasta': ['fasta', 'fa', 'fas'],
            'fastq': ['fastq', 'fq'],
            'swiss': ['swiss', 'sp'],
            'genbank': ['gb', 'gbk'],
            'embl': ['embl', 'em']
        }
        
        # Fast mode sampling thresholds
        self.fast_mode_thresholds = {
            'large_dataset': 10000,      # Start sampling at 10K sequences
            'very_large_dataset': 50000, # Aggressive sampling at 50K+ sequences
            'composition_sample': 5000,  # Sample size for composition analysis
            'annotation_sample': 3000,   # Sample size for annotation analysis
            'quality_sample': 2000       # Sample size for quality analysis
        }
    
    def detect_format(self, file_path: str) -> str:
        """
        Auto-detect file format based on extension and content.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Detected format string
        """
        path = Path(file_path)
        
        # Check for gzipped files
        if path.suffix == '.gz':
            extension = path.stem.split('.')[-1].lower()
        else:
            extension = path.suffix[1:].lower()
        
        # Map extension to format
        for fmt, exts in self.supported_formats.items():
            if extension in exts:
                return fmt
        
        # Try to detect from content if extension is ambiguous
        try:
            if file_path.endswith('.gz'):
                opener = gzip.open
                mode = 'rt'
            else:
                opener = open
                mode = 'r'
                
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with opener(file_path, mode, encoding=encoding) as handle:
                        first_line = str(handle.readline().strip())
                        
                        if first_line.startswith('>'):
                            return 'fasta'
                        elif first_line.startswith('@'):
                            return 'fastq'
                        elif first_line.startswith('ID   '):
                            return 'swiss'
                        elif first_line.startswith('LOCUS'):
                            return 'genbank'
                        break
                except UnicodeDecodeError:
                    continue
                    
        except Exception:
            pass
            
        return 'fasta'  # Default fallback
    
    def load_sequences(self, file_path: str, 
                      format_type: Optional[str] = None,
                      max_sequences: Optional[int] = None) -> List[SeqRecord]:
        """
        Load sequences from file with auto-format detection.
        
        Args:
            file_path: Path to input file
            format_type: Override auto-detection with specific format
            max_sequences: Limit number of sequences loaded
            
        Returns:
            List of SeqRecord objects
        """
        if format_type is None:
            format_type = self.detect_format(file_path)
        
        print(f"Loading sequences from: {file_path}")
        print(f"Detected format: {format_type}")
        
        sequences = []
        count = 0
        
        try:
            # Handle gzipped files with encoding detection
            if file_path.endswith('.gz'):
                opener = gzip.open
                mode = 'rt'
            else:
                opener = open
                mode = 'r'
                
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            sequences = []
            
            for encoding in encodings:
                try:
                    with opener(file_path, mode, encoding=encoding) as handle:
                        # Use more robust FASTA format for better compatibility
                        parse_format = format_type
                        if format_type == 'fasta':
                            parse_format = 'fasta-blast'  # Handles comments and is more robust
                        
                        for record in SeqIO.parse(handle, parse_format):
                            sequences.append(record)
                            count += 1
                            
                            if max_sequences and count >= max_sequences:
                                print(f"Limited to {max_sequences} sequences")
                                break
                                
                            # Progress reporting optimized for fast mode
                            if self.fast_mode:
                                if count % 100000 == 0:  # Less frequent updates in fast mode
                                    print(f"   Loaded {count:,} sequences...")
                            else:
                                if count % 50000 == 0:
                                    print(f"   Loaded {count:,} sequences...")
                    
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                    
                except UnicodeDecodeError as e:
                    print(f"Failed with {encoding} encoding: {str(e)}")
                    if encoding == encodings[-1]:  # Last encoding attempt
                        raise
                    continue
                except Exception as e:
                    # Try alternative FASTA formats if parsing fails
                    if format_type == 'fasta' and 'fasta-2line' not in str(e):
                        try:
                            print(f"Trying alternative FASTA format...")
                            with opener(file_path, mode, encoding=encoding) as handle:
                                for record in SeqIO.parse(handle, 'fasta'):
                                    sequences.append(record)
                                    count += 1
                                    
                                    if max_sequences and count >= max_sequences:
                                        print(f"Limited to {max_sequences} sequences")
                                        break
                                        
                                    if count % 50000 == 0:
                                        print(f"   Loaded {count:,} sequences...")
                            print(f"Successfully loaded with {encoding} encoding (standard FASTA)")
                            break
                        except Exception as e2:
                            print(f"Alternative FASTA format also failed: {str(e2)}")
                    
                    print(f"Error with {encoding}: {str(e)}")
                    raise
                        
            print(f"Loaded {len(sequences):,} sequences")
            
        except Exception as e:
            print(f"Failed to load sequences: {str(e)}")
            raise
            
        return sequences
    
    def calculate_basic_stats(self, sequences: List[SeqRecord]) -> Dict[str, Any]:
        """
        Calculate basic sequence statistics using vectorized operations.
        
        Args:
            sequences: List of sequence records
            
        Returns:
            Dictionary of basic statistics
        """
        print("Calculating basic sequence statistics...")
        
        lengths = np.array([len(seq) for seq in sequences], dtype=np.int32)
        
        stats = {
            'total_sequences': len(sequences),
            'min_length': int(lengths.min()) if len(lengths) > 0 else 0,
            'max_length': int(lengths.max()) if len(lengths) > 0 else 0,
            'mean_length': float(lengths.mean()) if len(lengths) > 0 else 0.0,
            'median_length': float(np.median(lengths)) if len(lengths) > 0 else 0.0,
            'std_length': float(lengths.std()) if len(lengths) > 0 else 0.0,
            'length_percentiles': {
                '25th': float(np.percentile(lengths, 25)) if len(lengths) > 0 else 0.0,
                '75th': float(np.percentile(lengths, 75)) if len(lengths) > 0 else 0.0,
                '90th': float(np.percentile(lengths, 90)) if len(lengths) > 0 else 0.0,
                '95th': float(np.percentile(lengths, 95)) if len(lengths) > 0 else 0.0
            }
        }
        
        return stats
    
    def safe_sequence_str(self, seq_record):
        """
        Safely extract sequence string with proper encoding handling.
        
        Args:
            seq_record: BioPython SeqRecord object
            
        Returns:
            String representation of the sequence
        """
        try:
            # Try direct string conversion first
            return str(seq_record.seq).upper()
        except UnicodeDecodeError:
            try:
                # If direct conversion fails, try accessing the raw data with different encodings
                if hasattr(seq_record.seq, '_data'):
                    # Try different encodings for the raw data
                    raw_data = seq_record.seq._data
                    if isinstance(raw_data, bytes):
                        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                return raw_data.decode(encoding).upper()
                            except (UnicodeDecodeError, UnicodeError):
                                continue
                    else:
                        # If it's already a string, return it
                        return str(raw_data).upper()
                
                # Fallback: try to extract sequence character by character
                seq_chars = []
                for i in range(len(seq_record.seq)):
                    try:
                        char = str(seq_record.seq[i])
                        seq_chars.append(char)
                    except (UnicodeDecodeError, UnicodeError):
                        # Replace problematic characters with 'N'
                        seq_chars.append('N')
                return ''.join(seq_chars).upper()
                
            except Exception:
                # Ultimate fallback: return empty string
                print(f"   Warning: Could not extract sequence data, skipping problematic sequence")
                return ""
        except Exception as e:
            print(f"   Warning: Unexpected error extracting sequence: {str(e)}")
            return ""

    def analyze_composition(self, sequences: List[SeqRecord], 
                          sequence_type: str = 'auto') -> Dict[str, Any]:
        """
        Analyze sequence composition (nucleotide or amino acid) with optimized performance.
        
        Args:
            sequences: List of sequence records
            sequence_type: 'dna', 'rna', 'protein', or 'auto'
            
        Returns:
            Composition analysis results
        """
        print("Analyzing sequence composition...")
        
        if sequence_type == 'auto':
            # Auto-detect sequence type from first few sequences
            sample_chars = set()
            detection_sample = min(100, len(sequences))
            for seq in sequences[:detection_sample]:
                seq_str = self.safe_sequence_str(seq)
                if seq_str:  # Only process if we could extract the sequence
                    sample_chars.update(seq_str)
            
            # Check for amino acid characters
            aa_chars = set('ARNDCQEGHILKMFPSTWYVBZX*')
            nt_chars = set('ATCGUNRYSWKMBDHV')
            
            if sample_chars - nt_chars:  # Has non-nucleotide characters
                sequence_type = 'protein'
            else:
                sequence_type = 'dna'
        
        print(f"   Detected sequence type: {sequence_type}")
        
        # Smart sampling strategy based on dataset size and fast mode
        if self.fast_mode and len(sequences) > self.fast_mode_thresholds['large_dataset']:
            sample_size = self.fast_mode_thresholds['composition_sample']
            print(f"   Fast mode: Analyzing {sample_size:,} representative sequences from {len(sequences):,} total...")
            
            # Stratified sampling for better representation
            step = max(1, len(sequences) // sample_size)
            sample_sequences = sequences[::step][:sample_size]
        elif len(sequences) > 50000:
            # Standard large dataset sampling
            print(f"   Large dataset detected ({len(sequences):,} sequences). Using sampling for efficiency...")
            sample_size = min(10000, len(sequences) // 2)
            step = len(sequences) // sample_size
            sample_sequences = sequences[::step][:sample_size]
            print(f"   Analyzing {len(sample_sequences):,} representative sequences...")
        else:
            sample_sequences = sequences
        
        # Optimized batch processing
        total_counter = Counter()
        
        # Dynamic batch sizing based on dataset size and mode
        if self.fast_mode:
            batch_size = min(2000, len(sample_sequences))  # Larger batches in fast mode
        else:
            batch_size = min(1000, len(sample_sequences))
        
        num_batches = (len(sample_sequences) + batch_size - 1) // batch_size
        
        print(f"   Processing {len(sample_sequences):,} sequences in {num_batches} batches (batch size: {batch_size})...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sample_sequences))
            batch = sample_sequences[start_idx:end_idx]
            
            # Process batch with optimized character extraction
            batch_counter = Counter()
            for seq in batch:
                seq_str = self.safe_sequence_str(seq)
                if seq_str:  # Only process if we could extract the sequence
                    batch_counter.update(seq_str)
            
            total_counter.update(batch_counter)
            
            # Progress update (less frequent in fast mode)
            progress_interval = 2 if self.fast_mode else 4
            if num_batches > progress_interval and batch_idx % max(1, num_batches // progress_interval) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{num_batches} batches)")
        
        # Calculate frequencies
        total_chars = sum(total_counter.values())
        if total_chars == 0:
            print("   Warning: No characters found in sequences!")
            return {
                'sequence_type': sequence_type,
                'composition': {},
                'total_characters': 0,
                'unique_characters': 0,
                'most_common': []
            }
        
        composition = {char: count/total_chars for char, count in total_counter.items()}
        
        # Sort by frequency
        composition = dict(sorted(composition.items(), key=lambda x: x[1], reverse=True))
        
        print(f"   Composition analysis complete. Found {len(total_counter)} unique characters.")
        
        # Add sampling info to results
        result = {
            'sequence_type': sequence_type,
            'composition': composition,
            'total_characters': total_chars,
            'unique_characters': len(total_counter),
            'most_common': total_counter.most_common(10)
        }
        
        if len(sample_sequences) < len(sequences):
            result['sampling_info'] = {
                'sampled_sequences': len(sample_sequences),
                'total_sequences': len(sequences),
                'sampling_ratio': len(sample_sequences) / len(sequences)
            }
        
        return result
    
    def analyze_annotations(self, sequences: List[SeqRecord]) -> Dict[str, Any]:
        """
        Analyze sequence annotations and metadata with fast mode optimization.
        
        Args:
            sequences: List of sequence records
            
        Returns:
            Annotation analysis results
        """
        print("Analyzing sequence annotations...")
        
        # Smart sampling for large datasets in fast mode
        if self.fast_mode and len(sequences) > self.fast_mode_thresholds['large_dataset']:
            sample_size = self.fast_mode_thresholds['annotation_sample']
            step = max(1, len(sequences) // sample_size)
            sample_sequences = sequences[::step][:sample_size]
            print(f"   Fast mode: Analyzing {len(sample_sequences):,} representative sequences from {len(sequences):,} total...")
        else:
            sample_sequences = sequences
        
        organism_counts = Counter()
        description_patterns = Counter()
        feature_counts = defaultdict(int)
        
        # Optimized batch processing
        if self.fast_mode:
            batch_size = 2000  # Larger batches in fast mode
        else:
            batch_size = 1000
        
        num_batches = (len(sample_sequences) + batch_size - 1) // batch_size
        
        sequences_with_organisms = 0
        sequences_with_descriptions = 0
        sequences_with_features = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sample_sequences))
            batch = sample_sequences[start_idx:end_idx]
            
            for seq in batch:
                # Organism information
                organism = seq.annotations.get('organism', 'Unknown')
                organism_counts[organism] += 1
                if organism != 'Unknown':
                    sequences_with_organisms += 1
                
                # Description patterns
                if seq.description:
                    sequences_with_descriptions += 1
                    desc_lower = seq.description.lower()
                    for pattern in ['hypothetical', 'putative', 'uncharacterized', 
                                   'fragment', 'precursor', 'partial']:
                        if pattern in desc_lower:
                            description_patterns[pattern] += 1
                
                # Features (skip in fast mode for very large datasets)
                if not (self.fast_mode and len(sequences) > self.fast_mode_thresholds['very_large_dataset']):
                    if hasattr(seq, 'features') and seq.features:
                        sequences_with_features += 1
                        for feature in seq.features:
                            feature_counts[feature.type] += 1
            
            # Progress update (less frequent in fast mode)
            progress_interval = 2 if self.fast_mode else 4
            if num_batches > progress_interval and batch_idx % max(1, num_batches // progress_interval) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"   Progress: {progress:.1f}% ({batch_idx + 1}/{num_batches} batches)")
        
        # Scale up results if sampling was used
        scaling_factor = len(sequences) / len(sample_sequences) if len(sample_sequences) < len(sequences) else 1
        
        if scaling_factor > 1:
            sequences_with_organisms = int(sequences_with_organisms * scaling_factor)
            sequences_with_descriptions = int(sequences_with_descriptions * scaling_factor)
            sequences_with_features = int(sequences_with_features * scaling_factor)
            print(f"   Results scaled by factor {scaling_factor:.2f} for full dataset estimation")
        
        print(f"   Annotation analysis complete. Processed {len(sample_sequences):,} sequences.")
        
        result = {
            'organism_distribution': dict(organism_counts.most_common(20)),
            'description_patterns': dict(description_patterns.most_common(10)),
            'feature_types': dict(feature_counts),
            'sequences_with_organisms': sequences_with_organisms,
            'sequences_with_descriptions': sequences_with_descriptions,
            'sequences_with_features': sequences_with_features
        }
        
        if len(sample_sequences) < len(sequences):
            result['sampling_info'] = {
                'sampled_sequences': len(sample_sequences),
                'total_sequences': len(sequences),
                'scaling_factor': scaling_factor
            }
        
        return result
    
    def analyze_quality(self, sequences: List[SeqRecord]) -> Dict[str, Any]:
        """
        Analyze sequence quality metrics with fast mode optimization.
        
        Args:
            sequences: List of sequence records
            
        Returns:
            Quality analysis results
        """
        print("Analyzing sequence quality...")
        
        # Smart sampling for quality analysis in fast mode
        if self.fast_mode and len(sequences) > self.fast_mode_thresholds['large_dataset']:
            sample_size = self.fast_mode_thresholds['quality_sample']
            step = max(1, len(sequences) // sample_size)
            sample_sequences = sequences[::step][:sample_size]
            print(f"   Fast mode: Analyzing {len(sample_sequences):,} representative sequences for quality...")
        else:
            sample_sequences = sequences
        
        quality_stats = {
            'sequences_with_quality': 0,
            'mean_quality': 0.0,
            'quality_distribution': {},
            'low_quality_sequences': 0
        }
        
        quality_scores = []
        
        for seq in sample_sequences:
            if hasattr(seq, 'letter_annotations') and 'phred_quality' in seq.letter_annotations:
                quality_stats['sequences_with_quality'] += 1
                qual_scores = seq.letter_annotations['phred_quality']
                mean_qual = np.mean(qual_scores)
                quality_scores.append(mean_qual)
                
                if mean_qual < 20:  # Low quality threshold
                    quality_stats['low_quality_sequences'] += 1
        
        # Scale up results if sampling was used
        scaling_factor = len(sequences) / len(sample_sequences) if len(sample_sequences) < len(sequences) else 1
        
        if scaling_factor > 1:
            quality_stats['sequences_with_quality'] = int(quality_stats['sequences_with_quality'] * scaling_factor)
            quality_stats['low_quality_sequences'] = int(quality_stats['low_quality_sequences'] * scaling_factor)
        
        if quality_scores:
            quality_stats['mean_quality'] = float(np.mean(quality_scores))
            quality_stats['quality_distribution'] = {
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores)),
                'median': float(np.median(quality_scores)),
                'std': float(np.std(quality_scores))
            }
        
        if len(sample_sequences) < len(sequences):
            quality_stats['sampling_info'] = {
                'sampled_sequences': len(sample_sequences),
                'total_sequences': len(sequences),
                'scaling_factor': scaling_factor
            }
        
        return quality_stats
    
    def calculate_diversity_metrics(self, sequences: List[SeqRecord]) -> Dict[str, float]:
        """
        Calculate biodiversity metrics.
        
        Args:
            sequences: List of sequence records
            
        Returns:
            Diversity metrics
        """
        print("Calculating biodiversity metrics...")
        
        # Use organism information if available, otherwise use sequence similarity
        organisms = [seq.annotations.get('organism', 'Unknown') for seq in sequences]
        organism_counts = Counter(organisms)
        
        # Shannon diversity index
        total = sum(organism_counts.values())
        proportions = [count/total for count in organism_counts.values()]
        shannon_diversity = -sum(p * np.log(p) for p in proportions if p > 0)
        
        # Simpson diversity index
        simpson_diversity = 1 - sum(p**2 for p in proportions)
        
        # Evenness
        num_species = len(organism_counts)
        evenness = shannon_diversity / np.log(num_species) if num_species > 1 else 0
        
        return {
            'shannon_diversity': float(shannon_diversity),
            'simpson_diversity': float(simpson_diversity),
            'evenness': float(evenness),
            'species_richness': num_species,
            'total_abundance': total
        }
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       output_path: str, dataset_name: str = "Dataset") -> None:
        """
        Generate comprehensive analysis report.
        
        Args:
            analysis_results: Combined analysis results
            output_path: Path to save the report
            dataset_name: Name of the dataset
        """
        print("Generating analysis report...")
        
        lines = []
        lines.append(f"# {dataset_name} Analysis Report")
        lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Basic Statistics
        if 'basic_stats' in analysis_results:
            stats = analysis_results['basic_stats']
            lines.append("## 📊 Basic Sequence Statistics")
            lines.append(f"- Total sequences: {stats['total_sequences']:,}")
            lines.append(f"- Minimum length: {stats['min_length']:,}")
            lines.append(f"- Maximum length: {stats['max_length']:,}")
            lines.append(f"- Mean length: {stats['mean_length']:.2f}")
            lines.append(f"- Median length: {stats['median_length']:.2f}")
            lines.append(f"- Standard deviation: {stats['std_length']:.2f}")
            lines.append("")
            
            # Length percentiles
            percs = stats['length_percentiles']
            lines.append("### Length Distribution Percentiles")
            lines.append(f"- 25th percentile: {percs['25th']:.2f}")
            lines.append(f"- 75th percentile: {percs['75th']:.2f}")
            lines.append(f"- 90th percentile: {percs['90th']:.2f}")
            lines.append(f"- 95th percentile: {percs['95th']:.2f}")
            lines.append("")
        
        # Composition Analysis
        if 'composition' in analysis_results:
            comp = analysis_results['composition']
            lines.append(f"## 🧬 Sequence Composition ({comp['sequence_type'].upper()})")
            lines.append(f"- Total characters analyzed: {comp['total_characters']:,}")
            lines.append(f"- Unique characters: {comp['unique_characters']}")
            lines.append("")
            
            lines.append("### Character Frequencies")
            lines.append("| Character | Frequency | Percentage |")
            lines.append("|-----------|-----------|------------|")
            for char, freq in list(comp['composition'].items())[:15]:  # Top 15
                lines.append(f"| {char} | {freq:.6f} | {freq*100:.3f}% |")
            lines.append("")
        
        # Annotation Analysis
        if 'annotations' in analysis_results:
            ann = analysis_results['annotations']
            lines.append("## 📝 Annotation Analysis")
            lines.append(f"- Sequences with organism info: {ann['sequences_with_organisms']:,}")
            lines.append(f"- Sequences with descriptions: {ann['sequences_with_descriptions']:,}")
            lines.append(f"- Sequences with features: {ann['sequences_with_features']:,}")
            lines.append("")
            
            # Organism distribution
            if ann['organism_distribution']:
                lines.append("### Top Organisms")
                lines.append("| Organism | Count |")
                lines.append("|----------|-------|")
                for org, count in list(ann['organism_distribution'].items())[:10]:
                    lines.append(f"| {org} | {count:,} |")
                lines.append("")
            
            # Description patterns
            if ann['description_patterns']:
                lines.append("### Description Patterns")
                lines.append("| Pattern | Count |")
                lines.append("|---------|-------|")
                for pattern, count in ann['description_patterns'].items():
                    lines.append(f"| {pattern} | {count:,} |")
                lines.append("")
        
        # Quality Analysis
        if 'quality' in analysis_results:
            qual = analysis_results['quality']
            lines.append("## 🔍 Quality Analysis")
            lines.append(f"- Sequences with quality scores: {qual['sequences_with_quality']:,}")
            lines.append(f"- Low quality sequences (Q<20): {qual['low_quality_sequences']:,}")
            if qual['sequences_with_quality'] > 0:
                lines.append(f"- Mean quality score: {qual['mean_quality']:.2f}")
                qual_dist = qual['quality_distribution']
                lines.append(f"- Quality range: {qual_dist['min']:.2f} - {qual_dist['max']:.2f}")
            lines.append("")
        
        # Diversity Metrics
        if 'diversity' in analysis_results:
            div = analysis_results['diversity']
            lines.append("## 🌿 Biodiversity Metrics")
            lines.append(f"- Species richness: {div['species_richness']:,}")
            lines.append(f"- Shannon diversity index: {div['shannon_diversity']:.4f}")
            lines.append(f"- Simpson diversity index: {div['simpson_diversity']:.4f}")
            lines.append(f"- Evenness: {div['evenness']:.4f}")
            lines.append(f"- Total abundance: {div['total_abundance']:,}")
            lines.append("")
        
        # Enhanced Diversity Analysis
        if 'enhanced_diversity' in analysis_results:
            self._add_enhanced_diversity_section(lines, analysis_results['enhanced_diversity'])
        
        # Advanced Taxonomic Analysis
        if 'advanced_taxonomic' in analysis_results:
            self._add_advanced_taxonomic_section(lines, analysis_results['advanced_taxonomic'])
        
        # Environmental Context Analysis
        if 'environmental_context' in analysis_results:
            self._add_environmental_context_section(lines, analysis_results['environmental_context'])
        
        # Processing Information
        if 'processing_info' in analysis_results:
            proc = analysis_results['processing_info']
            lines.append("## ⏱️ Processing Information")
            lines.append(f"- Total processing time: {proc['total_time']:.2f} seconds")
            lines.append(f"- File format: {proc['format']}")
            lines.append(f"- File size: {proc['file_size_mb']:.2f} MB")
            if 'step_times' in proc:
                lines.append("")
                lines.append("### Step-by-step timing:")
                for step, time_taken in proc['step_times'].items():
                    lines.append(f"- {step}: {time_taken:.2f}s")
        
        # Save report
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Report saved to: {output_path}")
    
    def _add_enhanced_diversity_section(self, lines: List[str], diversity_data: Dict[str, Any]) -> None:
        """Add enhanced diversity analysis section to report."""
        lines.append("## 🔬 Enhanced Diversity Analysis")
        
        # Alpha diversity metrics
        if 'alpha_diversity' in diversity_data:
            alpha = diversity_data['alpha_diversity']
            lines.append("### Alpha Diversity Metrics")
            lines.append(f"- Chao1 estimator: {alpha.get('chao1', 0):.2f}")
            lines.append(f"- ACE estimator: {alpha.get('ace', 0):.2f}")
            lines.append(f"- Fisher's alpha: {alpha.get('fisher_alpha', 0):.2f}")
            lines.append(f"- Effective species number: {alpha.get('effective_species_number', 0):.2f}")
            lines.append(f"- Berger-Parker index: {alpha.get('berger_parker', 0):.4f}")
            lines.append("")
        
        # Community structure
        if 'community_structure' in diversity_data:
            community = diversity_data['community_structure']
            lines.append("### Community Structure")
            lines.append(f"- Dominant species (>5%): {community.get('dominant_species_count', 0)}")
            lines.append(f"- Rare species (<1%): {community.get('rare_species_count', 0)}")
            lines.append(f"- Evenness category: {community.get('community_evenness_category', 'unknown')}")
            lines.append("")
        
        # Sampling adequacy
        if 'sampling_adequacy' in diversity_data:
            adequacy = diversity_data['sampling_adequacy']
            lines.append("### Sampling Adequacy")
            lines.append(f"- Completeness: {adequacy.get('sampling_completeness', 0):.2%}")
            lines.append(f"- Adequacy level: {adequacy.get('adequacy_level', 'unknown')}")
            if adequacy.get('additional_sampling_needed', False):
                lines.append("- ⚠️ Additional sampling recommended")
            lines.append("")
    
    def _add_advanced_taxonomic_section(self, lines: List[str], taxonomic_data: Dict[str, Any]) -> None:
        """Add advanced taxonomic analysis section to report."""
        lines.append("## 🦠 Advanced Taxonomic Analysis")
        
        # Confidence assessment
        if 'confidence_assessment' in taxonomic_data:
            confidence = taxonomic_data['confidence_assessment']
            lines.append("### Assignment Confidence")
            lines.append(f"- High confidence assignments: {confidence.get('high_confidence_count', 0)}")
            lines.append(f"- Medium confidence assignments: {confidence.get('medium_confidence_count', 0)}")
            lines.append(f"- Low confidence assignments: {confidence.get('low_confidence_count', 0)}")
            lines.append(f"- Average confidence: {confidence.get('average_confidence', 0):.2%}")
            lines.append("")
        
        # Novel candidate detection
        if 'novel_candidates' in taxonomic_data:
            novel_candidates = taxonomic_data['novel_candidates']
            lines.append("### Novel Species Candidates")
            
            # Handle novel_candidates as a list (which it actually is)
            if isinstance(novel_candidates, list):
                candidate_count = len(novel_candidates)
                lines.append(f"- Potential novel species: {candidate_count}")
                
                # Calculate percentage if we have assignment data
                total_assignments = taxonomic_data.get('total_sequences', 0)
                if total_assignments > 0:
                    novel_percentage = (candidate_count / total_assignments) * 100
                    lines.append(f"- Novel candidate percentage: {novel_percentage:.1f}%")
                else:
                    lines.append("- Novel candidate percentage: N/A")
                
                if candidate_count > 0:
                    lines.append("✨ Novel species candidates detected - consider further investigation")
                    
                    # Add details about top candidates
                    if candidate_count > 0:
                        lines.append("")
                        lines.append("**Top Novel Candidates:**")
                        for i, candidate in enumerate(novel_candidates[:5], 1):  # Top 5
                            # Additional safety check for each candidate
                            if isinstance(candidate, dict):
                                novelty_score = candidate.get('novelty_score', 0)
                                potential_rank = candidate.get('potential_rank', 'unknown')
                                lines.append(f"{i}. {candidate.get('sequence_id', 'Unknown')} (novelty: {novelty_score:.3f}, rank: {potential_rank})")
                            else:
                                lines.append(f"{i}. Invalid candidate format")
            
            # Fallback for unexpected data structure
            elif isinstance(novel_candidates, dict):
                candidate_count = novel_candidates.get('candidate_count', 0)
                lines.append(f"- Potential novel species: {candidate_count}")
                lines.append(f"- Novel candidate percentage: {novel_candidates.get('novel_percentage', 0):.1f}%")
                if candidate_count > 0:
                    lines.append("✨ Novel species candidates detected - consider further investigation")
            
            lines.append("")
        
        # Phylogenetic patterns
        if 'phylogenetic_patterns' in taxonomic_data:
            phylo = taxonomic_data['phylogenetic_patterns']
            lines.append("### Phylogenetic Diversity")
            lines.append(f"- Phylum diversity: {phylo.get('phylum_diversity', 0)}")
            lines.append(f"- Order diversity: {phylo.get('order_diversity', 0)}")
            lines.append(f"- Family diversity: {phylo.get('family_diversity', 0)}")
            lines.append("")
    
    def _add_environmental_context_section(self, lines: List[str], env_data: Dict[str, Any]) -> None:
        """Add environmental context analysis section to report."""
        lines.append("## 🌍 Environmental Context Analysis")
        
        # Habitat classification
        if 'habitat_classification' in env_data:
            habitat = env_data['habitat_classification']
            lines.append("### Habitat Classification")
            lines.append(f"- Primary habitat: {habitat.get('primary_habitat', 'unknown')}")
            lines.append(f"- Confidence: {habitat.get('classification_confidence', 0):.2%}")
            lines.append(f"- Habitat complexity: {habitat.get('habitat_complexity', 'unknown')}")
            lines.append("")
        
        # Ecological indicators
        if 'ecological_indicators' in env_data:
            indicators = env_data['ecological_indicators']
            lines.append("### Ecological Indicators")
            if indicators.get('indicator_species'):
                lines.append(f"- Indicator species detected: {len(indicators['indicator_species'])}")
            lines.append(f"- Ecosystem health: {indicators.get('ecosystem_health_score', 'unknown')}")
            lines.append(f"- Pollution indicators: {indicators.get('pollution_indicators', 'none detected')}")
            lines.append("")
        
        # Environmental stress
        if 'environmental_stress_indicators' in env_data:
            stress = env_data['environmental_stress_indicators']
            lines.append("### Environmental Stress Assessment")
            lines.append(f"- Stress level: {stress.get('stress_level', 'unknown')}")
            if stress.get('stress_indicators'):
                lines.append(f"- Stress indicators detected: {len(stress['stress_indicators'])}")
            lines.append("")
    
    def analyze_dataset(self, input_path: str, output_path: str,
                       dataset_name: Optional[str] = None,
                       format_type: Optional[str] = None,
                       max_sequences: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            input_path: Path to input dataset file
            output_path: Path to save analysis report
            dataset_name: Name for the dataset (auto-generated if None)
            format_type: Override format detection
            max_sequences: Limit analysis to N sequences
            
        Returns:
            Complete analysis results dictionary
        """
        start_time = time.time()
        step_times = {}
        
        if dataset_name is None:
            dataset_name = Path(input_path).stem
        
        print(f"Starting analysis of dataset: {dataset_name}")
        print(f"Input file: {input_path}")
        print(f"Output report: {output_path}")
        
        # Get file info
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        results = {}
        
        # Step 1: Load sequences
        step_start = time.time()
        sequences = self.load_sequences(input_path, format_type, max_sequences)
        step_times['sequence_loading'] = time.time() - step_start
        
        if not sequences:
            raise ValueError("No sequences loaded from file")
        
        # Step 2: Basic statistics
        step_start = time.time()
        results['basic_stats'] = self.calculate_basic_stats(sequences)
        step_times['basic_statistics'] = time.time() - step_start
        
        # Step 3: Composition analysis
        step_start = time.time()
        results['composition'] = self.analyze_composition(sequences)
        step_times['composition_analysis'] = time.time() - step_start
        
        # Step 4: Annotation analysis
        step_start = time.time()
        results['annotations'] = self.analyze_annotations(sequences)
        step_times['annotation_analysis'] = time.time() - step_start
        
        # Step 5: Quality analysis
        step_start = time.time()
        results['quality'] = self.analyze_quality(sequences)
        step_times['quality_analysis'] = time.time() - step_start
        
        # Step 6: Diversity metrics
        step_start = time.time()
        results['diversity'] = self.calculate_diversity_metrics(sequences)
        step_times['diversity_metrics'] = time.time() - step_start
        
        # Step 7: Enhanced Taxonomic Analysis
        if self.advanced_taxonomic_analyzer:
            try:
                step_start = time.time()
                results['advanced_taxonomic'] = self.advanced_taxonomic_analyzer.analyze_taxonomic_composition(sequences)
                step_times['advanced_taxonomic_analysis'] = time.time() - step_start
            except Exception as e:
                print(f"Enhanced taxonomic analysis failed: {str(e)}")
                results['advanced_taxonomic'] = {}
        
        # Step 8: Enhanced Diversity Analysis
        if self.enhanced_diversity_analyzer:
            try:
                step_start = time.time()
                # Create taxonomic data from annotations if available
                taxonomic_data = self._prepare_taxonomic_data(results.get('annotations', {}))
                results['enhanced_diversity'] = self.enhanced_diversity_analyzer.analyze_comprehensive_diversity(taxonomic_data)
                step_times['enhanced_diversity_analysis'] = time.time() - step_start
            except Exception as e:
                print(f"Enhanced diversity analysis failed: {str(e)}")
                results['enhanced_diversity'] = {}
        
        # Step 9: Environmental Context Analysis
        if self.environmental_context_analyzer:
            try:
                step_start = time.time()
                # Extract environmental data from sequence metadata
                environmental_data = self._extract_environmental_data(sequences)
                taxonomic_data = self._prepare_taxonomic_data(results.get('annotations', {}))
                results['environmental_context'] = self.environmental_context_analyzer.analyze_environmental_context(environmental_data, taxonomic_data)
                step_times['environmental_context_analysis'] = time.time() - step_start
            except Exception as e:
                print(f"Environmental context analysis failed: {str(e)}")
                results['environmental_context'] = {}
        
        # Processing information
        total_time = time.time() - start_time
        results['processing_info'] = {
            'total_time': total_time,
            'format': format_type or self.detect_format(input_path),
            'file_size_mb': file_size_mb,
            'step_times': step_times,
            'sequences_analyzed': len(sequences)
        }
        
        # Generate report
        step_start = time.time()
        self.generate_report(results, output_path, dataset_name)
        step_times['report_generation'] = time.time() - step_start
        
        print(f"Analysis complete!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Analyzed {len(sequences):,} sequences")
        
        return results
    
    def _prepare_taxonomic_data(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare taxonomic data from annotation results."""
        taxonomic_data = {
            'taxonomic_distribution': {
                'genus': {
                    'counts': {},
                    'unique_taxa': 0
                },
                'species': {
                    'counts': {},
                    'unique_taxa': 0
                }
            }
        }
        
        # Extract organism distribution from annotations
        if 'organism_distribution' in annotations:
            organism_dist = annotations['organism_distribution']
            
            # Convert organism names to taxonomic counts
            genus_counts = {}
            species_counts = {}
            
            for organism, count in organism_dist.items():
                # Ensure count is not None and is a valid integer
                if count is None:
                    count = 0
                count = int(count) if count is not None else 0
                
                # Simple parsing of organism names
                parts = organism.replace('_', ' ').split()
                if len(parts) >= 2:
                    genus = parts[0]
                    species = f"{parts[0]} {parts[1]}"
                    genus_counts[genus] = genus_counts.get(genus, 0) + count
                    species_counts[species] = species_counts.get(species, 0) + count
                elif len(parts) == 1:
                    genus_counts[parts[0]] = genus_counts.get(parts[0], 0) + count
            
            # Filter out None values and ensure positive counts
            genus_counts = {k: v for k, v in genus_counts.items() if v is not None and v > 0}
            species_counts = {k: v for k, v in species_counts.items() if v is not None and v > 0}
            
            taxonomic_data['taxonomic_distribution']['genus']['counts'] = genus_counts
            taxonomic_data['taxonomic_distribution']['genus']['unique_taxa'] = len(genus_counts)
            taxonomic_data['taxonomic_distribution']['species']['counts'] = species_counts
            taxonomic_data['taxonomic_distribution']['species']['unique_taxa'] = len(species_counts)
        
        return taxonomic_data
    
    def _extract_environmental_data(self, sequences: List[SeqRecord]) -> Dict[str, Any]:
        """Extract environmental data from sequence metadata."""
        environmental_data = {
            'sample_type': 'unknown',
            'habitat': 'unknown',
            'depth': None,
            'temperature': None,
            'salinity': None,
            'ph': None,
            'location': 'unknown',
            'collection_date': None
        }
        
        # Extract environmental info from sequence descriptions
        habitat_keywords = {
            'marine': ['marine', 'ocean', 'sea', 'seawater'],
            'freshwater': ['freshwater', 'lake', 'river', 'pond'],
            'terrestrial': ['soil', 'terrestrial', 'land'],
            'brackish': ['brackish', 'estuary', 'estuarine']
        }
        
        habitat_counts = {}
        
        for seq in sequences[:100]:  # Sample first 100 sequences
            description = str(seq.description).lower()
            
            for habitat, keywords in habitat_keywords.items():
                if any(keyword in description for keyword in keywords):
                    habitat_counts[habitat] = habitat_counts.get(habitat, 0) + 1
        
        # Set most common habitat
        if habitat_counts:
            most_common_habitat = max(habitat_counts.items(), key=lambda x: x[1])[0]
            environmental_data['habitat'] = most_common_habitat
            environmental_data['sample_type'] = most_common_habitat
        
        return environmental_data


def main():
    """Command-line interface for dataset analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal Dataset Analyzer for Biological Sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_analyzer.py data.fasta output_report.txt
  python dataset_analyzer.py data.fastq.gz report.txt --name "My Dataset"
  python dataset_analyzer.py swissprot.gz analysis.txt --format fasta --max 10000
        """
    )
    
    parser.add_argument('input_file', help='Input dataset file')
    parser.add_argument('output_file', help='Output analysis report file')
    parser.add_argument('--name', help='Dataset name for report')
    parser.add_argument('--format', help='Force specific format (fasta, fastq, swiss, etc.)')
    parser.add_argument('--max', type=int, help='Maximum number of sequences to analyze')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        analyzer = DatasetAnalyzer(args.config)
        analyzer.analyze_dataset(
            input_path=args.input_file,
            output_path=args.output_file,
            dataset_name=args.name,
            format_type=args.format,
            max_sequences=args.max
        )
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())