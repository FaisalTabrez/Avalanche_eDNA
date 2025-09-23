# Universal Dataset Analyzer

## 🎯 Overview

The Universal Dataset Analyzer is a new integrated system for the eDNA Biodiversity Assessment project that provides comprehensive analysis of biological sequence datasets. Instead of creating separate analysis scripts for each dataset type, this system offers a unified interface that can handle multiple input formats and generate standardized analysis reports.

## 🚀 Key Features

### ✅ **Unified Input Interface**
- **Single command** for all dataset types
- **Auto-format detection** (FASTA, FASTQ, Swiss-Prot, GenBank, EMBL)
- **Gzipped file support** for compressed datasets
- **Flexible input options** with format override capabilities

### ✅ **Comprehensive Analysis**
- **Basic sequence statistics** (length distribution, percentiles)
- **Composition analysis** (auto-detects DNA/RNA/protein sequences)
- **Annotation mining** (organism distribution, description patterns)
- **Quality assessment** (for FASTQ files with quality scores)
- **Biodiversity metrics** (Shannon/Simpson diversity, evenness)

### ✅ **Performance Optimized**
- **Parallel processing** for composition analysis
- **Vectorized calculations** using NumPy
- **Memory-efficient streaming** for large files
- **Progress indicators** for long-running analyses
- **Subset testing** capability for large datasets

### ✅ **Standardized Output**
- **Text report format** (.txt files)
- **Consistent structure** across all dataset types
- **Detailed timing information** for performance monitoring
- **Processing metadata** included in reports

## 📋 Usage Examples

### Basic Analysis
```bash
# Analyze any supported biological sequence file
python scripts/analyze_dataset.py input_file.fasta output_report.txt
```

### Advanced Options
```bash
# With custom dataset name
python scripts/analyze_dataset.py data.fastq.gz report.txt --name "My Dataset"

# Force specific format (override auto-detection)
python scripts/analyze_dataset.py sequences.gz report.txt --format fasta

# Quick test with subset of sequences
python scripts/analyze_dataset.py large_file.fasta test_report.txt --max 1000

# Verbose output for debugging
python scripts/analyze_dataset.py data.fasta report.txt --verbose
```

## 🔄 Real-World Examples

### Example 1: Swiss-Prot Protein Database
```bash
# Analyze full Swiss-Prot database (482,697 sequences)
python scripts/analyze_dataset.py data/raw/swissprot.gz results/swissprot_analysis.txt --name "Swiss-Prot Complete Database"

# Results: 14.39 seconds processing time
# Output: Comprehensive protein composition and annotation analysis
```

### Example 2: eDNA Sequences
```bash
# Analyze environmental DNA samples
python scripts/analyze_dataset.py data/sample/sample_edna_sequences.fasta results/edna_analysis.txt --name "eDNA Sample Sequences"

# Results: 0.07 seconds processing time
# Output: DNA composition analysis with nucleotide frequencies
```

### Example 3: Large Dataset Testing
```bash
# Quick analysis of subset for validation
python scripts/analyze_dataset.py huge_dataset.fasta quick_test.txt --max 5000
```

## 📊 Supported File Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| **FASTA** | `.fasta`, `.fa`, `.fas` | Most common sequence format |
| **FASTQ** | `.fastq`, `.fq` | Sequences with quality scores |
| **Swiss-Prot** | `.swiss`, `.sp` | Protein database format |
| **GenBank** | `.gb`, `.gbk` | NCBI GenBank format |
| **EMBL** | `.embl`, `.em` | EMBL database format |
| **Compressed** | `.gz` extension | Gzipped versions of any above |

## 📈 Performance Comparison

### Old Approach (Separate Scripts)
- ❌ New script needed for each dataset type
- ❌ Inconsistent output formats
- ❌ Code duplication and maintenance overhead
- ❌ Different optimization levels

### New Unified System
- ✅ **Single script** handles all formats
- ✅ **Consistent analysis** across dataset types
- ✅ **Optimized performance** with parallel processing
- ✅ **Standardized reports** for easy comparison

### Performance Results
| Dataset | Sequences | Processing Time | Format |
|---------|-----------|----------------|--------|
| Swiss-Prot (full) | 482,697 | 14.39s | Protein FASTA |
| Swiss-Prot (subset) | 5,000 | 0.20s | Protein FASTA |
| eDNA Samples | 1,000 | 0.07s | DNA FASTA |

## 🏗️ System Architecture

### Integration with eDNA Project
The analyzer is fully integrated with the existing eDNA Biodiversity Assessment system:

```
src/
├── analysis/                    # NEW: Universal analysis module
│   ├── __init__.py
│   └── dataset_analyzer.py      # Main analyzer class
├── utils/
│   └── config.py               # Existing configuration system
└── ...

scripts/
├── analyze_dataset.py          # NEW: Command-line interface
└── ...
```

### Key Components

1. **DatasetAnalyzer Class** (`src/analysis/dataset_analyzer.py`)
   - Core analysis engine
   - Format detection and loading
   - Statistical calculations
   - Report generation

2. **CLI Interface** (`scripts/analyze_dataset.py`)
   - User-friendly command-line interface
   - Argument parsing and validation
   - Progress reporting

3. **Configuration Integration**
   - Uses existing project configuration system
   - Fallback for standalone operation

## 📋 Analysis Report Structure

Each generated report includes:

### 📊 Basic Sequence Statistics
- Total sequence count
- Length distribution (min, max, mean, median, std)
- Percentile analysis (25th, 75th, 90th, 95th)

### 🧬 Sequence Composition
- Auto-detected sequence type (DNA/RNA/protein)
- Character frequency analysis
- Top 15 most common characters/nucleotides/amino acids

### 📝 Annotation Analysis
- Organism distribution (top 10)
- Description pattern analysis
- Feature type counting
- Metadata completeness statistics

### 🔍 Quality Analysis
- Quality score statistics (for FASTQ files)
- Low-quality sequence identification
- Quality distribution metrics

### 🌿 Biodiversity Metrics
- Species richness
- Shannon diversity index
- Simpson diversity index
- Evenness measure
- Total abundance

### ⏱️ Processing Information
- Total processing time
- File format detected
- File size information
- Step-by-step timing breakdown

## 🔧 Technical Implementation

### Memory Optimization
- **Streaming file processing** to handle large datasets
- **Chunked parallel processing** for composition analysis
- **Vectorized calculations** using NumPy for statistics

### Performance Features
- **Auto-detection** of CPU cores for optimal parallelization
- **Progress indicators** for long-running operations
- **Efficient data structures** (Counter, defaultdict)
- **Minimal memory footprint** for large files

### Error Handling
- **Robust format detection** with fallback mechanisms
- **Graceful error handling** with informative messages
- **Input validation** and file existence checking

## 🎯 Benefits Over Previous Approach

### 1. **Consistency**
- Same analysis methodology across all dataset types
- Standardized output format for easy comparison
- Consistent performance optimizations

### 2. **Maintainability**
- Single codebase to maintain instead of multiple scripts
- Centralized configuration and optimization
- Easier testing and validation

### 3. **Usability**
- Simple command-line interface
- Auto-format detection reduces user complexity
- Comprehensive help and examples

### 4. **Scalability**
- Optimized for large datasets
- Parallel processing capabilities
- Memory-efficient streaming

### 5. **Integration**
- Fits naturally into existing eDNA project structure
- Uses existing configuration system
- Compatible with project conventions

## 🚀 Future Enhancements

### Planned Features
1. **Database Integration** - Direct connection to sequence databases
2. **Advanced Visualization** - Plot generation for reports
3. **Batch Processing** - Analyze multiple files in one command
4. **Quality Filtering** - Integration with preprocessing pipeline
5. **Taxonomy Assignment** - Integration with reference databases
6. **Clustering Analysis** - Integration with existing clustering modules

### Extension Points
- **Custom Analysis Modules** - Plugin architecture for specialized analyses
- **Output Formats** - JSON, CSV, HTML report generation
- **Cloud Integration** - Support for cloud storage and processing

## ✅ Conclusion

The Universal Dataset Analyzer represents a significant improvement over the previous approach of creating separate analysis scripts for each dataset type. It provides:

- **Unified interface** for all biological sequence formats
- **Optimized performance** with parallel processing
- **Comprehensive analysis** with standardized reporting
- **Easy integration** with the existing eDNA project
- **Scalable architecture** for future enhancements

This system makes it much easier to analyze biological datasets consistently while maintaining high performance and providing detailed, standardized reports for research and analysis purposes.