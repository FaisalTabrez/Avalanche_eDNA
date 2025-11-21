# BLAST Integration Guide

## Overview

BLAST (Basic Local Alignment Search Tool) has been successfully integrated into the eDNA Biodiversity Assessment System for Windows. This integration provides taxonomic assignment capabilities using BLAST searches against reference databases.

## 🚀 Features Integrated

### 1. Windows BLAST Runner (`src/utils/blast_utils.py`)
- **WindowsBLASTRunner**: Optimized BLAST execution for Windows
- **Automatic BLAST verification**: Checks BLAST installation on startup
- **Database creation**: Creates BLAST databases from FASTA files
- **Sequence search**: Runs BLASTN searches with proper Windows path handling
- **Result parsing**: Parses BLAST XML output into structured results

### 2. Enhanced Taxonomy Assignment (`src/clustering/taxonomy.py`)
- **BlastTaxonomyAssigner**: Updated to use Windows BLAST utilities
- **Integrated workflow**: Seamless integration with existing pipeline
- **Configurable parameters**: E-value, identity thresholds, max targets
- **Result conversion**: Converts BLAST results to taxonomy assignment format

### 3. Database Setup Script (`scripts/build_blast_db.py`)
- **Enhanced script**: Updated with Windows-specific improvements
- **Multiple implementations**: Windows utilities + fallback methods
- **Better error handling**: Improved debugging and error reporting
- **Usage examples**: Sample commands for common use cases

### 4. Configuration Updates (`config/config.yaml`)
- **Windows paths**: Configured for BLAST 2.17.0+ installation
- **Executable paths**: Direct paths to BLAST binaries
- **Parameters**: Optimized settings for eDNA analysis

## 📋 Prerequisites

- ✅ **BLAST+ 2.17.0**: Installed at `C:\Program Files\NCBI\blast-2.17.0+\bin`
- ✅ **Python environment**: All dependencies installed
- ✅ **Sample data**: Available in `data/sample/sample_edna_sequences.fasta`

## 🔧 Usage Examples

### 1. Create a BLAST Database

```powershell
# Create database from sample data
python scripts\build_blast_db.py --fasta "data\sample\sample_edna_sequences.fasta" --db-out "reference\indices\sample_db"

# Create database with taxonomy mapping
python scripts\build_blast_db.py --fasta "reference.fasta" --taxid-map "taxonomy.txt" --db-out "reference\indices\my_db"
```

### 2. Use BLAST in Python Code

```python
from utils.blast_utils import WindowsBLASTRunner

# Initialize BLAST runner
blast_runner = WindowsBLASTRunner()

# Create database
success = blast_runner.create_blast_database(
    fasta_file="data/sample/sample_edna_sequences.fasta",
    database_name="my_database",
    database_type='nucl'
)

# Search sequences
results = blast_runner.run_blastn_search(
    query_sequences=["ATCGATCG..."],
    database_path="my_database",
    sequence_ids=["seq1"]
)
```

### 3. Taxonomy Assignment with BLAST

```python
from clustering.taxonomy import BlastTaxonomyAssigner

# Initialize taxonomy assigner
assigner = BlastTaxonomyAssigner(
    blast_db="reference/indices/my_database",
    identity_threshold=97.0
)

# Assign taxonomy
results = assigner.assign_taxonomy(
    sequences=["ATCGATCG...", "GCTAGCTA..."],
    sequence_ids=["seq1", "seq2"]
)

# Results include: sequence_id, taxonomy, identity, evalue, etc.
```

### 4. Integration with Main Pipeline

The BLAST integration is automatically used when:
- BLAST databases are available in the configured paths
- Taxonomy assignment methods include BLAST options
- Fallback taxonomy assignment is enabled

## 🔍 Configuration Options

Key configuration settings in `config/config.yaml`:

```yaml
taxonomy:
  blast:
    # Windows BLAST executable paths
    blastn_path: "C:\\Program Files\\NCBI\\blast-2.17.0+\\bin\\blastn.exe"
    makeblastdb_path: "C:\\Program Files\\NCBI\\blast-2.17.0+\\bin\\makeblastdb.exe"
    
    # Search parameters
    evalue: 1e-5
    max_targets: 10
    identity_threshold: 97.0
    num_threads: 4
    
  blast_fallback:
    enable: true
    min_identity_species: 97.0
```

## ✅ Tested Functionality

All components have been tested and verified:

- ✅ **BLAST Installation**: Version check and executable access
- ✅ **Database Creation**: FASTA to BLAST database conversion
- ✅ **Sequence Search**: Query sequences against databases
- ✅ **Taxonomy Assignment**: Full integration with taxonomy pipeline
- ✅ **Result Parsing**: XML output parsing and structuring
- ✅ **Windows Compatibility**: Proper path handling and execution

## 🐛 Troubleshooting

### Common Issues:

1. **"BLAST tools not found"**
   - Verify BLAST+ is installed at the configured path
   - Check Windows PATH environment variable

2. **"Database creation failed"**
   - Ensure input FASTA file exists and is readable
   - Check output directory permissions
   - Avoid paths with spaces when possible

3. **"No hits found"**
   - Verify database was created successfully
   - Check E-value and identity thresholds
   - Ensure query sequences are in correct format

### Debug Mode:

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Integration with Existing Pipeline

The BLAST integration seamlessly works with:

- **Data preprocessing**: Uses cleaned sequences for database creation
- **Embedding pipeline**: BLAST provides alternative to ML-based taxonomy
- **Clustering analysis**: BLAST results inform taxonomic clustering
- **Visualization**: Results displayed in dashboards and reports
- **Novelty detection**: Unknown sequences identified through BLAST searches

## 📊 Performance Notes

- Database creation: ~1-2 seconds for 1000 sequences
- Search performance: ~0.1-1 second per sequence (depends on database size)
- Memory usage: Minimal additional overhead
- Parallel processing: Configurable thread count for searches

## 🎯 Next Steps

The BLAST integration is ready for production use. Consider:

1. **Reference database setup**: Create comprehensive eDNA reference databases
2. **Taxonomy mapping**: Add NCBI taxonomy integration for better species identification
3. **Performance tuning**: Optimize parameters for specific use cases
4. **Custom databases**: Create specialized databases for specific environments

---

*Last updated: October 2, 2025*
*BLAST version: 2.17.0+*
*Integration status: ✅ Complete and tested*