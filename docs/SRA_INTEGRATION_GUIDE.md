# SRA Integration Guide

## Overview

The eDNA Biodiversity Assessment System now includes full integration with NCBI's Sequence Read Archive (SRA), allowing you to search, download, and analyze publicly available eDNA datasets directly from the web interface.

## Features

### 1. SRA Toolkit Integration
- Automated download and conversion of SRA datasets to FASTQ format
- Support for prefetch, fastq-dump, and sam-dump tools
- Configured paths in `config.yaml` for seamless operation

### 2. Dataset Search & Browse
- Search NCBI SRA database with custom keywords
- Filter by organism, platform, and data size
- View detailed metadata for each dataset
- Quick access to accession numbers, titles, and statistics

### 3. Direct Analysis Integration
- Download SRA datasets directly from the Analysis page
- Use SRA data for model training
- Automatic format detection and processing

### 4. Batch Download
- Queue multiple datasets for download
- Batch processing with progress tracking
- Manage downloaded datasets from central interface

## Installation

### SRA Toolkit Setup

The SRA Toolkit is already installed and configured at:
```
tools/sratoolkit.3.0.10-win64/bin/
```

Configuration paths are set in `config/config.yaml`:
```yaml
databases:
  sra:
    sra_tools:
      prefetch_path: 'C:\Volume D\Avalanche\tools\sratoolkit.3.0.10-win64\bin\prefetch.exe'
      fastq_dump_path: 'C:\Volume D\Avalanche\tools\sratoolkit.3.0.10-win64\bin\fastq-dump.exe'
      sam_dump_path: 'C:\Volume D\Avalanche\tools\sratoolkit.3.0.10-win64\bin\sam-dump.exe'
```

### Verify Installation

Run the verification script:
```bash
python scripts/sra_integration_example.py
```

Or check from Python:
```python
from src.utils.sra_integration import SRAIntegrationUI

sra_ui = SRAIntegrationUI()
print(f"SRA Toolkit available: {sra_ui.sra_toolkit_available}")
```

## Using SRA Integration in the Web Interface

### Dataset Analysis with SRA

1. Navigate to **Dataset Analysis** page
2. Select **"Download from SRA"** as data source
3. Enter an SRA accession (e.g., `SRR12345678`)
4. Click **"Download SRA Dataset"**
5. Once downloaded, proceed with analysis as normal

### Model Training with SRA Data

1. Navigate to **Model Training** page
2. In the "Data Selection" section, choose **"Download from NCBI SRA"**
3. Search for datasets or enter a specific accession
4. Download and select the dataset
5. Configure training parameters and start training

### SRA Browser

The dedicated **SRA Browser** page provides comprehensive dataset management:

#### Search & Browse Tab
- Enter search keywords (e.g., "eDNA", "18S rRNA", "marine")
- Set maximum results (10-100)
- Click **"Search"** to query NCBI SRA
- Filter results by organism and sequencing platform
- Download individual datasets or add to batch queue

#### Batch Download Tab
- Review queued datasets
- Download all datasets in queue with single click
- Track progress for each download
- Remove items from queue as needed

#### Downloaded Datasets Tab
- View all previously downloaded SRA datasets
- Quick access to file paths
- Direct links to analyze or use for training

## Using SRA Integration Programmatically

### Basic Download

```python
from pathlib import Path
from src.utils.sra_integration import SRAIntegrationUI

# Initialize SRA interface
sra_ui = SRAIntegrationUI()

# Download a dataset
accession = "SRR12345678"
output_dir = Path("data/sra") / accession

success, file_path = sra_ui.download_sra_dataset(
    accession,
    output_dir,
    progress_callback=lambda msg: print(msg)
)

if success:
    print(f"Downloaded to: {file_path}")
```

### Search for Datasets

```python
from src.utils.sra_integration import SRAIntegrationUI

sra_ui = SRAIntegrationUI()

# Search for eDNA datasets
results = sra_ui.search_sra_datasets(
    keywords=["eDNA", "18S rRNA"],
    max_results=50
)

for study in results:
    print(f"{study['accession']}: {study['title']}")
    print(f"  Organism: {study.get('organism', 'N/A')}")
    print(f"  Spots: {study.get('spots', 'N/A')}")
```

### Using in Analysis Pipeline

```python
from pathlib import Path
from src.utils.sra_integration import SRAIntegrationUI
from src.analysis.dataset_analyzer import DatasetAnalyzer

# Download SRA dataset
sra_ui = SRAIntegrationUI()
accession = "SRR12345678"
output_dir = Path("data/sra") / accession

success, file_path = sra_ui.download_sra_dataset(accession, output_dir)

if success:
    # Analyze the downloaded dataset
    analyzer = DatasetAnalyzer()
    results = analyzer.analyze_dataset(
        input_path=str(file_path),
        output_path=f"results/{accession}_analysis.txt",
        dataset_name=accession
    )
    print("Analysis complete!")
```

## Configuration

### SRA Search Parameters

Edit `config/config.yaml` to customize search behavior:

```yaml
databases:
  sra:
    search:
      edna_keywords:
        - "eDNA"
        - "environmental DNA"
        - "metabarcoding"
        - "18S rRNA"
        - "16S rRNA"
        - "COI"
        - "marine"
        - "ocean"
      min_spots: 1000000  # Minimum number of spots to consider
```

### Download Settings

Configure download timeouts and behavior:

```yaml
databases:
  sra:
    download:
      timeout: 1200  # seconds
      retry_attempts: 3
      compression: gzip  # Output format
```

## Troubleshooting

### SRA Toolkit Not Detected

**Problem:** "SRA Toolkit not detected" message appears

**Solutions:**
1. Run installation script: `python install_sra_toolkit.py`
2. Verify paths in `config.yaml` are correct
3. Check that toolkit binaries exist in `tools/sratoolkit.3.0.10-win64/bin/`

### Download Failures

**Problem:** Dataset download fails

**Solutions:**
1. Check internet connection
2. Verify accession number is correct
3. Try downloading from NCBI website manually to confirm dataset exists
4. Check logs for specific error messages

### Slow Downloads

**Problem:** Downloads are very slow

**Solutions:**
1. Check network bandwidth
2. Try downloading during off-peak hours
3. For large datasets, consider using command-line tools directly
4. Use batch download for multiple datasets to optimize

### Format Conversion Errors

**Problem:** SRA to FASTQ conversion fails

**Solutions:**
1. Ensure sufficient disk space
2. Check SRA file integrity
3. Update SRA Toolkit to latest version
4. Try using alternative conversion parameters

## Best Practices

### Dataset Selection
- Start with smaller datasets (< 1GB) for testing
- Use search filters to find relevant eDNA studies
- Check metadata before downloading large datasets
- Verify organism and sequencing platform match your needs

### Storage Management
- Downloaded files are stored in `data/sra/{accession}/`
- Clean up old datasets periodically
- Use symbolic links for large datasets if needed
- Consider external storage for large collections

### Performance Optimization
- Use batch download for multiple datasets
- Enable fast mode for large dataset analysis
- Limit max_sequences for initial exploration
- Use parallel processing when available

### Workflow Integration
1. Search and browse datasets in SRA Browser
2. Add interesting datasets to batch queue
3. Download in batch overnight or during low-activity periods
4. Analyze downloaded datasets using Dataset Analysis page
5. Use analyzed data for model training or further research

## API Reference

### SRAIntegrationUI Class

Main class for SRA integration in Streamlit interface.

**Methods:**

- `search_sra_datasets(keywords, max_results)` - Search NCBI SRA
- `download_sra_dataset(accession, output_dir, progress_callback)` - Download dataset
- `show_sra_browser()` - Display SRA browser UI
- `show_sra_toolkit_status()` - Display toolkit status

### Helper Functions

- `create_sra_data_source_selector()` - Unified data source selector with SRA support

## Examples

### Example 1: Download and Analyze Marine eDNA

```python
from pathlib import Path
from src.utils.sra_integration import SRAIntegrationUI
from src.analysis.dataset_analyzer import DatasetAnalyzer

# Search for marine eDNA datasets
sra_ui = SRAIntegrationUI()
results = sra_ui.search_sra_datasets(["marine", "eDNA", "18S"], max_results=10)

# Download first result
if results:
    accession = results[0]['accession']
    output_dir = Path("data/sra") / accession
    success, file_path = sra_ui.download_sra_dataset(accession, output_dir)
    
    if success:
        # Analyze
        analyzer = DatasetAnalyzer(fast_mode=True)
        results = analyzer.analyze_dataset(
            input_path=str(file_path),
            output_path=f"results/{accession}_report.txt",
            dataset_name=f"Marine eDNA - {accession}"
        )
```

### Example 2: Batch Download for Training Dataset

```python
from pathlib import Path
from src.utils.sra_integration import SRAIntegrationUI

sra_ui = SRAIntegrationUI()

# List of accessions to download
accessions = ["SRR12345678", "SRR87654321", "SRR11111111"]

downloaded_files = []
for accession in accessions:
    output_dir = Path("data/training/sra") / accession
    success, file_path = sra_ui.download_sra_dataset(
        accession,
        output_dir,
        progress_callback=lambda msg: print(f"[{accession}] {msg}")
    )
    
    if success:
        downloaded_files.append(file_path)

print(f"Successfully downloaded {len(downloaded_files)} datasets")
```

## Advanced Topics

### Custom Search Queries

You can customize search queries by modifying the search logic in `src/utils/sra_integration.py`:

```python
# Example: Search for specific date range
def search_with_date_filter(keywords, start_date, end_date):
    query = ' OR '.join(f'"{term}"' for term in keywords)
    query += f' AND "{start_date}"[Publication Date] : "{end_date}"[Publication Date]'
    # ... rest of search logic
```

### Parallel Downloads

For large-scale dataset collection, implement parallel downloading:

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.utils.sra_integration import SRAIntegrationUI

def download_dataset(accession):
    sra_ui = SRAIntegrationUI()
    output_dir = Path("data/sra") / accession
    return sra_ui.download_sra_dataset(accession, output_dir)

accessions = ["SRR1", "SRR2", "SRR3", "SRR4"]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(download_dataset, accessions))
```

### Integration with External Databases

Combine SRA data with other sequence databases:

```python
# Download from SRA
sra_file = download_sra_dataset("SRR12345678")

# Merge with local sequences
from Bio import SeqIO

local_seqs = list(SeqIO.parse("local_data.fasta", "fasta"))
sra_seqs = list(SeqIO.parse(sra_file, "fastq"))

combined = local_seqs + sra_seqs
SeqIO.write(combined, "combined_dataset.fasta", "fasta")
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review SRA Toolkit documentation: https://github.com/ncbi/sra-tools
3. Check NCBI SRA help: https://www.ncbi.nlm.nih.gov/sra/docs/
4. Review project logs in `logs/` directory

## References

- NCBI SRA: https://www.ncbi.nlm.nih.gov/sra
- SRA Toolkit: https://github.com/ncbi/sra-tools
- E-utilities API: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- eDNA Metadata Standards: https://www.ncbi.nlm.nih.gov/biosample/docs/packages/
