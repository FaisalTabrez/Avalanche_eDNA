# SRA Data Download Guide

This guide explains how to download datasets from NCBI Sequence Read Archive (SRA) for eDNA analysis.

## Overview

The Avalanche eDNA system supports downloading data from SRA in **two ways**:

1. **Direct ENA Download** (No installation required, works immediately)
2. **SRA Toolkit** (Faster, requires installation)

## Method 1: Direct ENA Download (Recommended for Beginners)

This method uses the European Nucleotide Archive (ENA) mirror and **requires no software installation**.

### How It Works

- Downloads FASTQ files directly from ENA's HTTP servers
- Automatically converts SRX (experiment) to SRR (run) accessions
- Works with any SRA accession number
- No external tools required

### Usage

1. Navigate to "SRA Browser" in the Streamlit UI
2. Search for datasets using keywords (e.g., "marine eDNA", "18S rRNA")
3. Select a dataset from the results
4. Click "Download" - the system will automatically use ENA if SRA Toolkit is not installed
5. Wait for download to complete

### Supported Accession Formats

- **SRR**: Single run (e.g., `SRR35551197`) - Direct download
- **SRX**: Experiment (e.g., `SRX31101225`) - Automatically converted to SRR first
- **SRP**: Project - Will list all runs within the project

### Example

```python
# Using the download script
from src.utils.sra_integration import SRAIntegrationUI

sra_ui = SRAIntegrationUI()

# Download SRX31101225 (will auto-convert to SRR)
success, file_path = sra_ui.download_sra_dataset(
    "SRX31101225",
    Path("data/sra/SRX31101225")
)

if success:
    print(f"Downloaded to: {file_path}")
```

## Method 2: SRA Toolkit (Faster, Advanced)

For frequent downloads or large datasets, installing SRA Toolkit provides faster performance.

### Installation (Windows)

#### Option A: Automated Installer (Easiest)

```bash
python install_sra_toolkit.py
```

This script will:
1. Download SRA Toolkit v3.0.10 for Windows
2. Extract to `%USERPROFILE%/sra-toolkit`
3. Configure the toolkit
4. Show you how to add it to your PATH

#### Option B: Manual Installation

1. Download SRA Toolkit from: https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit
2. Extract to a permanent location (e.g., `C:\Tools\sratoolkit`)
3. Add the `bin` folder to your PATH:
   - Open System Environment Variables
   - Edit Path under User Variables
   - Add: `C:\Tools\sratoolkit.3.0.10-win64\bin`
4. Restart your terminal

#### Verify Installation

```bash
prefetch --version
fastq-dump --version
```

### Configuration

Update `config/config.yaml` with the paths:

```yaml
databases:
  sra:
    sra_tools:
      prefetch_path: "C:/Users/YourName/sra-toolkit/sratoolkit.3.0.10-win64/bin/prefetch.exe"
      fastq_dump_path: "C:/Users/YourName/sra-toolkit/sratoolkit.3.0.10-win64/bin/fastq-dump.exe"
      fasterq_dump_path: "C:/Users/YourName/sra-toolkit/sratoolkit.3.0.10-win64/bin/fasterq-dump.exe"
```

## Troubleshooting

### Issue: "Failed to download SRX31101225"

**Solution**: SRX accessions need to be converted to SRR. The system now does this automatically. If it still fails:

1. Manually find the SRR accession:
   - Go to https://www.ncbi.nlm.nih.gov/sra
   - Search for `SRX31101225`
   - Click on the result
   - Look for "Run" section to find SRR accession

2. Use the SRR accession directly

### Issue: "SRA Toolkit not found"

**Solution**: The system will automatically fall back to ENA download. No action needed, but you can install SRA Toolkit for better performance:

```bash
python install_sra_toolkit.py
```

### Issue: "Download timeout"

**Solutions**:

1. **Check Internet Connection**: Ensure stable connection
2. **Use Smaller Datasets**: Try datasets with fewer reads first
3. **Retry**: Network issues are often temporary
4. **Use ENA Method**: More reliable for large files

### Issue: "No FASTQ files available"

**Cause**: Some SRA entries don't have pre-generated FASTQ files on ENA.

**Solution**: 
1. Install SRA Toolkit (it can convert .sra files to FASTQ)
2. Or contact the data submitters for raw files

### Issue: "Conversion from SRX to SRR failed"

**Cause**: The experiment might not have any runs, or NCBI API is down.

**Solution**:
1. Check NCBI website to verify the accession exists
2. Wait a few minutes and retry
3. Use the SRR accession directly if you know it

## Download Speed Comparison

| Method | Speed | Reliability | Requirements |
|--------|-------|-------------|--------------|
| ENA Direct | Moderate | High | None |
| SRA Toolkit | Fast | Moderate | Installation |

## Best Practices

1. **Start with ENA**: Try the automatic ENA download first
2. **Install SRA Toolkit for bulk downloads**: If downloading many datasets, install the toolkit
3. **Use SRR accessions when possible**: Direct run accessions skip the conversion step
4. **Monitor disk space**: FASTQ files can be large (100MB - 10GB+)
5. **Test with small datasets first**: Try `SRR1553606` (small test dataset)

## Command Line Usage

### Download using Python script

```bash
# Download a single dataset
python scripts/download_sra_data.py --accession SRR35551197 --output data/sra

# Download multiple datasets
python scripts/download_sra_data.py --accessions SRR1234567 SRR1234568 --output data/sra

# Search and download eDNA datasets
python scripts/download_sra_data.py --search "marine eDNA" --max-results 10 --download
```

### Using SRA Toolkit directly

```bash
# Method 1: Prefetch then convert
prefetch SRR35551197
fastq-dump --gzip --split-3 SRR35551197

# Method 2: Direct download and convert (faster)
fasterq-dump SRR35551197
gzip *.fastq
```

## Integration with Pipeline

Downloaded SRA data automatically integrates with the analysis pipeline:

```python
from scripts.run_pipeline import Pipeline

pipeline = Pipeline()

# Run analysis on downloaded SRA data
results = pipeline.run(
    input_path="data/sra/SRR35551197/SRR35551197.fastq.gz",
    dataset_name="Marine_Sample_SRR35551197"
)
```

## Common SRA Accession Prefixes

- **SRR**: Single sequencing run (download this)
- **SRX**: Experiment (contains one or more SRR)
- **SRS**: Sample (biological sample)
- **SRP**: Study/Project (contains multiple experiments)
- **SRA**: Submission

## Data Storage

Downloaded files are stored in:
```
data/
  sra/
    SRR35551197/
      SRR35551197.fastq.gz
    SRX31101225/
      SRR123456.fastq.gz  # Auto-converted
```

## API Rate Limits

- NCBI E-utilities: 3 requests/second without API key
- ENA API: No strict limits, but be reasonable
- Recommendation: Add delays between batch downloads

## Additional Resources

- NCBI SRA: https://www.ncbi.nlm.nih.gov/sra
- SRA Toolkit Wiki: https://github.com/ncbi/sra-tools/wiki
- ENA Browser: https://www.ebi.ac.uk/ena/browser/
- SRA FAQ: https://www.ncbi.nlm.nih.gov/sra/docs/sra-faq/

## Example Workflows

### Workflow 1: Explore and Download

1. Open Streamlit UI
2. Go to "SRA Browser"
3. Search: "marine metabarcoding 18S"
4. Browse results
5. Select interesting dataset
6. Click Download
7. Analyze in "Analysis" page

### Workflow 2: Batch Download Multiple Datasets

```python
from src.utils.sra_integration import SRAIntegrationUI
from pathlib import Path

sra_ui = SRAIntegrationUI()

accessions = ["SRR35551197", "SRX31101225", "SRR1553606"]

for acc in accessions:
    print(f"Downloading {acc}...")
    success, path = sra_ui.download_sra_dataset(
        acc,
        Path(f"data/sra/{acc}")
    )
    if success:
        print(f"✓ Downloaded: {path}")
    else:
        print(f"✗ Failed: {acc}")
```

### Workflow 3: Search, Download, and Analyze

```python
# Search for datasets
results = sra_ui.search_sra_datasets(
    keywords=["eDNA", "marine", "COI"],
    max_results=20
)

# Download top result
if results:
    first_result = results[0]
    acc = first_result['accession']
    
    success, path = sra_ui.download_sra_dataset(
        acc,
        Path(f"data/sra/{acc}")
    )
    
    if success:
        # Run pipeline
        from scripts.run_pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.run(path, dataset_name=acc)
```

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Check `logs/sra_download.log` for detailed error messages
3. Verify your internet connection
4. Try the alternative download method (ENA vs SRA Toolkit)
5. Report issues on GitHub with the error log
