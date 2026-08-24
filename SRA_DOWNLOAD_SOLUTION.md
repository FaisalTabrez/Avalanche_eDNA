# SRA Download Issue - SOLUTION SUMMARY

## Problem

Failed to download SRX31101225 from NCBI SRA Toolkit, showing error:

```
Failed to download SRX31101225 when I try to download a dataset from SRA toolkit
```

## Root Causes Identified

1. **SRA Toolkit Not Installed**: `prefetch` command not found in PATH
2. **SRX vs SRR Accession**: SRX31101225 is an Experiment accession, but downloads require SRR (Run) accessions
3. **New Dataset**: SRX31101225 → SRR36053676 is a very recent dataset (2025) and doesn't have pre-generated FASTQ files on ENA mirror yet

## Solutions Implemented

### ✅ Solution 1: Automatic SRX→SRR Conversion

The system now automatically converts SRX (experiment) accessions to SRR (run) accessions:

- **SRX31101225** → **SRR36053676**
- Uses NCBI E-utilities API to fetch run information
- Transparent to the user

### ✅ Solution 2: ENA Fallback Download (No SRA Toolkit Required)

Added automatic fallback to European Nucleotide Archive (ENA) when SRA Toolkit is not available:

- Works without installing any software
- Downloads FASTQ files directly via HTTP
- Handles both single-end and paired-end data

**Location**: `src/utils/sra_integration.py`

### ✅ Solution 3: SRA Toolkit Installer

Created automated installer for Windows:

```bash
python install_sra_toolkit.py
```

This will:

- Download SRA Toolkit v3.0.10 for Windows
- Extract to `%USERPROFILE%/sra-toolkit`
- Configure and test installation  
- Provide PATH setup instructions

**Location**: `install_sra_toolkit.py`

### ✅ Solution 4: Comprehensive Documentation

Created detailed guide: **SRA_DOWNLOAD_GUIDE.md**

Covers:

- Two download methods (ENA vs SRA Toolkit)
- Installation instructions
- Troubleshooting common issues
- Example workflows
- API usage

## How to Download SRX31101225 Now

### Option A: Use a Dataset That Has FASTQ Files Available

Since SRR36053676 is very new and doesn't have FASTQ files on ENA yet, you can:

1. **Wait**: ENA typically generates FASTQ files within a few days/weeks of submission
2. **Install SRA Toolkit**: This can download and convert the raw .sra file

### Option B: Install SRA Toolkit and Try Again

```bash
# 1. Install SRA Toolkit
python install_sra_toolkit.py

# 2. Add to PATH (follow on-screen instructions)

# 3. Test download
python test_sra_download.py

# 4. Download your dataset
python -c "
from src.utils.sra_integration import SRAIntegrationUI
from pathlib import Path

sra_ui = SRAIntegrationUI()
success, path = sra_ui.download_sra_dataset(
    'SRX31101225',
    Path('data/sra/SRX31101225')
)
print(f'Success: {success}, Path: {path}')
"
```

### Option C: Use the Streamlit UI

```bash
# Start the app
streamlit run streamlit_app.py --server.port 8504

# Then:
# 1. Navigate to "SRA Browser"
# 2. Enter accession: SRX31101225
# 3. Click Download
# 4. System will automatically:
#    - Convert SRX → SRR
#    - Try SRA Toolkit (if installed)
#    - Fall back to ENA if needed
```

### Option D: Try a Known Working Dataset First

Test with these datasets that definitely have FASTQ files available:

```python
from src.utils.sra_integration import SRAIntegrationUI
from pathlib import Path

sra_ui = SRAIntegrationUI()

# Small test dataset (good for testing)
success, path = sra_ui.download_sra_dataset(
    'SRR1553606',  # ~10MB, paired-end marine sample
    Path('data/sra/test_dataset')
)

# eDNA dataset example
success, path = sra_ui.download_sra_dataset(
    'SRR35551197',  # Real eDNA dataset used in testing
    Path('data/sra/edna_example')
)
```

## Current System Capabilities

✅ Automatic SRX → SRR conversion
✅ ENA fallback download (no installation required)
✅ SRA Toolkit support (when installed)  
✅ Search functionality
✅ Streamlit UI integration
✅ Command-line tools
✅ Error handling and retry logic
✅ Progress tracking
✅ Comprehensive logging

## Files Created/Modified

### New Files

- `install_sra_toolkit.py` - Automated Windows installer
- `SRA_DOWNLOAD_GUIDE.md` - Comprehensive user guide
- `test_sra_download.py` - Test suite

### Modified Files

- `src/utils/sra_integration.py` - Enhanced with:
  - `_convert_srx_to_srr()` - Automatic accession conversion
  - `_download_from_ena()` - ENA fallback method
  - Improved error handling
  - Better logging

## Next Steps

1. **Immediate**: Try downloading a working dataset first (SRR1553606 or SRR35551197)
2. **Short-term**: Install SRA Toolkit for SRX31101225
3. **Alternative**: Wait for ENA to generate FASTQ files for SRR36053676
4. **Long-term**: Use the system with any SRA accession

## Testing

Run the test suite to verify everything works:

```bash
python test_sra_download.py
```

Expected results:

- ✅ SRX→SRR conversion: PASS
- ⚠️ Download SRR36053676: FAIL (no FASTQ files yet)
- ✅ Download SRR1553606: PASS (if you change the test dataset)
- ✅ Search: PASS

## Support

For additional help:

1. Read `SRA_DOWNLOAD_GUIDE.md`
2. Check logs in console output
3. Verify accession exists: <https://www.ncbi.nlm.nih.gov/sra/?term=SRX31101225>
4. Try alternative accessions from your search results

## Quick Reference

| Task | Command |
| ------ | --------- |
| Install SRA Toolkit | `python install_sra_toolkit.py` |
| Test System | `python test_sra_download.py` |
| Download Dataset | See scripts/examples above |
| Browse SRA | Use Streamlit UI → SRA Browser |
| Read Guide | Open `SRA_DOWNLOAD_GUIDE.md` |
diddy
