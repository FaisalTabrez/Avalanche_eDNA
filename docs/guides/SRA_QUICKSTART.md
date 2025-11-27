# Quick Start: Downloading SRA Data

## Your Specific Issue: SRX31101225

You tried to download **SRX31101225** and it failed. Here's what happened and how to fix it:

### Problem Analysis

1. **SRA Toolkit Not Installed**: The `prefetch` command wasn't found
2. **Accession Type**: SRX31101225 is an *experiment* - it needs to be converted to an SRR *run* accession
3. **New Dataset**: This converts to SRR36053676, which is very recent (2025) and doesn't have pre-generated FASTQ files on ENA yet

### ✅ SOLUTIONS (3 Options)

---

## OPTION 1: Quick Download (Try This First) 🚀

```bash
python download_sra_quick.py SRX31101225
```

This will:
- ✅ Automatically convert SRX → SRR
- ✅ Try multiple download methods
- ✅ Work without SRA Toolkit (uses ENA mirror)
- ✅ Show progress and helpful error messages

**Note**: This specific dataset (SRR36053676) might fail because it's too new. If it does, try Option 3 below.

---

## OPTION 2: Install SRA Toolkit (Recommended for Frequent Use) 📦

If you plan to download many datasets, install SRA Toolkit:

```bash
# 1. Run the installer
python install_sra_toolkit.py

# 2. Follow the on-screen PATH setup instructions

# 3. Restart your terminal

# 4. Try downloading again
python download_sra_quick.py SRX31101225
```

**Benefits**:
- Faster downloads
- Can handle datasets not on ENA yet
- Converts .sra files to FASTQ automatically

---

## OPTION 3: Test with a Known Working Dataset First ✅

Since SRX31101225 is very new, try this proven dataset instead:

```bash
# Small test dataset (10MB, fast download)
python download_sra_quick.py SRR1553606

# Or a real eDNA dataset we've tested
python download_sra_quick.py SRR35551197
```

Then go back and try SRX31101225 in a few days once ENA generates the FASTQ files.

---

## Using the Streamlit UI

```bash
# 1. Start the app
streamlit run streamlit_app.py --server.port 8504

# 2. Navigate to "SRA Browser" page

# 3. Either:
#    - Search for datasets using keywords
#    - Or enter accession directly: SRX31101225

# 4. Click "Download"
#    - System handles SRX→SRR conversion automatically
#    - Shows progress
#    - Downloads to data/sra/
```

---

## Command Reference

| What You Want | Command |
|--------------|---------|
| Download SRX31101225 | `python download_sra_quick.py SRX31101225` |
| Download test dataset | `python download_sra_quick.py SRR1553606` |
| Install SRA Toolkit | `python install_sra_toolkit.py` |
| Test system | `python test_sra_download.py` |
| Use UI | `streamlit run streamlit_app.py --server.port 8504` |

---

## Troubleshooting

### ❌ "Failed to download SRX31101225"

**Reason**: SRR36053676 (what it converts to) is too new - no FASTQ files on ENA yet.

**Solutions**:
1. Install SRA Toolkit (Option 2 above)
2. Try a different dataset (Option 3 above)
3. Wait a few days and retry

### ❌ "SRA Toolkit not found"

**Reason**: Not installed or not in PATH.

**Solution**: Run `python install_sra_toolkit.py`

### ❌ "No FASTQ files available"

**Reason**: Dataset too new or restricted access.

**Solutions**:
1. Install SRA Toolkit (can download raw .sra files)
2. Try alternative accession
3. Check if dataset is public: https://www.ncbi.nlm.nih.gov/sra/?term=SRX31101225

---

## What We Fixed

✅ **Automatic SRX→SRR conversion** - No need to manually find run accessions
✅ **ENA fallback download** - Works without installing anything
✅ **SRA Toolkit installer** - Automated installation for Windows
✅ **Better error messages** - Tells you exactly what went wrong
✅ **Multiple download methods** - Tries toolkit, falls back to ENA
✅ **Progress tracking** - See download status in real-time

---

## Next Steps

1. **Try downloading** a known working dataset:
   ```bash
   python download_sra_quick.py SRR1553606
   ```

2. **If successful**, analyze it in the UI or run the pipeline

3. **Install SRA Toolkit** if you'll download more data:
   ```bash
   python install_sra_toolkit.py
   ```

4. **Try SRX31101225 again** after SRA Toolkit is installed:
   ```bash
   python download_sra_quick.py SRX31101225
   ```

---

## Full Documentation

- **Complete Guide**: `SRA_DOWNLOAD_GUIDE.md` - Everything about SRA downloads
- **Solution Details**: `SRA_DOWNLOAD_SOLUTION.md` - Technical details of the fix
- **Code Reference**: `src/utils/sra_integration.py` - Implementation

---

## Quick Example: End-to-End Workflow

```bash
# 1. Download a test dataset
python download_sra_quick.py SRR1553606

# 2. Start the UI
streamlit run streamlit_app.py --server.port 8504

# 3. Go to "Analysis" page

# 4. Select the downloaded file from:
#    data/sra/SRR1553606/SRR1553606_1.fastq.gz

# 5. Run analysis!
```

---

## Still Having Issues?

1. Check your internet connection
2. Verify the accession: https://www.ncbi.nlm.nih.gov/sra/?term=SRX31101225
3. Review logs in the console output
4. Try with `SRR1553606` first to test the system
5. Read `SRA_DOWNLOAD_GUIDE.md` for detailed troubleshooting

---

**TL;DR**: Run `python download_sra_quick.py SRR1553606` to test, then try `SRX31101225` after installing SRA Toolkit with `python install_sra_toolkit.py`.
