#!/usr/bin/env python3
"""
Quick SRA Download Script

Download SRX31101225 (or any SRA accession) with automatic fallback methods.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.sra_integration import SRAIntegrationUI
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_sra(accession: str, output_base: str = "data/sra"):
    """
    Download an SRA dataset with automatic method selection
    
    Args:
        accession: SRA accession (SRR, SRX, etc.)
        output_base: Base directory for downloads
    """
    print("="*60)
    print(f"Downloading SRA Dataset: {accession}")
    print("="*60)
    
    sra_ui = SRAIntegrationUI()
    
    # Check SRA Toolkit status
    if sra_ui.sra_toolkit_available:
        print("\n[INFO] SRA Toolkit is available - will try that first")
    else:
        print("\n[INFO] SRA Toolkit not found - using ENA mirror")
        print("[TIP] For faster downloads, run: python install_sra_toolkit.py")
    
    # Create output directory
    output_dir = Path(output_base) / accession
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Output directory: {output_dir}")
    print("[INFO] Starting download...\n")
    
    # Download
    success, file_path = sra_ui.download_sra_dataset(
        accession,
        output_dir,
        progress_callback=lambda msg: print(f"  > {msg}")
    )
    
    print("\n" + "="*60)
    if success and file_path:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Download completed!")
        print(f"  File: {file_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print("="*60)
        
        print("\n[NEXT STEPS]")
        print(f"1. Analyze this dataset in the Streamlit UI")
        print(f"2. Or run pipeline: python scripts/run_pipeline.py --input {file_path}")
        print(f"3. File location: {file_path.absolute()}")
        
        return file_path
    else:
        print(f"[FAILED] Could not download {accession}")
        print("="*60)
        
        print("\n[TROUBLESHOOTING]")
        print(f"1. Verify accession exists: https://www.ncbi.nlm.nih.gov/sra/?term={accession}")
        print(f"2. Check if it's a recent dataset (may not have FASTQ files on ENA yet)")
        print(f"3. Try installing SRA Toolkit: python install_sra_toolkit.py")
        print(f"4. Try a different accession (e.g., SRR1553606)")
        
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download datasets from NCBI SRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific accession
  python download_sra_quick.py SRX31101225
  
  # Download to custom location
  python download_sra_quick.py SRR1553606 --output my_data/sra
  
  # Download known working test dataset
  python download_sra_quick.py SRR1553606
  
Supported accession types:
  SRR - Run accession (direct download)
  SRX - Experiment accession (auto-converted to SRR)
  SRP - Project accession (lists all runs)
        """
    )
    
    parser.add_argument(
        "accession",
        nargs="?",
        default="SRX31101225",
        help="SRA accession to download (default: SRX31101225)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        default="data/sra",
        help="Output directory (default: data/sra)"
    )
    
    args = parser.parse_args()
    
    print(f"\nDownloading: {args.accession}")
    print(f"Output: {args.output}\n")
    
    result = download_sra(args.accession, args.output)
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
