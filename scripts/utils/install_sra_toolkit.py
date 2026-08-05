#!/usr/bin/env python3
"""
SRA Toolkit Installation Script for Windows

This script downloads and installs NCBI SRA Toolkit on Windows systems.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# SRA Toolkit version and download URL for Windows
SRA_VERSION = "3.0.10"
SRA_DOWNLOAD_URL = f"https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/{SRA_VERSION}/sratoolkit.{SRA_VERSION}-win64.zip"


def download_sra_toolkit(install_dir: Path):
    """Download SRA Toolkit"""
    print(f"Downloading SRA Toolkit {SRA_VERSION}...")

    zip_path = install_dir / f"sratoolkit.{SRA_VERSION}-win64.zip"

    try:
        urllib.request.urlretrieve(SRA_DOWNLOAD_URL, zip_path)
        print(f"Downloaded to: {zip_path}")
        return zip_path
    except Exception as e:
        print(f"Error downloading SRA Toolkit: {e}")
        return None


def extract_sra_toolkit(zip_path: Path, install_dir: Path):
    """Extract SRA Toolkit"""
    print("Extracting SRA Toolkit...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(install_dir)

        # Find the extracted directory
        extracted_dir = install_dir / f"sratoolkit.{SRA_VERSION}-win64"
        if extracted_dir.exists():
            print(f"Extracted to: {extracted_dir}")
            return extracted_dir
        else:
            print("Could not find extracted directory")
            return None

    except Exception as e:
        print(f"Error extracting SRA Toolkit: {e}")
        return None


def configure_sra_toolkit(sra_dir: Path):
    """Configure SRA Toolkit"""
    print("Configuring SRA Toolkit...")

    bin_dir = sra_dir / "bin"

    if not bin_dir.exists():
        print(f"Bin directory not found: {bin_dir}")
        return False

    # Test if prefetch works
    try:
        prefetch_exe = bin_dir / "prefetch.exe"
        result = subprocess.run(
            [str(prefetch_exe), "--version"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print("SRA Toolkit configured successfully!")
            print(f"Prefetch version: {result.stdout.strip()}")
            return True
        else:
            print(f"Prefetch test failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error testing SRA Toolkit: {e}")
        return False


def update_config_yaml(sra_dir: Path):
    """Update config.yaml with SRA Toolkit paths"""
    config_file = Path("config/config.yaml")

    if not config_file.exists():
        print("config.yaml not found, skipping configuration update")
        return

    bin_dir = sra_dir / "bin"
    prefetch_path = str(bin_dir / "prefetch.exe").replace("\\", "/")
    fastq_dump_path = str(bin_dir / "fastq-dump.exe").replace("\\", "/")
    fasterq_dump_path = str(bin_dir / "fasterq-dump.exe").replace("\\", "/")

    print(f"\nAdd these paths to your config/config.yaml:")
    print("\ndatabases:")
    print("  sra:")
    print("    sra_tools:")
    print(f'      prefetch_path: "{prefetch_path}"')
    print(f'      fastq_dump_path: "{fastq_dump_path}"')
    print(f'      fasterq_dump_path: "{fasterq_dump_path}"')
    print(f"      bin_dir: \"{str(bin_dir).replace('\\', '/')}\"")


def add_to_path(bin_dir: Path):
    """Instructions to add to PATH"""
    print("\n" + "=" * 60)
    print("IMPORTANT: Add SRA Toolkit to your PATH")
    print("=" * 60)
    print(f"\nAdd this directory to your system PATH:")
    print(f"  {bin_dir}")
    print("\nSteps:")
    print("1. Open 'Environment Variables' (search in Windows Start)")
    print("2. Under 'User variables', select 'Path' and click 'Edit'")
    print("3. Click 'New' and add the path above")
    print("4. Click 'OK' to save")
    print("5. Restart your terminal/command prompt")
    print("\nOr run this PowerShell command as Administrator:")
    print(
        f'[Environment]::SetEnvironmentVariable("Path", $env:Path + ";{bin_dir}", "User")'
    )


def main():
    print("=" * 60)
    print("SRA Toolkit Installation Script for Windows")
    print("=" * 60)
    print()

    # Set installation directory
    install_dir = Path.home() / "sra-toolkit"
    install_dir.mkdir(parents=True, exist_ok=True)

    print(f"Installation directory: {install_dir}")
    print()

    # Check if already installed
    existing_sra = install_dir / f"sratoolkit.{SRA_VERSION}-win64"
    if existing_sra.exists():
        print(f"SRA Toolkit {SRA_VERSION} already exists at {existing_sra}")
        response = input("Reinstall? (y/n): ")
        if response.lower() != "y":
            print("Installation cancelled")
            update_config_yaml(existing_sra)
            add_to_path(existing_sra / "bin")
            return

    # Download
    zip_path = download_sra_toolkit(install_dir)
    if not zip_path:
        print("Download failed. Installation aborted.")
        return

    # Extract
    sra_dir = extract_sra_toolkit(zip_path, install_dir)
    if not sra_dir:
        print("Extraction failed. Installation aborted.")
        return

    # Configure
    if configure_sra_toolkit(sra_dir):
        print(f"\nSRA Toolkit successfully installed to: {sra_dir}")

        # Update config
        update_config_yaml(sra_dir)

        # PATH instructions
        add_to_path(sra_dir / "bin")

        print("\nInstallation complete!")
    else:
        print("\nInstallation completed but configuration failed.")
        print("Please check the installation manually.")

    # Cleanup
    try:
        zip_path.unlink()
        print(f"\nCleaned up temporary file: {zip_path}")
    except:
        pass


if __name__ == "__main__":
    main()
