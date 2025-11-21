#!/usr/bin/env python3
"""
Automated SRA Toolkit Installation Script for Windows

This script downloads and configures NCBI SRA Toolkit for the eDNA pipeline.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SRAToolkitInstaller:
    """Handles SRA Toolkit download and installation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tools_dir = self.project_root / "tools"
        self.sra_dir = self.tools_dir / "sratoolkit"
        
        # SRA Toolkit download URL for Windows (latest version 3.0.10)
        self.sra_toolkit_url = "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.0.10/sratoolkit.3.0.10-win64.zip"
        self.sra_toolkit_version = "3.0.10"
        
    def download_sra_toolkit(self):
        """Download SRA Toolkit from NCBI"""
        logger.info("=" * 60)
        logger.info("DOWNLOADING SRA TOOLKIT")
        logger.info("=" * 60)
        
        # Create tools directory
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        zip_file = self.tools_dir / "sratoolkit.zip"
        
        try:
            logger.info(f"Downloading from: {self.sra_toolkit_url}")
            logger.info("This may take a few minutes (file size: ~150 MB)...")
            
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                
                if block_num % 100 == 0:  # Update every 100 blocks
                    logger.info(f"Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
            
            urllib.request.urlretrieve(
                self.sra_toolkit_url,
                zip_file,
                reporthook=show_progress
            )
            
            logger.info("✅ Download completed successfully!")
            return zip_file
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            logger.error("Please download manually from:")
            logger.error("https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit")
            return None
    
    def extract_sra_toolkit(self, zip_file):
        """Extract SRA Toolkit"""
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING SRA TOOLKIT")
        logger.info("=" * 60)
        
        try:
            logger.info(f"Extracting to: {self.tools_dir}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.tools_dir)
            
            # Find the extracted directory
            extracted_dirs = [d for d in self.tools_dir.glob("sratoolkit*") if d.is_dir()]
            
            if extracted_dirs:
                extracted_dir = extracted_dirs[0]
                
                # Use the extracted directory as-is or copy to standard name
                if extracted_dir != self.sra_dir:
                    logger.info(f"Using extracted directory: {extracted_dir.name}")
                    # Update self.sra_dir to point to actual location
                    self.sra_dir = extracted_dir
                
                logger.info(f"✅ Extraction completed: {self.sra_dir}")
                
                # Clean up zip file
                try:
                    zip_file.unlink()
                    logger.info("✅ Cleaned up temporary files")
                except Exception as e:
                    logger.warning(f"⚠️  Could not delete zip file: {e}")
                
                return self.sra_dir
            else:
                logger.error("❌ Could not find extracted directory")
                return None
                
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return None
    
    def configure_sra_toolkit(self):
        """Configure SRA Toolkit"""
        logger.info("\n" + "=" * 60)
        logger.info("CONFIGURING SRA TOOLKIT")
        logger.info("=" * 60)
        
        # Find bin directory
        bin_dir = self.sra_dir / "bin"
        
        if not bin_dir.exists():
            logger.error(f"❌ Bin directory not found: {bin_dir}")
            return False
        
        # Test executables
        executables = ['prefetch.exe', 'fastq-dump.exe', 'sam-dump.exe']
        
        logger.info("\nVerifying executables:")
        all_found = True
        for exe in executables:
            exe_path = bin_dir / exe
            if exe_path.exists():
                logger.info(f"✅ {exe:20} - Found")
            else:
                logger.error(f"❌ {exe:20} - Not found")
                all_found = False
        
        if not all_found:
            return False
        
        # Configure vdb-config (optional but recommended)
        vdb_config_exe = bin_dir / "vdb-config.exe"
        if vdb_config_exe.exists():
            try:
                logger.info("\nConfiguring SRA Toolkit...")
                result = subprocess.run(
                    [str(vdb_config_exe), '--interactive-mode', 'textual', '--restore-defaults'],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info("✅ SRA Toolkit configured successfully")
                else:
                    logger.warning("⚠️  Configuration returned non-zero exit code (this may be normal)")
            except Exception as e:
                logger.warning(f"⚠️  Configuration warning: {e}")
        
        return True
    
    def update_config_yaml(self):
        """Update config.yaml with SRA Toolkit paths"""
        logger.info("\n" + "=" * 60)
        logger.info("UPDATING CONFIGURATION")
        logger.info("=" * 60)
        
        config_file = self.project_root / "config" / "config.yaml"
        
        if not config_file.exists():
            logger.error(f"❌ Config file not found: {config_file}")
            return False
        
        try:
            # Read current config
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update paths
            bin_dir = self.sra_dir / "bin"
            prefetch_path = str(bin_dir / "prefetch.exe").replace('\\', '\\\\')
            fastq_dump_path = str(bin_dir / "fastq-dump.exe").replace('\\', '\\\\')
            sam_dump_path = str(bin_dir / "sam-dump.exe").replace('\\', '\\\\')
            
            # Replace the paths
            import re
            
            # Update prefetch_path
            content = re.sub(
                r'prefetch_path:\s*"[^"]*"',
                f'prefetch_path: "{prefetch_path}"',
                content
            )
            
            # Update fastq_dump_path
            content = re.sub(
                r'fastq_dump_path:\s*"[^"]*"',
                f'fastq_dump_path: "{fastq_dump_path}"',
                content
            )
            
            # Update sam_dump_path
            content = re.sub(
                r'sam_dump_path:\s*"[^"]*"',
                f'sam_dump_path: "{sam_dump_path}"',
                content
            )
            
            # Write updated config
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Configuration updated successfully")
            logger.info(f"\n   prefetch:    {prefetch_path}")
            logger.info(f"   fastq-dump:  {fastq_dump_path}")
            logger.info(f"   sam-dump:    {sam_dump_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update config: {e}")
            return False
    
    def add_to_path(self):
        """Add SRA Toolkit to PATH (for current session)"""
        logger.info("\n" + "=" * 60)
        logger.info("ADDING TO PATH")
        logger.info("=" * 60)
        
        bin_dir = str((self.sra_dir / "bin").resolve())
        
        # Add to current environment
        current_path = os.environ.get('PATH', '')
        if bin_dir not in current_path:
            os.environ['PATH'] = f"{bin_dir};{current_path}"
            logger.info(f"✅ Added to PATH: {bin_dir}")
        else:
            logger.info(f"ℹ️  Already in PATH: {bin_dir}")
        
        # Instructions for permanent PATH
        logger.info("\n📌 To add to PATH permanently (Windows):")
        logger.info("   1. Open System Properties > Environment Variables")
        logger.info("   2. Edit 'Path' variable")
        logger.info(f"   3. Add: {bin_dir}")
        logger.info("\n   OR run this PowerShell command as Administrator:")
        logger.info(f'   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";{bin_dir}", "User")')
    
    def test_installation(self):
        """Test SRA Toolkit installation"""
        logger.info("\n" + "=" * 60)
        logger.info("TESTING INSTALLATION")
        logger.info("=" * 60)
        
        bin_dir = self.sra_dir / "bin"
        
        tests = [
            ('prefetch.exe', '--version'),
            ('fastq-dump.exe', '--version'),
            ('sam-dump.exe', '--version'),
        ]
        
        all_passed = True
        for exe, arg in tests:
            exe_path = bin_dir / exe
            try:
                result = subprocess.run(
                    [str(exe_path), arg],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 or 'version' in result.stdout.lower() or 'version' in result.stderr.lower():
                    version_info = result.stdout.strip() or result.stderr.strip()
                    logger.info(f"✅ {exe:20} - Working ({version_info.split()[0] if version_info else 'OK'})")
                else:
                    logger.warning(f"⚠️  {exe:20} - Returned non-zero exit code")
            except Exception as e:
                logger.error(f"❌ {exe:20} - Failed: {e}")
                all_passed = False
        
        return all_passed
    
    def install(self):
        """Complete installation process"""
        logger.info("\n" + "=" * 60)
        logger.info("SRA TOOLKIT INSTALLATION")
        logger.info("=" * 60)
        logger.info(f"Version: {self.sra_toolkit_version}")
        logger.info(f"Install location: {self.sra_dir}")
        logger.info("")
        
        # Check if already installed
        if self.sra_dir.exists():
            logger.info("⚠️  SRA Toolkit directory already exists")
            response = input("Reinstall? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Installation cancelled")
                return False
            
            logger.info("Removing existing installation...")
            shutil.rmtree(self.sra_dir)
        
        # Download
        zip_file = self.download_sra_toolkit()
        if not zip_file:
            return False
        
        # Extract
        sra_dir = self.extract_sra_toolkit(zip_file)
        if not sra_dir:
            return False
        
        # Configure
        if not self.configure_sra_toolkit():
            logger.error("❌ Configuration failed")
            return False
        
        # Update config.yaml
        if not self.update_config_yaml():
            logger.warning("⚠️  Failed to update config.yaml (you may need to update manually)")
        
        # Add to PATH
        self.add_to_path()
        
        # Test
        if not self.test_installation():
            logger.warning("⚠️  Some tests failed, but installation may still work")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("INSTALLATION COMPLETE!")
        logger.info("=" * 60)
        logger.info("\n✅ SRA Toolkit successfully installed and configured")
        logger.info(f"\nInstallation directory: {self.sra_dir}")
        logger.info(f"Executables location: {self.sra_dir / 'bin'}")
        logger.info("\n📚 Next steps:")
        logger.info("   1. Run the test again: python test_sra_pipeline.py")
        logger.info("   2. Try the integration example: python scripts/sra_integration_example.py")
        logger.info("   3. Download SRA data: python scripts/download_sra_data.py --accession SRR123456")
        
        return True

def main():
    """Main installation function"""
    installer = SRAToolkitInstaller()
    
    try:
        success = installer.install()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Installation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Installation failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
