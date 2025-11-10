"""
Automated NCBI Taxdump Updating System

This module provides utilities for:
- Automatically downloading the latest NCBI taxdump files
- Checking for updates and managing versions
- Optimizing lineage lookup performance
- Backup and rollback functionality
- Scheduled updating capabilities
"""

import os
import sys
import hashlib
import shutil
import gzip
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import urllib.request
import urllib.error
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class TaxdumpVersion:
    """Information about a taxdump version"""
    version_date: datetime
    download_url: str
    local_path: Path
    file_hash: str
    file_size: int
    is_current: bool = False
    backup_path: Optional[Path] = None

class TaxdumpUpdater:
    """Manages NCBI taxdump downloads and updates"""
    
    def __init__(self, 
                 taxdump_dir: str,
                 backup_dir: Optional[str] = None,
                 keep_backups: int = 3):
        self.taxdump_dir = Path(taxdump_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.taxdump_dir / "backups"
        self.keep_backups = keep_backups
        
        # NCBI taxdump URL
        self.taxdump_url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
        self.taxdump_md5_url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz.md5"
        
        # Version tracking file
        self.version_file = self.taxdump_dir / "version_info.json"
        
        # Required taxdump files
        self.required_files = ["names.dmp", "nodes.dmp", "merged.dmp"]
        
        # Initialize directories
        self._init_directories()
        
    def _init_directories(self) -> None:
        """Initialize required directories"""
        self.taxdump_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Taxdump directory: {self.taxdump_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")
    
    def check_for_updates(self) -> Tuple[bool, Optional[str]]:
        """Check if newer taxdump version is available"""
        try:
            # Get remote file info
            remote_info = self._get_remote_file_info()
            if not remote_info:
                return False, "Could not retrieve remote file information"
            
            # Get current version info
            current_info = self._get_current_version_info()
            
            if not current_info:
                return True, "No local taxdump found - update needed"
            
            # Compare hashes
            if remote_info['hash'] != current_info.get('file_hash'):
                return True, f"New version available (hash changed)"
            
            return False, "Local taxdump is up to date"
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return False, f"Error checking for updates: {e}"
    
    def _get_remote_file_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the remote taxdump file"""
        try:
            # Get MD5 hash from NCBI
            with urllib.request.urlopen(self.taxdump_md5_url, timeout=30) as response:
                md5_content = response.read().decode('utf-8').strip()
                remote_hash = md5_content.split()[0]  # First part is the hash
            
            # Get file size and last-modified date
            req = urllib.request.Request(self.taxdump_url)
            req.get_method = lambda: 'HEAD'
            
            with urllib.request.urlopen(req, timeout=30) as response:
                headers = response.headers
                file_size = int(headers.get('Content-Length', 0))
                last_modified = headers.get('Last-Modified')
                
                # Parse last-modified date
                if last_modified:
                    from email.utils import parsedate_to_datetime
                    modified_date = parsedate_to_datetime(last_modified)
                else:
                    modified_date = datetime.now()
            
            return {
                'hash': remote_hash,
                'size': file_size,
                'last_modified': modified_date,
                'url': self.taxdump_url
            }
            
        except Exception as e:
            logger.error(f"Error getting remote file info: {e}")
            return None
    
    def _get_current_version_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current local version"""
        if not self.version_file.exists():
            return None
        
        try:
            with open(self.version_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading version file: {e}")
            return None
    
    def _save_version_info(self, version_info: Dict[str, Any]) -> None:
        """Save version information to file"""
        try:
            version_info['last_updated'] = datetime.now().isoformat()
            with open(self.version_file, 'w') as f:
                json.dump(version_info, f, indent=2, default=str)
            logger.info(f"Version info saved: {version_info['file_hash'][:8]}...")
        except Exception as e:
            logger.error(f"Error saving version info: {e}")
    
    def download_and_update(self, force: bool = False) -> Tuple[bool, str]:
        """Download and install new taxdump"""
        try:
            # Check if update needed (unless forced)
            if not force:
                needs_update, reason = self.check_for_updates()
                if not needs_update:
                    return True, f"No update needed: {reason}"
            
            logger.info("Starting taxdump download and update...")
            
            # Create backup of current version
            backup_success = self._create_backup()
            if not backup_success:
                logger.warning("Backup creation failed, but continuing with update")
            
            # Download new taxdump
            download_path = self._download_taxdump()
            if not download_path:
                return False, "Download failed"
            
            # Extract and validate
            extract_success = self._extract_taxdump(download_path)
            if not extract_success:
                return False, "Extraction/validation failed"
            
            # Get file info for version tracking
            file_hash = self._calculate_file_hash(download_path)
            file_size = download_path.stat().st_size
            
            # Save version info
            version_info = {
                'file_hash': file_hash,
                'file_size': file_size,
                'download_date': datetime.now().isoformat(),
                'download_url': self.taxdump_url,
                'backup_available': backup_success
            }
            self._save_version_info(version_info)
            
            # Cleanup temporary download
            download_path.unlink(missing_ok=True)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info("Taxdump update completed successfully")
            return True, "Update completed successfully"
            
        except Exception as e:
            logger.error(f"Error during taxdump update: {e}")
            return False, f"Update failed: {e}"
    
    def _download_taxdump(self) -> Optional[Path]:
        """Download taxdump file"""
        try:
            # Create temporary download path
            temp_dir = tempfile.mkdtemp(prefix="taxdump_download_")
            download_path = Path(temp_dir) / "taxdump.tar.gz"
            
            logger.info(f"Downloading taxdump from {self.taxdump_url}")
            
            # Download with progress tracking
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    if block_num % 100 == 0:  # Log every 100 blocks to avoid spam
                        logger.info(f"Download progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
            
            urllib.request.urlretrieve(
                self.taxdump_url,
                download_path,
                reporthook=progress_hook
            )
            
            logger.info(f"Download completed: {download_path}")
            return download_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def _extract_taxdump(self, archive_path: Path) -> bool:
        """Extract and validate taxdump archive"""
        try:
            logger.info("Extracting taxdump archive...")
            
            # Extract to temporary directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract tar.gz
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_path)
                
                # Validate required files exist
                missing_files = []
                for required_file in self.required_files:
                    if not (temp_path / required_file).exists():
                        missing_files.append(required_file)
                
                if missing_files:
                    logger.error(f"Missing required files: {missing_files}")
                    return False
                
                # Move files to final destination
                for file_path in temp_path.glob("*.dmp"):
                    dest_path = self.taxdump_dir / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    logger.info(f"Installed: {file_path.name}")
                
                # Also move any other taxonomy files
                for file_path in temp_path.glob("*.txt"):
                    dest_path = self.taxdump_dir / file_path.name
                    shutil.move(str(file_path), str(dest_path))
            
            logger.info("Taxdump extraction completed")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """Create backup of current taxdump"""
        try:
            if not any((self.taxdump_dir / f).exists() for f in self.required_files):
                logger.info("No existing taxdump to backup")
                return True
            
            backup_name = f"taxdump_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all .dmp files
            backed_up_files = []
            for file_path in self.taxdump_dir.glob("*.dmp"):
                dest_path = backup_path / file_path.name
                shutil.copy2(file_path, dest_path)
                backed_up_files.append(file_path.name)
            
            # Copy version info if it exists
            if self.version_file.exists():
                shutil.copy2(self.version_file, backup_path / "version_info.json")
            
            logger.info(f"Created backup: {backup_path} ({len(backed_up_files)} files)")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backups beyond the keep limit"""
        try:
            # Get all backup directories
            backups = []
            for backup_path in self.backup_dir.iterdir():
                if backup_path.is_dir() and backup_path.name.startswith("taxdump_backup_"):
                    try:
                        # Extract date from directory name
                        date_str = backup_path.name.replace("taxdump_backup_", "")
                        backup_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        backups.append((backup_date, backup_path))
                    except ValueError:
                        logger.warning(f"Could not parse backup date: {backup_path.name}")
            
            # Sort by date (newest first)
            backups.sort(key=lambda x: x[0], reverse=True)
            
            # Remove excess backups
            if len(backups) > self.keep_backups:
                to_remove = backups[self.keep_backups:]
                for _, backup_path in to_remove:
                    shutil.rmtree(backup_path)
                    logger.info(f"Removed old backup: {backup_path.name}")
            
        except Exception as e:
            logger.warning(f"Backup cleanup failed: {e}")
    
    def restore_backup(self, backup_name: Optional[str] = None) -> Tuple[bool, str]:
        """Restore from backup"""
        try:
            # Find backup to restore
            if backup_name:
                backup_path = self.backup_dir / backup_name
                if not backup_path.exists():
                    return False, f"Backup not found: {backup_name}"
            else:
                # Find most recent backup
                backups = []
                for backup_path in self.backup_dir.iterdir():
                    if backup_path.is_dir() and backup_path.name.startswith("taxdump_backup_"):
                        try:
                            date_str = backup_path.name.replace("taxdump_backup_", "")
                            backup_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                            backups.append((backup_date, backup_path))
                        except ValueError:
                            continue
                
                if not backups:
                    return False, "No backups available"
                
                backups.sort(key=lambda x: x[0], reverse=True)
                backup_path = backups[0][1]
            
            logger.info(f"Restoring backup: {backup_path.name}")
            
            # Restore files
            restored_files = []
            for file_path in backup_path.glob("*.dmp"):
                dest_path = self.taxdump_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                restored_files.append(file_path.name)
            
            # Restore version info
            backup_version_file = backup_path / "version_info.json"
            if backup_version_file.exists():
                shutil.copy2(backup_version_file, self.version_file)
            
            logger.info(f"Backup restored successfully ({len(restored_files)} files)")
            return True, f"Restored {len(restored_files)} files from {backup_path.name}"
            
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            return False, f"Restore failed: {e}"
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir() and backup_path.name.startswith("taxdump_backup_"):
                try:
                    date_str = backup_path.name.replace("taxdump_backup_", "")
                    backup_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    
                    # Count files
                    file_count = len(list(backup_path.glob("*.dmp")))
                    
                    # Get size
                    total_size = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())
                    
                    backups.append({
                        'name': backup_path.name,
                        'date': backup_date,
                        'file_count': file_count,
                        'size_bytes': total_size,
                        'path': str(backup_path)
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing backup {backup_path.name}: {e}")
        
        # Sort by date (newest first)
        backups.sort(key=lambda x: x['date'], reverse=True)
        return backups
    
    def get_status(self) -> Dict[str, Any]:
        """Get current taxdump status"""
        status = {
            'taxdump_dir': str(self.taxdump_dir),
            'has_required_files': all((self.taxdump_dir / f).exists() for f in self.required_files),
            'required_files': self.required_files,
            'existing_files': [],
            'version_info': None,
            'last_check': None,
            'backups_available': 0
        }
        
        # Check existing files
        for file_path in self.taxdump_dir.glob("*.dmp"):
            file_stat = file_path.stat()
            status['existing_files'].append({
                'name': file_path.name,
                'size': file_stat.st_size,
                'modified': datetime.fromtimestamp(file_stat.st_mtime)
            })
        
        # Get version info
        status['version_info'] = self._get_current_version_info()
        
        # Count backups
        status['backups_available'] = len(self.list_backups())
        
        return status
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def schedule_updates(self, check_interval_days: int = 7) -> None:
        """Schedule automatic update checks (basic implementation)"""
        # This is a basic implementation - in production you might want to use
        # a proper task scheduler like celery, APScheduler, or system cron
        
        logger.info(f"Scheduled update checks every {check_interval_days} days")
        
        # For now, just log the schedule - actual implementation would depend on
        # the application's architecture and deployment environment
        schedule_info = {
            'enabled': True,
            'check_interval_days': check_interval_days,
            'next_check': (datetime.now() + timedelta(days=check_interval_days)).isoformat(),
            'auto_update': False  # Manual approval required by default
        }
        
        # Save schedule info
        schedule_file = self.taxdump_dir / "update_schedule.json"
        with open(schedule_file, 'w') as f:
            json.dump(schedule_info, f, indent=2)

def create_updater_from_config() -> TaxdumpUpdater:
    """Create taxdump updater from configuration"""
    taxonomy_config = config.get('taxonomy', {})
    
    taxdump_dir = taxonomy_config.get('taxdump_dir')
    if not taxdump_dir:
        raise ValueError("taxdump_dir not configured")
    
    backup_dir = taxonomy_config.get('taxdump_backup_dir')
    keep_backups = taxonomy_config.get('keep_taxdump_backups', 3)
    
    return TaxdumpUpdater(
        taxdump_dir=taxdump_dir,
        backup_dir=backup_dir,
        keep_backups=keep_backups
    )

# Command-line interface for manual updates
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NCBI Taxdump Updater")
    parser.add_argument("--taxdump-dir", required=True, help="Taxdump directory path")
    parser.add_argument("--backup-dir", help="Backup directory path")
    parser.add_argument("--check", action="store_true", help="Check for updates only")
    parser.add_argument("--update", action="store_true", help="Download and install updates")
    parser.add_argument("--force", action="store_true", help="Force update even if current")
    parser.add_argument("--restore", help="Restore from backup (backup name)")
    parser.add_argument("--list-backups", action="store_true", help="List available backups")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    args = parser.parse_args()
    
    # Setup logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    updater = TaxdumpUpdater(args.taxdump_dir, args.backup_dir)
    
    if args.check:
        needs_update, reason = updater.check_for_updates()
        print(f"Update needed: {needs_update}")
        print(f"Reason: {reason}")
    
    elif args.update:
        success, message = updater.download_and_update(force=args.force)
        print(f"Update {'successful' if success else 'failed'}: {message}")
    
    elif args.restore:
        success, message = updater.restore_backup(args.restore if args.restore != "latest" else None)
        print(f"Restore {'successful' if success else 'failed'}: {message}")
    
    elif args.list_backups:
        backups = updater.list_backups()
        if backups:
            print(f"Available backups ({len(backups)}):")
            for backup in backups:
                size_mb = backup['size_bytes'] / (1024 * 1024)
                print(f"  - {backup['name']} ({backup['date'].strftime('%Y-%m-%d %H:%M')})")
                print(f"    Files: {backup['file_count']}, Size: {size_mb:.1f} MB")
        else:
            print("No backups available")
    
    elif args.status:
        status = updater.get_status()
        print(f"Taxdump Status:")
        print(f"  Directory: {status['taxdump_dir']}")
        print(f"  Has required files: {status['has_required_files']}")
        print(f"  Existing files: {len(status['existing_files'])}")
        print(f"  Backups available: {status['backups_available']}")
        
        if status['version_info']:
            vi = status['version_info']
            print(f"  Current version: {vi.get('file_hash', 'Unknown')[:8]}...")
            print(f"  Last updated: {vi.get('last_updated', 'Unknown')}")
    
    else:
        parser.print_help()