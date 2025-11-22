#!/usr/bin/env python3
"""
Backup Manager - Comprehensive backup system for Avalanche
Supports PostgreSQL, file system, and cloud storage backups
"""
import os
import sys
import json
import shutil
import logging
import hashlib
import tarfile
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess

# Optional cloud dependencies
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import config


@dataclass
class BackupMetadata:
    """Metadata for a backup"""
    backup_id: str
    timestamp: str
    backup_type: str  # 'database', 'files', 'full'
    size_bytes: int
    checksum: str
    files_included: List[str]
    compressed: bool
    encrypted: bool
    status: str  # 'success', 'failed', 'in_progress'
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class BackupManager:
    """Manages automated backups for Avalanche"""
    
    def __init__(self, backup_root: Optional[Path] = None):
        """Initialize backup manager
        
        Args:
            backup_root: Root directory for backups (default: data/backups)
        """
        self.backup_root = backup_root or Path(config.get('backup.root_dir', 'data/backups'))
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Backup configuration
        self.retention_policy = {
            'daily': config.get('backup.retention.daily', 7),
            'weekly': config.get('backup.retention.weekly', 4),
            'monthly': config.get('backup.retention.monthly', 12)
        }
        
        self.compression_enabled = config.get('backup.compression', True)
        self.encryption_enabled = config.get('backup.encryption', False)
        
        # Cloud storage configuration
        self.cloud_enabled = config.get('backup.cloud.enabled', False)
        self.cloud_provider = config.get('backup.cloud.provider', 'local')  # 's3', 'azure', 'gcs'
        
        # Database configuration
        self.db_type = os.getenv('DB_TYPE', 'sqlite')
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'avalanche_edna')
        self.db_user = os.getenv('DB_USER', 'avalanche')
        self.db_password = os.getenv('DB_PASSWORD', '')
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for backup operations"""
        log_dir = self.backup_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger('BackupManager')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = log_dir / f'backup_{datetime.now().strftime("%Y%m%d")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def create_backup_id(self, backup_type: str) -> str:
        """Generate unique backup ID
        
        Args:
            backup_type: Type of backup ('database', 'files', 'full')
            
        Returns:
            Unique backup ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{backup_type}_{timestamp}"
    
    def backup_database(self, output_dir: Optional[Path] = None) -> Tuple[bool, Optional[BackupMetadata]]:
        """Backup PostgreSQL or SQLite database
        
        Args:
            output_dir: Directory to store backup (default: backup_root/database)
            
        Returns:
            (success, metadata)
        """
        output_dir = output_dir or (self.backup_root / 'database')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        backup_id = self.create_backup_id('database')
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Starting database backup: {backup_id}")
        
        try:
            if self.db_type == 'postgresql':
                backup_file = output_dir / f"{backup_id}.sql"
                success = self._backup_postgresql(backup_file)
            elif self.db_type == 'sqlite':
                backup_file = output_dir / f"{backup_id}.db"
                success = self._backup_sqlite(backup_file)
            else:
                self.logger.error(f"Unsupported database type: {self.db_type}")
                return False, None
            
            if not success:
                return False, None
            
            # Compress if enabled
            if self.compression_enabled:
                compressed_file = self._compress_file(backup_file)
                backup_file.unlink()  # Remove uncompressed file
                backup_file = compressed_file
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_file)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type='database',
                size_bytes=backup_file.stat().st_size,
                checksum=checksum,
                files_included=[str(backup_file.name)],
                compressed=self.compression_enabled,
                encrypted=self.encryption_enabled,
                status='success'
            )
            
            # Save metadata
            self._save_metadata(metadata, output_dir)
            
            self.logger.info(f"Database backup completed: {backup_file}")
            self.logger.info(f"Backup size: {metadata.size_bytes / (1024**2):.2f} MB")
            
            return True, metadata
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}", exc_info=True)
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type='database',
                size_bytes=0,
                checksum='',
                files_included=[],
                compressed=False,
                encrypted=False,
                status='failed',
                error_message=str(e)
            )
            return False, metadata
    
    def _backup_postgresql(self, output_file: Path) -> bool:
        """Backup PostgreSQL database using pg_dump
        
        Args:
            output_file: Output file path
            
        Returns:
            Success status
        """
        self.logger.info("Backing up PostgreSQL database...")
        
        # Set password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_password
        
        # Build pg_dump command
        cmd = [
            'pg_dump',
            '-h', self.db_host,
            '-p', self.db_port,
            '-U', self.db_user,
            '-d', self.db_name,
            '-F', 'p',  # Plain text format
            '-f', str(output_file)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                self.logger.info("PostgreSQL backup successful")
                return True
            else:
                self.logger.error(f"pg_dump failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            self.logger.error("pg_dump not found. Please install PostgreSQL client tools.")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"pg_dump error: {e.stderr}")
            return False
    
    def _backup_sqlite(self, output_file: Path) -> bool:
        """Backup SQLite database using file copy
        
        Args:
            output_file: Output file path
            
        Returns:
            Success status
        """
        self.logger.info("Backing up SQLite database...")
        
        sqlite_path = Path(os.getenv('SQLITE_PATH', 'data/avalanche.db'))
        
        if not sqlite_path.exists():
            self.logger.error(f"SQLite database not found: {sqlite_path}")
            return False
        
        try:
            shutil.copy2(sqlite_path, output_file)
            self.logger.info(f"SQLite backup successful: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"SQLite backup failed: {e}")
            return False
    
    def backup_files(self, paths: List[Path], output_dir: Optional[Path] = None) -> Tuple[bool, Optional[BackupMetadata]]:
        """Backup file system directories
        
        Args:
            paths: List of paths to backup
            output_dir: Directory to store backup (default: backup_root/files)
            
        Returns:
            (success, metadata)
        """
        output_dir = output_dir or (self.backup_root / 'files')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        backup_id = self.create_backup_id('files')
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Starting file backup: {backup_id}")
        self.logger.info(f"Backing up {len(paths)} paths")
        
        try:
            # Create tar archive
            archive_path = output_dir / f"{backup_id}.tar"
            
            with tarfile.open(archive_path, 'w') as tar:
                for path in paths:
                    if path.exists():
                        self.logger.info(f"Adding to archive: {path}")
                        tar.add(path, arcname=path.name)
                    else:
                        self.logger.warning(f"Path not found, skipping: {path}")
            
            # Compress if enabled
            if self.compression_enabled:
                compressed_file = self._compress_file(archive_path)
                archive_path.unlink()
                archive_path = compressed_file
            
            # Calculate checksum
            checksum = self._calculate_checksum(archive_path)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type='files',
                size_bytes=archive_path.stat().st_size,
                checksum=checksum,
                files_included=[str(p) for p in paths if p.exists()],
                compressed=self.compression_enabled,
                encrypted=self.encryption_enabled,
                status='success'
            )
            
            # Save metadata
            self._save_metadata(metadata, output_dir)
            
            self.logger.info(f"File backup completed: {archive_path}")
            self.logger.info(f"Backup size: {metadata.size_bytes / (1024**2):.2f} MB")
            
            return True, metadata
            
        except Exception as e:
            self.logger.error(f"File backup failed: {e}", exc_info=True)
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type='files',
                size_bytes=0,
                checksum='',
                files_included=[],
                compressed=False,
                encrypted=False,
                status='failed',
                error_message=str(e)
            )
            return False, metadata
    
    def backup_full(self) -> Tuple[bool, List[BackupMetadata]]:
        """Perform full system backup (database + files)
        
        Returns:
            (success, list of metadata)
        """
        self.logger.info("Starting full system backup")
        
        all_metadata = []
        
        # Backup database
        db_success, db_metadata = self.backup_database()
        if db_metadata:
            all_metadata.append(db_metadata)
        
        # Backup important directories
        paths_to_backup = [
            Path('data/datasets'),
            Path('data/raw'),
            Path('data/processed'),
            Path('data/reference'),
            Path('data/report_storage'),
            Path('consolidated_data'),
            Path('config')
        ]
        
        files_success, files_metadata = self.backup_files(paths_to_backup)
        if files_metadata:
            all_metadata.append(files_metadata)
        
        overall_success = db_success and files_success
        
        if overall_success:
            self.logger.info("Full backup completed successfully")
        else:
            self.logger.warning("Full backup completed with errors")
        
        return overall_success, all_metadata
    
    def _compress_file(self, file_path: Path) -> Path:
        """Compress file using gzip
        
        Args:
            file_path: File to compress
            
        Returns:
            Compressed file path
        """
        self.logger.info(f"Compressing {file_path}...")
        
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        original_size = file_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100
        
        self.logger.info(f"Compression: {original_size / (1024**2):.2f} MB → {compressed_size / (1024**2):.2f} MB ({ratio:.1f}% reduction)")
        
        return compressed_path
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum
        
        Args:
            file_path: File to checksum
            algorithm: Hash algorithm ('sha256', 'md5')
            
        Returns:
            Hexadecimal checksum
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _save_metadata(self, metadata: BackupMetadata, output_dir: Path):
        """Save backup metadata to JSON file
        
        Args:
            metadata: Backup metadata
            output_dir: Directory to save metadata
        """
        metadata_file = output_dir / f"{metadata.backup_id}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        self.logger.info(f"Metadata saved: {metadata_file}")
    
    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupMetadata]:
        """List all available backups
        
        Args:
            backup_type: Filter by type ('database', 'files', 'full')
            
        Returns:
            List of backup metadata
        """
        backups = []
        
        for metadata_file in self.backup_root.rglob('*_metadata.json'):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata = BackupMetadata.from_dict(data)
                    
                    if backup_type is None or metadata.backup_type == backup_type:
                        backups.append(metadata)
            except Exception as e:
                self.logger.error(f"Error reading metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    def cleanup_old_backups(self):
        """Remove backups according to retention policy"""
        self.logger.info("Starting backup cleanup...")
        
        now = datetime.now()
        
        for backup in self.list_backups():
            backup_date = datetime.fromisoformat(backup.timestamp)
            age_days = (now - backup_date).days
            
            should_delete = False
            
            # Daily retention
            if age_days > self.retention_policy['daily']:
                # Check if it's a weekly backup (Sunday)
                if backup_date.weekday() == 6:  # Sunday
                    # Check if it's within weekly retention
                    age_weeks = age_days // 7
                    if age_weeks > self.retention_policy['weekly']:
                        # Check if it's a monthly backup (first of month)
                        if backup_date.day == 1:
                            # Check if it's within monthly retention
                            age_months = (now.year - backup_date.year) * 12 + (now.month - backup_date.month)
                            if age_months > self.retention_policy['monthly']:
                                should_delete = True
                        else:
                            should_delete = True
                else:
                    should_delete = True
            
            if should_delete:
                self._delete_backup(backup)
    
    def _delete_backup(self, metadata: BackupMetadata):
        """Delete a backup and its metadata
        
        Args:
            metadata: Backup metadata
        """
        self.logger.info(f"Deleting old backup: {metadata.backup_id}")
        
        # Find and delete backup files
        for backup_file in self.backup_root.rglob(f"{metadata.backup_id}*"):
            try:
                backup_file.unlink()
                self.logger.info(f"Deleted: {backup_file}")
            except Exception as e:
                self.logger.error(f"Failed to delete {backup_file}: {e}")
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity using checksum
        
        Args:
            backup_id: Backup ID to verify
            
        Returns:
            True if backup is valid
        """
        self.logger.info(f"Verifying backup: {backup_id}")
        
        # Find metadata file
        metadata_files = list(self.backup_root.rglob(f"{backup_id}_metadata.json"))
        
        if not metadata_files:
            self.logger.error(f"Metadata not found for backup: {backup_id}")
            return False
        
        metadata_file = metadata_files[0]
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = BackupMetadata.from_dict(data)
            
            # Find backup file
            backup_files = list(self.backup_root.rglob(f"{backup_id}.*"))
            backup_files = [f for f in backup_files if not f.name.endswith('_metadata.json')]
            
            if not backup_files:
                self.logger.error(f"Backup file not found: {backup_id}")
                return False
            
            backup_file = backup_files[0]
            
            # Calculate current checksum
            current_checksum = self._calculate_checksum(backup_file)
            
            if current_checksum == metadata.checksum:
                self.logger.info(f"Backup verification successful: {backup_id}")
                return True
            else:
                self.logger.error(f"Checksum mismatch for backup: {backup_id}")
                self.logger.error(f"Expected: {metadata.checksum}")
                self.logger.error(f"Got: {current_checksum}")
                return False
                
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False
    
    def upload_to_cloud(self, backup_id: str) -> bool:
        """Upload backup to cloud storage
        
        Args:
            backup_id: Backup ID to upload
            
        Returns:
            Success status
        """
        if not self.cloud_enabled:
            self.logger.warning("Cloud backup is disabled")
            return False
        
        self.logger.info(f"Uploading backup to cloud: {backup_id}")
        
        if self.cloud_provider == 's3':
            return self._upload_to_s3(backup_id)
        elif self.cloud_provider == 'azure':
            return self._upload_to_azure(backup_id)
        elif self.cloud_provider == 'gcs':
            return self._upload_to_gcs(backup_id)
        else:
            self.logger.error(f"Unsupported cloud provider: {self.cloud_provider}")
            return False
    
    def _upload_to_s3(self, backup_id: str) -> bool:
        """Upload backup to AWS S3
        
        Args:
            backup_id: Backup ID to upload
            
        Returns:
            Success status
        """
        if not BOTO3_AVAILABLE:
            self.logger.error("boto3 not installed. Install with: pip install boto3")
            return False
            
        try:
            s3_client = boto3.client('s3')
            bucket = config.get('backup.cloud.s3.bucket')
            prefix = config.get('backup.cloud.s3.prefix', 'backups/')
            
            # Find backup files
            backup_files = list(self.backup_root.rglob(f"{backup_id}*"))
            
            for backup_file in backup_files:
                key = f"{prefix}{backup_file.name}"
                self.logger.info(f"Uploading to s3://{bucket}/{key}")
                
                s3_client.upload_file(
                    str(backup_file),
                    bucket,
                    key
                )
            
            self.logger.info("Cloud upload successful")
            return True
            
        except Exception as e:
            self.logger.error(f"S3 upload failed: {e}")
            return False
    
    def _upload_to_azure(self, backup_id: str) -> bool:
        """Upload backup to Azure Blob Storage
        
        Args:
            backup_id: Backup ID to upload
            
        Returns:
            Success status
        """
        if not AZURE_AVAILABLE:
            self.logger.error("azure-storage-blob not installed. Install with: pip install azure-storage-blob")
            return False
            
        try:
            connection_string = config.get('backup.cloud.azure.connection_string')
            container = config.get('backup.cloud.azure.container')
            prefix = config.get('backup.cloud.azure.prefix', 'backups/')
            
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service.get_container_client(container)
            
            # Find backup files
            backup_files = list(self.backup_root.rglob(f"{backup_id}*"))
            
            for backup_file in backup_files:
                blob_name = f"{prefix}{backup_file.name}"
                self.logger.info(f"Uploading to Azure: {blob_name}")
                
                blob_client = container_client.get_blob_client(blob_name)
                
                with open(backup_file, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
            
            self.logger.info("Azure upload successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Azure upload failed: {e}")
            return False
    
    def _upload_to_gcs(self, backup_id: str) -> bool:
        """Upload backup to Google Cloud Storage
        
        Args:
            backup_id: Backup ID to upload
            
        Returns:
            Success status
        """
        try:
            from google.cloud import storage
            
            bucket_name = config.get('backup.cloud.gcs.bucket')
            prefix = config.get('backup.cloud.gcs.prefix', 'backups/')
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Find backup files
            backup_files = list(self.backup_root.rglob(f"{backup_id}*"))
            
            for backup_file in backup_files:
                blob_name = f"{prefix}{backup_file.name}"
                self.logger.info(f"Uploading to GCS: gs://{bucket_name}/{blob_name}")
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(backup_file))
            
            self.logger.info("GCS upload successful")
            return True
            
        except ImportError:
            self.logger.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            return False
        except Exception as e:
            self.logger.error(f"GCS upload failed: {e}")
            return False


def main():
    """Main entry point for backup script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Avalanche Backup Manager')
    parser.add_argument(
        'action',
        choices=['database', 'files', 'full', 'list', 'cleanup', 'verify', 'upload'],
        help='Backup action to perform'
    )
    parser.add_argument('--backup-id', help='Backup ID (for verify and upload actions)')
    parser.add_argument('--type', choices=['database', 'files', 'full'], help='Backup type filter (for list action)')
    
    args = parser.parse_args()
    
    manager = BackupManager()
    
    if args.action == 'database':
        success, metadata = manager.backup_database()
        sys.exit(0 if success else 1)
    
    elif args.action == 'files':
        paths = [
            Path('data/datasets'),
            Path('data/raw'),
            Path('data/processed'),
            Path('data/reference'),
            Path('data/report_storage'),
            Path('consolidated_data'),
            Path('config')
        ]
        success, metadata = manager.backup_files(paths)
        sys.exit(0 if success else 1)
    
    elif args.action == 'full':
        success, metadata_list = manager.backup_full()
        sys.exit(0 if success else 1)
    
    elif args.action == 'list':
        backups = manager.list_backups(backup_type=args.type)
        print(f"\nFound {len(backups)} backups:\n")
        for backup in backups:
            print(f"ID: {backup.backup_id}")
            print(f"Type: {backup.backup_type}")
            print(f"Date: {backup.timestamp}")
            print(f"Size: {backup.size_bytes / (1024**2):.2f} MB")
            print(f"Status: {backup.status}")
            print(f"Compressed: {backup.compressed}")
            print("-" * 60)
    
    elif args.action == 'cleanup':
        manager.cleanup_old_backups()
    
    elif args.action == 'verify':
        if not args.backup_id:
            print("Error: --backup-id required for verify action")
            sys.exit(1)
        success = manager.verify_backup(args.backup_id)
        sys.exit(0 if success else 1)
    
    elif args.action == 'upload':
        if not args.backup_id:
            print("Error: --backup-id required for upload action")
            sys.exit(1)
        success = manager.upload_to_cloud(args.backup_id)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
