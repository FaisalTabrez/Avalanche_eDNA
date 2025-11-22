#!/usr/bin/env python3
"""
Restore Manager - Restore Avalanche from backups
"""
import os
import sys
import json
import shutil
import tarfile
import gzip
import logging
from pathlib import Path
from typing import Optional, Tuple
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import config


class RestoreManager:
    """Manages restoration from backups"""
    
    def __init__(self, backup_root: Optional[Path] = None):
        """Initialize restore manager
        
        Args:
            backup_root: Root directory for backups (default: data/backups)
        """
        self.backup_root = backup_root or Path(config.get('backup.root_dir', 'data/backups'))
        
        if not self.backup_root.exists():
            raise ValueError(f"Backup directory not found: {self.backup_root}")
        
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
        """Setup logging for restore operations"""
        logger = logging.getLogger('RestoreManager')
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
        return logger
    
    def list_backups(self, backup_type: Optional[str] = None):
        """List available backups
        
        Args:
            backup_type: Filter by type ('database', 'files', 'full')
        """
        backups = []
        
        for metadata_file in self.backup_root.rglob('*_metadata.json'):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    
                    if backup_type is None or data.get('backup_type') == backup_type:
                        backups.append(data)
            except Exception as e:
                self.logger.error(f"Error reading metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return backups
    
    def restore_database(self, backup_id: str, force: bool = False) -> bool:
        """Restore database from backup
        
        Args:
            backup_id: Backup ID to restore
            force: Skip confirmation prompts
            
        Returns:
            Success status
        """
        self.logger.info(f"Restoring database from backup: {backup_id}")
        
        # Find backup file
        backup_files = list(self.backup_root.rglob(f"{backup_id}.*"))
        backup_files = [f for f in backup_files if not f.name.endswith('_metadata.json')]
        
        if not backup_files:
            self.logger.error(f"Backup file not found: {backup_id}")
            return False
        
        backup_file = backup_files[0]
        
        # Confirm restoration
        if not force:
            print(f"\n⚠️  WARNING: This will overwrite the current database!")
            print(f"Backup file: {backup_file}")
            print(f"Database: {self.db_type} - {self.db_name}")
            response = input("\nContinue? (yes/no): ")
            
            if response.lower() != 'yes':
                self.logger.info("Restore cancelled by user")
                return False
        
        try:
            # Decompress if needed
            if backup_file.suffix == '.gz':
                backup_file = self._decompress_file(backup_file)
            
            # Restore based on database type
            if self.db_type == 'postgresql':
                success = self._restore_postgresql(backup_file)
            elif self.db_type == 'sqlite':
                success = self._restore_sqlite(backup_file)
            else:
                self.logger.error(f"Unsupported database type: {self.db_type}")
                return False
            
            if success:
                self.logger.info("Database restore completed successfully")
            else:
                self.logger.error("Database restore failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Database restore failed: {e}", exc_info=True)
            return False
    
    def _restore_postgresql(self, backup_file: Path) -> bool:
        """Restore PostgreSQL database using psql
        
        Args:
            backup_file: SQL backup file
            
        Returns:
            Success status
        """
        self.logger.info("Restoring PostgreSQL database...")
        
        # Set password environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = self.db_password
        
        # Drop and recreate database
        self.logger.warning(f"Dropping database: {self.db_name}")
        
        drop_cmd = [
            'psql',
            '-h', self.db_host,
            '-p', self.db_port,
            '-U', self.db_user,
            '-d', 'postgres',
            '-c', f'DROP DATABASE IF EXISTS {self.db_name}'
        ]
        
        create_cmd = [
            'psql',
            '-h', self.db_host,
            '-p', self.db_port,
            '-U', self.db_user,
            '-d', 'postgres',
            '-c', f'CREATE DATABASE {self.db_name}'
        ]
        
        try:
            subprocess.run(drop_cmd, env=env, check=True, capture_output=True)
            subprocess.run(create_cmd, env=env, check=True, capture_output=True)
            
            # Restore from backup
            restore_cmd = [
                'psql',
                '-h', self.db_host,
                '-p', self.db_port,
                '-U', self.db_user,
                '-d', self.db_name,
                '-f', str(backup_file)
            ]
            
            result = subprocess.run(
                restore_cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info("PostgreSQL restore successful")
            return True
            
        except FileNotFoundError:
            self.logger.error("psql not found. Please install PostgreSQL client tools.")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"psql error: {e.stderr}")
            return False
    
    def _restore_sqlite(self, backup_file: Path) -> bool:
        """Restore SQLite database using file copy
        
        Args:
            backup_file: SQLite database file
            
        Returns:
            Success status
        """
        self.logger.info("Restoring SQLite database...")
        
        sqlite_path = Path(os.getenv('SQLITE_PATH', 'data/avalanche.db'))
        
        # Backup current database
        if sqlite_path.exists():
            backup_current = sqlite_path.with_suffix('.db.backup')
            self.logger.info(f"Backing up current database to: {backup_current}")
            shutil.copy2(sqlite_path, backup_current)
        
        try:
            # Copy backup to database location
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_file, sqlite_path)
            
            self.logger.info(f"SQLite restore successful: {sqlite_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"SQLite restore failed: {e}")
            
            # Restore from backup if available
            if backup_current.exists():
                self.logger.info("Restoring previous database...")
                shutil.copy2(backup_current, sqlite_path)
            
            return False
    
    def restore_files(self, backup_id: str, target_dir: Optional[Path] = None, force: bool = False) -> bool:
        """Restore files from backup
        
        Args:
            backup_id: Backup ID to restore
            target_dir: Target directory for restoration (default: project root)
            force: Skip confirmation prompts
            
        Returns:
            Success status
        """
        self.logger.info(f"Restoring files from backup: {backup_id}")
        
        # Find backup file
        backup_files = list(self.backup_root.rglob(f"{backup_id}.*"))
        backup_files = [f for f in backup_files if not f.name.endswith('_metadata.json')]
        
        if not backup_files:
            self.logger.error(f"Backup file not found: {backup_id}")
            return False
        
        backup_file = backup_files[0]
        target_dir = target_dir or project_root
        
        # Confirm restoration
        if not force:
            print(f"\n⚠️  WARNING: This will overwrite files in: {target_dir}")
            print(f"Backup file: {backup_file}")
            response = input("\nContinue? (yes/no): ")
            
            if response.lower() != 'yes':
                self.logger.info("Restore cancelled by user")
                return False
        
        try:
            # Decompress if needed
            if backup_file.suffix == '.gz':
                backup_file = self._decompress_file(backup_file)
            
            # Extract tar archive
            self.logger.info(f"Extracting archive to: {target_dir}")
            
            with tarfile.open(backup_file, 'r') as tar:
                tar.extractall(target_dir)
            
            self.logger.info("File restore completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"File restore failed: {e}", exc_info=True)
            return False
    
    def _decompress_file(self, compressed_file: Path) -> Path:
        """Decompress gzipped file
        
        Args:
            compressed_file: Gzipped file
            
        Returns:
            Decompressed file path
        """
        self.logger.info(f"Decompressing {compressed_file}...")
        
        decompressed_path = compressed_file.with_suffix('')
        
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return decompressed_path


def main():
    """Main entry point for restore script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Avalanche Restore Manager')
    parser.add_argument(
        'action',
        choices=['list', 'database', 'files'],
        help='Restore action to perform'
    )
    parser.add_argument('--backup-id', help='Backup ID to restore')
    parser.add_argument('--type', choices=['database', 'files'], help='Backup type filter (for list action)')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--target-dir', help='Target directory for file restoration')
    
    args = parser.parse_args()
    
    manager = RestoreManager()
    
    if args.action == 'list':
        backups = manager.list_backups(backup_type=args.type)
        print(f"\nFound {len(backups)} backups:\n")
        for backup in backups:
            print(f"ID: {backup['backup_id']}")
            print(f"Type: {backup['backup_type']}")
            print(f"Date: {backup['timestamp']}")
            print(f"Size: {backup['size_bytes'] / (1024**2):.2f} MB")
            print(f"Status: {backup['status']}")
            print("-" * 60)
    
    elif args.action == 'database':
        if not args.backup_id:
            print("Error: --backup-id required for database restore")
            sys.exit(1)
        success = manager.restore_database(args.backup_id, force=args.force)
        sys.exit(0 if success else 1)
    
    elif args.action == 'files':
        if not args.backup_id:
            print("Error: --backup-id required for files restore")
            sys.exit(1)
        target_dir = Path(args.target_dir) if args.target_dir else None
        success = manager.restore_files(args.backup_id, target_dir=target_dir, force=args.force)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
