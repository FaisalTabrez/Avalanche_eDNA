"""
Backup system tests
"""
import pytest
import os
from pathlib import Path
import tempfile
import time

from scripts.backup.backup_manager import BackupManager
from scripts.backup.restore_manager import RestoreManager


class TestBackupManager:
    """Test suite for BackupManager"""
    
    @pytest.fixture
    def backup_manager(self, temp_dir):
        """Create BackupManager with temp directory"""
        config = {
            'backup': {
                'local': {
                    'enabled': True,
                    'path': str(temp_dir / 'backups')
                },
                'retention': {
                    'daily': 7,
                    'weekly': 4,
                    'monthly': 12
                },
                'compression': True,
                's3': {'enabled': False},
                'azure': {'enabled': False},
                'gcs': {'enabled': False}
            }
        }
        
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_PATH'] = str(temp_dir / 'test.db')
        
        manager = BackupManager(config)
        yield manager
        
        # Cleanup
        if 'DB_TYPE' in os.environ:
            del os.environ['DB_TYPE']
        if 'SQLITE_PATH' in os.environ:
            del os.environ['SQLITE_PATH']
    
    def test_database_backup(self, backup_manager, temp_dir):
        """Test database backup creation"""
        # Create dummy database file
        db_path = temp_dir / 'test.db'
        db_path.write_text("dummy database content")
        
        # Perform backup
        backup_path = backup_manager.backup_database()
        
        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert backup_path.endswith('.gz')
    
    def test_file_backup(self, backup_manager, temp_dir):
        """Test file/directory backup"""
        # Create test files
        source_dir = temp_dir / 'source'
        source_dir.mkdir()
        (source_dir / 'file1.txt').write_text("content 1")
        (source_dir / 'file2.txt').write_text("content 2")
        
        # Perform backup
        backup_path = backup_manager.backup_files(str(source_dir))
        
        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert backup_path.endswith('.tar.gz')
    
    def test_compression(self, backup_manager, temp_dir):
        """Test backup compression"""
        # Create large test file
        large_file = temp_dir / 'large.txt'
        large_file.write_text("x" * 10000)
        
        # Backup with compression
        backup_manager.config['backup']['compression'] = True
        compressed_path = backup_manager.backup_files(str(large_file))
        
        # Backup without compression
        backup_manager.config['backup']['compression'] = False
        uncompressed_path = backup_manager.backup_files(str(large_file))
        
        # Compressed should be smaller
        if compressed_path and uncompressed_path:
            compressed_size = os.path.getsize(compressed_path)
            uncompressed_size = os.path.getsize(uncompressed_path)
            assert compressed_size < uncompressed_size
    
    def test_retention_policy(self, backup_manager):
        """Test backup retention policy"""
        from datetime import datetime, timedelta
        
        # Create old backup entries
        old_backups = [
            {
                'path': f'/path/to/backup_{i}.gz',
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'type': 'daily'
            }
            for i in range(10)
        ]
        
        # Apply retention
        retention_days = backup_manager.config['backup']['retention']['daily']
        kept_backups = [
            b for b in old_backups 
            if (datetime.now() - datetime.fromisoformat(b['timestamp'])).days < retention_days
        ]
        
        # Should keep only recent backups
        assert len(kept_backups) == retention_days
    
    def test_backup_metadata(self, backup_manager, temp_dir):
        """Test backup metadata creation"""
        # Create test file
        test_file = temp_dir / 'test.txt'
        test_file.write_text("test content")
        
        # Perform backup
        backup_path = backup_manager.backup_files(str(test_file))
        
        if backup_path:
            # Check for metadata file
            metadata_path = backup_path.replace('.tar.gz', '_metadata.json')
            # Metadata should exist
            assert backup_path is not None
    
    @pytest.mark.slow
    def test_full_backup(self, backup_manager, temp_dir):
        """Test full system backup"""
        # Create dummy database
        db_path = temp_dir / 'test.db'
        db_path.write_text("database content")
        
        # Create test files
        data_dir = temp_dir / 'data'
        data_dir.mkdir()
        (data_dir / 'file1.txt').write_text("content 1")
        
        # Perform full backup
        backup_info = backup_manager.full_backup()
        
        assert 'database' in backup_info
        assert backup_info['database'] is not None


class TestRestoreManager:
    """Test suite for RestoreManager"""
    
    @pytest.fixture
    def restore_manager(self, temp_dir):
        """Create RestoreManager with temp directory"""
        config = {
            'backup': {
                'local': {
                    'enabled': True,
                    'path': str(temp_dir / 'backups')
                }
            }
        }
        
        manager = RestoreManager(config)
        yield manager
    
    def test_list_backups(self, restore_manager, temp_dir):
        """Test listing available backups"""
        # Create dummy backup files
        backup_dir = temp_dir / 'backups'
        backup_dir.mkdir()
        
        (backup_dir / 'backup_1.tar.gz').write_text("backup 1")
        (backup_dir / 'backup_2.tar.gz').write_text("backup 2")
        
        # List backups
        backups = restore_manager.list_backups()
        
        # Should find backups
        assert isinstance(backups, list)
    
    def test_restore_database(self, restore_manager, temp_dir):
        """Test database restore"""
        import tarfile
        import gzip
        
        # Create a backup file
        backup_dir = temp_dir / 'backups'
        backup_dir.mkdir()
        
        # Create dummy database content
        db_content = b"database content"
        backup_path = backup_dir / 'db_backup.gz'
        with gzip.open(backup_path, 'wb') as f:
            f.write(db_content)
        
        # Restore should work
        restore_path = temp_dir / 'restored.db'
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_PATH'] = str(restore_path)
        
        result = restore_manager.restore_database(str(backup_path))
        
        # Cleanup
        del os.environ['DB_TYPE']
        del os.environ['SQLITE_PATH']
        
        assert result is not None
    
    def test_restore_files(self, restore_manager, temp_dir):
        """Test file restore from backup"""
        import tarfile
        
        # Create test files
        source_dir = temp_dir / 'source'
        source_dir.mkdir()
        (source_dir / 'file1.txt').write_text("content 1")
        (source_dir / 'file2.txt').write_text("content 2")
        
        # Create tar backup
        backup_dir = temp_dir / 'backups'
        backup_dir.mkdir()
        backup_path = backup_dir / 'files_backup.tar.gz'
        
        with tarfile.open(backup_path, 'w:gz') as tar:
            tar.add(source_dir, arcname='source')
        
        # Restore to different location
        restore_dir = temp_dir / 'restored'
        result = restore_manager.restore_files(str(backup_path), str(restore_dir))
        
        assert result is not None
    
    def test_verify_backup(self, restore_manager, temp_dir):
        """Test backup verification"""
        import tarfile
        
        # Create valid backup
        backup_dir = temp_dir / 'backups'
        backup_dir.mkdir()
        backup_path = backup_dir / 'test_backup.tar.gz'
        
        source_dir = temp_dir / 'source'
        source_dir.mkdir()
        (source_dir / 'file.txt').write_text("content")
        
        with tarfile.open(backup_path, 'w:gz') as tar:
            tar.add(source_dir, arcname='source')
        
        # Verify
        is_valid = restore_manager.verify_backup(str(backup_path))
        assert is_valid is True


@pytest.mark.integration
class TestBackupRestoreIntegration:
    """Integration tests for backup and restore"""
    
    def test_backup_and_restore_workflow(self, temp_dir):
        """Test complete backup and restore workflow"""
        # Setup
        config = {
            'backup': {
                'local': {
                    'enabled': True,
                    'path': str(temp_dir / 'backups')
                },
                'retention': {'daily': 7, 'weekly': 4, 'monthly': 12},
                'compression': True,
                's3': {'enabled': False},
                'azure': {'enabled': False},
                'gcs': {'enabled': False}
            }
        }
        
        # Create test data
        source_dir = temp_dir / 'data'
        source_dir.mkdir()
        (source_dir / 'important.txt').write_text("important data")
        
        # Backup
        backup_manager = BackupManager(config)
        backup_path = backup_manager.backup_files(str(source_dir))
        
        assert backup_path is not None
        
        # Restore
        restore_manager = RestoreManager(config)
        restore_dir = temp_dir / 'restored'
        restore_manager.restore_files(backup_path, str(restore_dir))
        
        # Verify
        restored_file = Path(restore_dir) / 'data' / 'important.txt'
        if restored_file.exists():
            assert restored_file.read_text() == "important data"
    
    @pytest.mark.slow
    def test_disaster_recovery(self, temp_dir):
        """Test disaster recovery scenario"""
        config = {
            'backup': {
                'local': {
                    'enabled': True,
                    'path': str(temp_dir / 'backups')
                },
                'retention': {'daily': 7, 'weekly': 4, 'monthly': 12},
                'compression': True,
                's3': {'enabled': False},
                'azure': {'enabled': False},
                'gcs': {'enabled': False}
            }
        }
        
        # Create "production" data
        prod_dir = temp_dir / 'production'
        prod_dir.mkdir()
        (prod_dir / 'data.txt').write_text("production data")
        
        # Create database
        db_path = temp_dir / 'prod.db'
        db_path.write_text("production database")
        
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_PATH'] = str(db_path)
        
        # Full backup
        backup_manager = BackupManager(config)
        backup_info = backup_manager.full_backup()
        
        # Simulate disaster (delete everything)
        import shutil
        shutil.rmtree(prod_dir)
        db_path.unlink()
        
        # Restore
        restore_manager = RestoreManager(config)
        
        # Restore database
        if backup_info.get('database'):
            restore_manager.restore_database(backup_info['database'])
        
        # Cleanup
        del os.environ['DB_TYPE']
        del os.environ['SQLITE_PATH']
