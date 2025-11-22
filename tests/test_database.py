"""
Database module tests
"""
import pytest
import os
from pathlib import Path
import tempfile

from src.database.database import DatabaseManager


class TestDatabaseManager:
    """Test suite for DatabaseManager"""
    
    @pytest.fixture
    def sqlite_db(self):
        """Create temporary SQLite database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        os.environ['DB_TYPE'] = 'sqlite'
        os.environ['SQLITE_PATH'] = db_path
        
        db = DatabaseManager()
        yield db
        
        # Cleanup
        db.close()
        if os.path.exists(db_path):
            os.unlink(db_path)
        if 'DB_TYPE' in os.environ:
            del os.environ['DB_TYPE']
        if 'SQLITE_PATH' in os.environ:
            del os.environ['SQLITE_PATH']
    
    def test_database_initialization(self, sqlite_db):
        """Test database initialization"""
        assert sqlite_db is not None
        assert sqlite_db.engine is not None
        assert sqlite_db.Session is not None
    
    def test_session_management(self, sqlite_db):
        """Test database session creation and cleanup"""
        session = sqlite_db.get_session()
        assert session is not None
        
        # Session should be usable
        result = session.execute("SELECT 1")
        assert result.fetchone()[0] == 1
        
        session.close()
    
    def test_context_manager(self, sqlite_db):
        """Test database context manager"""
        with sqlite_db.get_session() as session:
            result = session.execute("SELECT 1")
            assert result.fetchone()[0] == 1
        
        # Session should be closed after context
        assert session.is_active is False
    
    @pytest.mark.integration
    def test_transaction_rollback(self, sqlite_db):
        """Test transaction rollback on error"""
        from src.database.models import User
        
        try:
            with sqlite_db.get_session() as session:
                # Create user
                user = User(
                    username="test_user",
                    email="test@example.com",
                    password_hash="hashed_password"
                )
                session.add(user)
                session.flush()
                
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # User should not exist due to rollback
        with sqlite_db.get_session() as session:
            user = session.query(User).filter_by(username="test_user").first()
            assert user is None
    
    def test_connection_pooling(self, sqlite_db):
        """Test connection pool management"""
        # Create multiple sessions
        sessions = [sqlite_db.get_session() for _ in range(5)]
        
        # All should be valid
        for session in sessions:
            result = session.execute("SELECT 1")
            assert result.fetchone()[0] == 1
        
        # Cleanup
        for session in sessions:
            session.close()


@pytest.mark.integration
class TestDatabaseMigration:
    """Test suite for database migrations"""
    
    def test_schema_creation(self, sqlite_db):
        """Test database schema creation"""
        from src.database.models import Base
        
        # Create all tables
        Base.metadata.create_all(sqlite_db.engine)
        
        # Verify tables exist
        inspector = sqlite_db.engine.dialect.get_inspector(sqlite_db.engine)
        tables = inspector.get_table_names()
        
        assert 'users' in tables
        assert 'datasets' in tables
        assert 'analysis_runs' in tables
    
    def test_migration_script(self):
        """Test migration script execution"""
        # This would test the actual migration scripts
        # Placeholder for migration testing
        pass


@pytest.mark.integration
class TestDatabaseModels:
    """Test suite for database models"""
    
    def test_user_model(self, sqlite_db):
        """Test User model CRUD operations"""
        from src.database.models import Base, User
        
        # Create schema
        Base.metadata.create_all(sqlite_db.engine)
        
        with sqlite_db.get_session() as session:
            # Create user
            user = User(
                username="testuser",
                email="test@example.com",
                password_hash="hashed_pw"
            )
            session.add(user)
            session.commit()
            
            # Retrieve user
            retrieved = session.query(User).filter_by(username="testuser").first()
            assert retrieved is not None
            assert retrieved.email == "test@example.com"
            
            # Update user
            retrieved.email = "newemail@example.com"
            session.commit()
            
            # Verify update
            updated = session.query(User).filter_by(username="testuser").first()
            assert updated.email == "newemail@example.com"
            
            # Delete user
            session.delete(updated)
            session.commit()
            
            # Verify deletion
            deleted = session.query(User).filter_by(username="testuser").first()
            assert deleted is None
    
    def test_dataset_model(self, sqlite_db):
        """Test Dataset model"""
        from src.database.models import Base, Dataset
        
        Base.metadata.create_all(sqlite_db.engine)
        
        with sqlite_db.get_session() as session:
            dataset = Dataset(
                name="test_dataset",
                description="Test dataset",
                file_path="/path/to/dataset.fasta",
                file_size=1024
            )
            session.add(dataset)
            session.commit()
            
            retrieved = session.query(Dataset).filter_by(name="test_dataset").first()
            assert retrieved is not None
            assert retrieved.file_size == 1024
    
    def test_analysis_run_model(self, sqlite_db):
        """Test AnalysisRun model"""
        from src.database.models import Base, AnalysisRun
        
        Base.metadata.create_all(sqlite_db.engine)
        
        with sqlite_db.get_session() as session:
            run = AnalysisRun(
                name="test_run",
                dataset_id=1,
                status="running",
                parameters={"param1": "value1"}
            )
            session.add(run)
            session.commit()
            
            retrieved = session.query(AnalysisRun).filter_by(name="test_run").first()
            assert retrieved is not None
            assert retrieved.status == "running"


@pytest.mark.slow
class TestDatabasePerformance:
    """Performance tests for database operations"""
    
    def test_bulk_insert_performance(self, sqlite_db):
        """Test bulk insert performance"""
        from src.database.models import Base, Dataset
        import time
        
        Base.metadata.create_all(sqlite_db.engine)
        
        # Create 1000 records
        datasets = [
            Dataset(
                name=f"dataset_{i}",
                description=f"Test dataset {i}",
                file_path=f"/path/to/dataset_{i}.fasta",
                file_size=1024 * i
            )
            for i in range(1000)
        ]
        
        start = time.time()
        with sqlite_db.get_session() as session:
            session.bulk_save_objects(datasets)
            session.commit()
        duration = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert duration < 5.0
        
        # Verify count
        with sqlite_db.get_session() as session:
            count = session.query(Dataset).count()
            assert count == 1000
    
    def test_query_performance(self, sqlite_db):
        """Test query performance with indices"""
        from src.database.models import Base, Dataset
        import time
        
        Base.metadata.create_all(sqlite_db.engine)
        
        # Create test data
        datasets = [
            Dataset(
                name=f"dataset_{i}",
                description=f"Test dataset {i}",
                file_path=f"/path/to/dataset_{i}.fasta",
                file_size=1024 * i
            )
            for i in range(100)
        ]
        
        with sqlite_db.get_session() as session:
            session.bulk_save_objects(datasets)
            session.commit()
        
        # Query with filter
        start = time.time()
        with sqlite_db.get_session() as session:
            results = session.query(Dataset).filter(
                Dataset.file_size > 50000
            ).all()
        duration = time.time() - start
        
        # Should be fast (< 0.1 seconds)
        assert duration < 0.1
        assert len(results) > 0
