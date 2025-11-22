"""
Test configuration and fixtures for pytest
"""
import os
import sys
from pathlib import Path
import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock
import factory
from faker import Faker

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize Faker
fake = Faker()


# ============================================================================
# Directory and File Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file"""
    file_path = temp_dir / f"temp_{fake.uuid4()}.txt"
    file_path.touch()
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture(scope="session")
def sample_fasta_file(test_data_dir):
    """Path to sample FASTA file"""
    fasta_path = test_data_dir / "sample.fasta"
    if not fasta_path.exists():
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        fasta_path.write_text(""">seq1 Test sequence 1
ATCGATCGATCGATCGATCG
>seq2 Test sequence 2
GCTAGCTAGCTAGCTAGCTA
>seq3 Test sequence 3
TTAACCGGTTAACCGGTTAA
""")
    return fasta_path


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_database():
    """Mock database connection"""
    # Set test database environment
    os.environ['DB_TYPE'] = 'sqlite'
    os.environ['SQLITE_PATH'] = ':memory:'
    yield
    # Cleanup
    if 'DB_TYPE' in os.environ:
        del os.environ['DB_TYPE']
    if 'SQLITE_PATH' in os.environ:
        del os.environ['SQLITE_PATH']


@pytest.fixture
def db_session(mock_database):
    """Create a database session for testing"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    
    engine = create_engine('sqlite:///:memory:', echo=False)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def mock_postgresql_connection():
    """Mock PostgreSQL connection"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    return mock_conn


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_config():
    """Mock configuration"""
    config = {
        'storage': {
            'datasets_dir': 'data/datasets',
            'runs_dir': 'runs',
            'backups_dir': 'data/backups'
        },
        'security': {
            'max_file_size_mb': 500,
            'allowed_extensions': ['.fasta', '.fastq', '.gz']
        },
        'backup': {
            'retention': {
                'daily': 7,
                'weekly': 4,
                'monthly': 12
            },
            'compression': True
        },
        'analysis': {
            'max_sequences': 100000,
            'batch_size': 1000
        }
    }
    return config


@pytest.fixture
def mock_environment(monkeypatch):
    """Mock environment variables"""
    test_env = {
        'DATABASE_URL': 'sqlite:///:memory:',
        'REDIS_URL': 'redis://localhost:6379/0',
        'SECRET_KEY': 'test-secret-key',
        'ENVIRONMENT': 'test'
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env


# ============================================================================
# Sample Data Fixtures
# ============================================================================
    """Sample DNA sequences for testing"""
    return [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "TTAACCGGTTAA",
        "AAAATTTCCCGG"
    ]


@pytest.fixture
def sample_fasta_content():
    """Sample FASTA format content"""
    return """>seq1 Sample sequence 1
ATCGATCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGATCG
>seq2 Sample sequence 2
GCTAGCTAGCTAGCTAGCTAGCTA
GCTAGCTAGCTAGCTAGCTAGCTA
>seq3 Sample sequence 3
TTAACCGGTTAACCGGTTAACCGG
TTAACCGGTTAACCGGTTAACCGG
"""


# ============================================================================
# Celery Fixtures
# ============================================================================

@pytest.fixture
def mock_celery_app():
    """Mock Celery application"""
    from celery import Celery
    app = Celery('test_app', broker='memory://', backend='cache+memory://')
    app.conf.task_always_eager = True  # Execute tasks synchronously
    app.conf.task_eager_propagates = True  # Propagate exceptions
    return app


@pytest.fixture
def mock_celery_task(mock_celery_app):
    """Mock Celery task"""
    task = MagicMock()
    task.request.id = fake.uuid4()
    task.request.retries = 0
    task.update_state = MagicMock()
    return task


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    return redis_mock


# ============================================================================
# API Fixtures
# ============================================================================

@pytest.fixture
def api_client():
    """Create a test API client"""
    from flask import Flask
    from flask.testing import FlaskClient
    
    app = Flask('test_app')
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_request():
    """Mock Flask request object"""
    request_mock = MagicMock()
    request_mock.method = 'GET'
    request_mock.path = '/test'
    request_mock.args = {}
    request_mock.json = {}
    request_mock.headers = {}
    return request_mock


# ============================================================================
# Model Factory Fixtures
# ============================================================================

@pytest.fixture
def user_factory():
    """Factory for creating test users"""
    def _create_user(**kwargs):
        defaults = {
            'id': fake.uuid4(),
            'username': fake.user_name(),
            'email': fake.email(),
            'created_at': datetime.utcnow(),
            'is_active': True
        }
        defaults.update(kwargs)
        return type('User', (), defaults)
    return _create_user


@pytest.fixture
def dataset_factory():
    """Factory for creating test datasets"""
    def _create_dataset(**kwargs):
        defaults = {
            'id': fake.uuid4(),
            'name': fake.word(),
            'description': fake.sentence(),
            'file_path': f'/data/{fake.uuid4()}.fasta',
            'size_bytes': fake.random_int(min=1000, max=1000000),
            'sequence_count': fake.random_int(min=10, max=10000),
            'created_at': datetime.utcnow(),
            'status': 'uploaded'
        }
        defaults.update(kwargs)
        return type('Dataset', (), defaults)
    return _create_dataset


@pytest.fixture
def analysis_run_factory():
    """Factory for creating test analysis runs"""
    def _create_analysis_run(**kwargs):
        defaults = {
            'id': fake.uuid4(),
            'dataset_id': fake.uuid4(),
            'analysis_type': 'taxonomic',
            'status': 'completed',
            'started_at': datetime.utcnow() - timedelta(hours=1),
            'completed_at': datetime.utcnow(),
            'sequences_processed': fake.random_int(min=100, max=10000),
            'results_path': f'/results/{fake.uuid4()}.json'
        }
        defaults.update(kwargs)
        return type('AnalysisRun', (), defaults)
    return _create_analysis_run


# ============================================================================
# Time Fixtures
# ============================================================================

@pytest.fixture
def freeze_time():
    """Freeze time for testing"""
    from freezegun import freeze_time as _freeze_time
    frozen_time = datetime(2025, 11, 22, 12, 0, 0)
    with _freeze_time(frozen_time):
        yield frozen_time


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests requiring database"
    )
    config.addinivalue_line(
        "markers", "celery: marks tests for Celery tasks"
    )
    config.addinivalue_line(
        "markers", "api: marks API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance tests"
    )
