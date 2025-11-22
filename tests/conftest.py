"""
Test configuration and fixtures for pytest
"""
import os
import sys
from pathlib import Path
import pytest
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_fasta_file(test_data_dir):
    """Path to sample FASTA file"""
    return test_data_dir / "sample.fasta"


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
        }
    }
    return config


@pytest.fixture
def sample_sequences():
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
