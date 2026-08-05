"""
Pytest configuration — applied automatically to every test in the suite.

Sets EDNA_ENV=testing before any test module is imported, which causes
DatabaseConfig.get_config() to select the SQLite testing backend and
Config._resolve_env_overrides() to pick up test-safe paths.
"""

import os
import pytest
from pathlib import Path


def pytest_configure(config):
    """Force testing environment before any test is collected."""
    os.environ.setdefault("EDNA_ENV", "testing")
    os.environ.setdefault("LOG_DIR", "logs/test")
    os.environ.setdefault("DATA_RAW_DIR", "data/raw")
    os.environ.setdefault("DATA_PROCESSED_DIR", "data/processed")
    os.environ.setdefault("DATA_OUTPUT_DIR", "data/output")


@pytest.fixture(scope="session", autouse=True)
def ensure_dirs():
    """Create directories that pipeline code expects to exist."""
    dirs = [
        Path("logs/test"),
        Path("data/raw"),
        Path("data/processed"),
        Path("data/output"),
        Path("data/reference"),
        Path("models/trained"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
