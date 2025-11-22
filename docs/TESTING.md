# Testing Guide

## Overview

The Avalanche eDNA platform uses **pytest** as its testing framework with comprehensive test coverage across unit, integration, and end-to-end tests.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── test_backup.py              # Backup system tests
├── test_database.py            # Database tests
├── test_security.py            # Security tests
├── test_system.py              # System tests
├── test_celery_tasks.py        # Celery task integration tests
├── test_api_integration.py     # API endpoint integration tests
├── test_e2e_workflows.py       # End-to-end workflow tests
└── data/                       # Test data files
```

## Test Categories

### Unit Tests
Test individual functions and methods in isolation.

```python
@pytest.mark.unit
def test_sequence_validation():
    assert is_valid_dna("ATCG") == True
    assert is_valid_dna("XYZ") == False
```

### Integration Tests
Test component interactions and integrations.

```python
@pytest.mark.integration
@pytest.mark.celery
def test_celery_task_execution():
    result = my_task.apply()
    assert result.status == 'SUCCESS'
```

### End-to-End Tests
Test complete workflows from start to finish.

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_complete_analysis_pipeline():
    # Upload → Preprocess → Analyze → Report
    ...
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# End-to-end tests
pytest -m e2e

# Exclude slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Single file
pytest tests/test_database.py

# Multiple files
pytest tests/test_api_integration.py tests/test_celery_tasks.py

# Specific test function
pytest tests/test_database.py::TestDatabaseBackup::test_backup_success
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Parallel Execution

```bash
# Run tests in parallel (4 workers)
pytest -n 4

# Auto-detect CPU count
pytest -n auto
```

### Verbose Output

```bash
# Show detailed test output
pytest -v

# Show extra summary info
pytest -ra

# Show local variables on failure
pytest -l
```

## Test Configuration

### pytest.ini

Configuration file defining test discovery, markers, and coverage settings.

```ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow-running tests
    security: Security tests
```

### Coverage Settings

Target: **80% minimum coverage**

```bash
# Check coverage
pytest --cov=src --cov-fail-under=80

# Generate detailed report
pytest --cov=src --cov-report=term-missing
```

## Fixtures

### Common Fixtures (conftest.py)

#### Directory Fixtures
- `test_data_dir`: Path to test data directory
- `temp_dir`: Temporary directory for test files
- `temp_file`: Temporary file

#### Database Fixtures
- `mock_database`: Mock database connection
- `db_session`: SQLAlchemy session
- `mock_postgresql_connection`: PostgreSQL mock

#### Configuration Fixtures
- `mock_config`: Mock configuration dict
- `mock_environment`: Mock environment variables

#### Celery Fixtures
- `mock_celery_app`: Celery application
- `mock_celery_task`: Celery task mock
- `mock_redis`: Redis client mock

#### API Fixtures
- `api_client`: Flask test client
- `mock_request`: Mock Flask request

#### Factory Fixtures
- `user_factory`: Create test users
- `dataset_factory`: Create test datasets
- `analysis_run_factory`: Create test analysis runs

### Using Fixtures

```python
def test_with_fixtures(temp_dir, dataset_factory, api_client):
    # Create test dataset
    dataset = dataset_factory(name='test_dataset')
    
    # Make API request
    response = api_client.get(f'/api/v1/datasets/{dataset.id}')
    
    assert response.status_code == 200
```

## Mocking

### Patching Functions

```python
from unittest.mock import patch

def test_with_mock():
    with patch('src.module.function') as mock_func:
        mock_func.return_value = 'mocked_value'
        
        result = my_function()
        
        assert result == 'mocked_value'
        mock_func.assert_called_once()
```

### Mocking Celery Tasks

```python
@pytest.mark.celery
def test_celery_task(mock_celery_app):
    from src.tasks.analysis_tasks import run_analysis
    
    with patch('src.tasks.analysis_tasks.perform_analysis') as mock_perform:
        mock_perform.return_value = {'status': 'success'}
        
        result = run_analysis('dataset_id', {})
        
        assert result['status'] == 'success'
```

### Mocking API Responses

```python
def test_api_endpoint(api_client):
    with patch('src.api.datasets.get_dataset_by_id') as mock_get:
        mock_get.return_value = {'id': '123', 'name': 'test'}
        
        response = api_client.get('/api/v1/datasets/123')
        
        assert response.status_code == 200
```

## Writing Tests

### Test Structure

```python
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestMyFeature:
    """Tests for my feature"""
    
    def test_success_case(self, fixture1, fixture2):
        """Test successful execution"""
        # Arrange
        input_data = 'test'
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_error_case(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            my_function(invalid_input)
    
    def test_edge_case(self):
        """Test edge case"""
        result = my_function(edge_case_input)
        assert result is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("ATCG", True),
    ("GCTA", True),
    ("XYZW", False),
    ("", False),
])
def test_sequence_validation(input, expected):
    assert is_valid_dna(input) == expected
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

## Test Best Practices

### 1. Follow AAA Pattern
- **Arrange**: Set up test data and mocks
- **Act**: Execute the function being tested
- **Assert**: Verify the results

### 2. One Assert Per Test (Generally)
Focus each test on a single assertion when possible.

```python
def test_create_user():
    user = create_user('test@example.com')
    assert user.email == 'test@example.com'

def test_user_is_active():
    user = create_user('test@example.com')
    assert user.is_active == True
```

### 3. Use Descriptive Test Names
Test names should describe what they test.

```python
# Good
def test_download_task_retries_on_network_error():
    ...

# Bad
def test_download():
    ...
```

### 4. Test Edge Cases
```python
def test_empty_input():
    assert my_function([]) == []

def test_none_input():
    with pytest.raises(TypeError):
        my_function(None)

def test_large_input():
    large_data = [i for i in range(10000)]
    result = my_function(large_data)
    assert len(result) == 10000
```

### 5. Keep Tests Independent
Tests should not depend on each other.

```python
# Bad - tests depend on order
def test_create():
    global user
    user = create_user()

def test_update():
    user.update(name='New Name')

# Good - independent tests
def test_create(user_factory):
    user = user_factory()
    assert user.id is not None

def test_update(user_factory):
    user = user_factory()
    user.update(name='New Name')
    assert user.name == 'New Name'
```

### 6. Mock External Dependencies
Always mock external services, databases, APIs.

```python
def test_api_call():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'data': 'test'}
        
        result = fetch_data()
        
        assert result == {'data': 'test'}
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Every push to main branches
- Pull requests
- Scheduled nightly builds

### Pre-commit Hooks

Tests run before commits:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Debugging Tests

### Run with Debugger

```bash
# Drop into pdb on failure
pytest --pdb

# Drop into pdb at start of test
pytest --trace
```

### Print Debug Output

```python
def test_with_debug():
    result = my_function()
    print(f"Result: {result}")  # Will show in output with -s
    assert result == expected

# Run with output
pytest -s tests/test_file.py
```

### Capture Logs

```bash
# Show log output
pytest --log-cli-level=DEBUG

# Capture to file
pytest --log-file=test.log
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
def test_performance(benchmark):
    result = benchmark(my_function, arg1, arg2)
    
    # Assert performance threshold
    assert benchmark.stats.mean < 0.1  # < 100ms
```

### Load Testing

```python
@pytest.mark.performance
def test_concurrent_requests(api_client):
    from concurrent.futures import ThreadPoolExecutor
    
    def make_request():
        return api_client.get('/api/v1/datasets')
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(100)]
        results = [f.result() for f in futures]
    
    assert all(r.status_code == 200 for r in results)
```

## Troubleshooting

### Tests Fail Locally But Pass in CI
- Check Python versions match
- Verify dependencies are the same
- Check for environment-specific code

### Slow Test Execution
```bash
# Find slowest tests
pytest --durations=10

# Run in parallel
pytest -n auto
```

### Flaky Tests
- Add retries for network-dependent tests
- Use freezegun for time-dependent tests
- Increase timeouts for slow operations

### Import Errors
```bash
# Install test dependencies
pip install -r requirements.txt

# Add project to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Coverage Goals

### Current Coverage
Run coverage to see current state:

```bash
pytest --cov=src --cov-report=term-missing
```

### Minimum Requirements
- **Overall**: 80%+
- **Critical modules**: 90%+
  - `src/database/`
  - `src/security/`
  - `src/backup/`

### Improving Coverage
1. Run coverage report
2. Identify uncovered lines
3. Add tests for uncovered code
4. Focus on critical paths first

```bash
# Generate HTML report with highlighting
pytest --cov=src --cov-report=html

# Open report
open htmlcov/index.html
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Factory Boy](https://factoryboy.readthedocs.io/)
