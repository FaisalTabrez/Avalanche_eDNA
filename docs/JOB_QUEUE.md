# Job Queue System Documentation

## Overview

The Avalanche eDNA platform uses **Celery** with **Redis** as a distributed task queue system to handle long-running operations asynchronously. This enables the application to remain responsive while processing computationally intensive tasks in the background.

## Architecture

```
┌─────────────────┐
│  Streamlit App  │ ──┐
└─────────────────┘   │
                      │  Submit tasks
┌─────────────────┐   │
│   Flask API     │ ──┤
└─────────────────┘   │
                      ▼
                ┌───────────┐
                │   Redis   │ (Message Broker)
                └───────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Worker  │ │ Worker  │ │ Worker  │
    │ (queue: │ │ (queue: │ │ (queue: │
    │analysis)│ │training)│ │download)│
    └─────────┘ └─────────┘ └─────────┘
          │           │           │
          └───────────┴───────────┘
                      │
                      ▼
                ┌───────────┐
                │ PostgreSQL│ (Results Backend)
                └───────────┘
                      │
                      ▼
                ┌───────────┐
                │  Flower   │ (Monitoring)
                └───────────┘
```

## Components

### 1. Celery Application (`src/tasks/celery_app.py`)

Central configuration for the task queue system:

- **Broker**: Redis (`redis://redis:6379/0`)
- **Backend**: Redis (for storing task results)
- **Task Queues**:
  - `default`: General tasks
  - `analysis`: eDNA analysis workflows
  - `training`: Model training tasks
  - `downloads`: Data download tasks
  - `maintenance`: System maintenance tasks

### 2. Task Modules

#### Analysis Tasks (`src/tasks/analysis_tasks.py`)

- `run_analysis`: Run eDNA taxonomic/novelty/clustering analysis
- `run_blast_search`: BLAST search against reference databases
- `run_multiple_analyses`: Batch analysis processing

**Example Usage**:
```python
from src.tasks.analysis_tasks import run_analysis

# Submit analysis task
result = run_analysis.delay(
    dataset_path='data/raw/sample.fasta',
    analysis_type='taxonomic',
    parameters={'min_confidence': 0.8}
)

# Check status
print(f"Task ID: {result.id}")
print(f"Status: {result.status}")

# Get result (blocking)
output = result.get(timeout=3600)
print(output)
```

#### Training Tasks (`src/tasks/training_tasks.py`)

- `train_model`: Train ML models (transformer, CNN, LSTM)
- `evaluate_model`: Evaluate trained models
- `hyperparameter_tuning`: Grid search optimization

**Example Usage**:
```python
from src.tasks.training_tasks import train_model

result = train_model.delay(
    training_data_path='data/processed/training.csv',
    model_type='transformer',
    hyperparameters={
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
)
```

#### Download Tasks (`src/tasks/download_tasks.py`)

- `download_sra_dataset`: Download from NCBI SRA
- `download_batch_sra`: Batch SRA downloads
- `download_reference_database`: Download SILVA, PR2, etc.
- `update_reference_databases`: Update all databases

**Example Usage**:
```python
from src.tasks.download_tasks import download_sra_dataset

result = download_sra_dataset.delay(
    accession='SRR1234567',
    output_dir='data/raw/sra'
)
```

#### Maintenance Tasks (`src/tasks/maintenance_tasks.py`)

Scheduled periodic tasks:

- `cleanup_old_results`: Remove results older than 30 days (daily at 2 AM)
- `cleanup_temp_files`: Clean temporary files (every 6 hours)
- `backup_database`: Database backup (daily at 3 AM)
- `monitor_system_health`: Health checks (every 15 minutes)
- `cleanup_failed_tasks`: Remove failed task records (weekly)
- `optimize_database`: VACUUM and ANALYZE (weekly)

### 3. Celery Workers

Workers process tasks from specific queues:

```bash
# Start worker for analysis queue
celery -A src.tasks.celery_app worker -Q analysis --loglevel=info

# Start worker for all queues
celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4
```

**Configuration**:
- **Concurrency**: 4 (number of parallel tasks)
- **Prefetch Multiplier**: 1 (fair task distribution)
- **Max Tasks Per Child**: 100 (prevent memory leaks)
- **Time Limits**: 
  - Hard: 3600s (1 hour)
  - Soft: 3300s (55 minutes)

### 4. Celery Beat (Scheduler)

Runs periodic tasks on schedule:

```bash
# Start beat scheduler
celery -A src.tasks.celery_app beat --loglevel=info
```

**Schedule** (configured in `celery_app.py`):
- `cleanup-old-results`: Daily at 2:00 AM
- `cleanup-temp-files`: Every 6 hours
- `backup-database`: Daily at 3:00 AM
- `monitor-system-health`: Every 15 minutes

### 5. Flower (Monitoring Dashboard)

Web-based monitoring tool for Celery:

```bash
# Start Flower
celery -A src.tasks.celery_app flower --port=5555
```

**Access**: http://localhost:5555

**Features**:
- Real-time task monitoring
- Worker status and statistics
- Task history and results
- Task rate limiting
- Task routing visualization

**Authentication**: 
- Username: `admin`
- Password: `admin` (change in production!)

## Docker Deployment

### Services in docker-compose.yml

```yaml
services:
  # Redis broker
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  # Celery worker
  celery-worker:
    build: .
    command: celery -A src.tasks.celery_app worker --loglevel=info --concurrency=4
    depends_on:
      - redis
      - postgres
  
  # Celery beat scheduler
  celery-beat:
    build: .
    command: celery -A src.tasks.celery_app beat --loglevel=info
    depends_on:
      - redis
      - postgres
  
  # Flower monitoring
  flower:
    build: .
    command: celery -A src.tasks.celery_app flower --port=5555
    ports:
      - "5555:5555"
    profiles:
      - dev-tools
```

### Start All Services

```bash
# Start all services
docker-compose up -d

# Start with Flower monitoring
docker-compose --profile dev-tools up -d

# View worker logs
docker-compose logs -f celery-worker

# View beat scheduler logs
docker-compose logs -f celery-beat
```

## Development Usage

### 1. Install Dependencies

```bash
# Install Celery and Redis
pip install celery redis flower

# Or use requirements.txt
pip install -r requirements.txt
```

### 2. Start Redis Locally

```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or install Redis locally (Ubuntu/Debian)
sudo apt install redis-server
sudo systemctl start redis
```

### 3. Start Celery Worker

```bash
# From project root
export PYTHONPATH=$PWD
export REDIS_URL=redis://localhost:6379/0
export DB_TYPE=sqlite
export SQLITE_PATH=data/database.db

# Start worker
celery -A src.tasks.celery_app worker --loglevel=info
```

### 4. Start Celery Beat (Optional)

```bash
# In separate terminal
celery -A src.tasks.celery_app beat --loglevel=info
```

### 5. Start Flower (Optional)

```bash
# In separate terminal
celery -A src.tasks.celery_app flower --port=5555
```

### 6. Submit Tasks

```python
from src.tasks.analysis_tasks import run_analysis

# Submit task
result = run_analysis.delay(
    dataset_path='data/sample.fasta',
    analysis_type='taxonomic'
)

# Get task ID
print(f"Task submitted: {result.id}")

# Check status
print(f"Status: {result.status}")

# Wait for result
try:
    output = result.get(timeout=300)
    print(f"Result: {output}")
except Exception as e:
    print(f"Task failed: {e}")
```

## Task Progress Tracking

All tasks inherit from `ProgressTask` base class with progress tracking:

```python
@shared_task(bind=True)
def my_task(self, data):
    total = len(data)
    
    for i, item in enumerate(data):
        # Process item
        process(item)
        
        # Update progress
        self.update_progress(
            current=i+1,
            total=total,
            status='Processing items'
        )
    
    return {'status': 'success'}
```

**Retrieve Progress**:

```python
result = my_task.delay(data)

# Get progress
info = result.info
if result.state == 'PROGRESS':
    print(f"Progress: {info['current']}/{info['total']} ({info['percent']}%)")
```

## Task States

Celery tracks task states:

- `PENDING`: Task waiting to be executed
- `STARTED`: Task has been started
- `RETRY`: Task is being retried
- `FAILURE`: Task failed with exception
- `SUCCESS`: Task completed successfully
- `REVOKED`: Task was revoked/cancelled

**Check State**:

```python
result = run_analysis.delay(...)

print(result.state)  # PENDING, STARTED, SUCCESS, etc.
print(result.ready())  # True if completed (success or failure)
print(result.successful())  # True if completed successfully
print(result.failed())  # True if failed
```

## Task Chaining and Groups

### Chain: Execute tasks sequentially

```python
from celery import chain

# Download → Analyze → Cleanup
workflow = chain(
    download_sra_dataset.s('SRR1234567'),
    run_analysis.s(analysis_type='taxonomic'),
    cleanup_temp_files.s()
)

result = workflow.apply_async()
```

### Group: Execute tasks in parallel

```python
from celery import group

# Analyze multiple datasets in parallel
job = group(
    run_analysis.s(f'dataset_{i}.fasta', 'taxonomic')
    for i in range(10)
)

result = job.apply_async()
results = result.get()  # Wait for all
```

### Chord: Parallel execution with callback

```python
from celery import chord

# Download multiple datasets, then run combined analysis
workflow = chord(
    [download_sra_dataset.s(acc) for acc in accessions]
)(run_combined_analysis.s())

result = workflow.apply_async()
```

## Error Handling

### Automatic Retry

```python
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def flaky_task(self):
    try:
        # Task logic
        result = do_something()
        return result
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
```

### Error Callbacks

```python
@shared_task
def on_error(request, exc, traceback):
    """Called when task fails"""
    logger.error(f"Task {request.id} failed: {exc}")
    # Send alert, update database, etc.

# Link error callback
result = run_analysis.apply_async(
    args=(...),
    link_error=on_error.s()
)
```

## Monitoring and Debugging

### 1. Flower Dashboard

Access http://localhost:5555 to:
- View active workers
- Monitor task queues
- Inspect task details
- View task history
- See task rates and statistics

### 2. Redis CLI

```bash
# Connect to Redis
docker exec -it avalanche-redis redis-cli

# Monitor commands
MONITOR

# Check queue lengths
LLEN celery
LLEN analysis
LLEN training

# View task IDs in queue
LRANGE analysis 0 -1
```

### 3. Task Logs

```bash
# Worker logs
docker-compose logs -f celery-worker

# Beat scheduler logs
docker-compose logs -f celery-beat

# All Celery logs
docker-compose logs -f celery-worker celery-beat
```

### 4. Programmatic Monitoring

```python
from src.tasks.celery_app import celery_app

# Get active workers
workers = celery_app.control.inspect().active()
print(f"Active workers: {workers}")

# Get stats
stats = celery_app.control.inspect().stats()
print(f"Worker stats: {stats}")

# Get scheduled tasks
scheduled = celery_app.control.inspect().scheduled()
print(f"Scheduled: {scheduled}")
```

## Performance Tuning

### 1. Worker Concurrency

```bash
# More workers for CPU-bound tasks
celery -A src.tasks.celery_app worker --concurrency=8

# Use eventlet for I/O-bound tasks
celery -A src.tasks.celery_app worker --pool=eventlet --concurrency=100
```

### 2. Task Routing

Route tasks to specialized workers:

```bash
# Start analysis-only worker
celery -A src.tasks.celery_app worker -Q analysis -n analysis@%h

# Start download-only worker
celery -A src.tasks.celery_app worker -Q downloads -n downloads@%h
```

### 3. Result Backend

Consider using database for large results:

```python
celery_app.conf.update(
    result_backend='db+postgresql://user:pass@localhost/avalanche',
    result_persistent=True,
)
```

### 4. Task Compression

Enable compression for large task payloads:

```python
celery_app.conf.update(
    task_compression='gzip',
    result_compression='gzip',
)
```

## Troubleshooting

### Issue: Tasks not being processed

**Check**:
1. Is Redis running? `docker-compose ps redis`
2. Are workers running? `docker-compose ps celery-worker`
3. Are tasks in the queue? Check Flower dashboard
4. Check worker logs: `docker-compose logs celery-worker`

### Issue: Tasks timing out

**Solutions**:
- Increase time limits in `celery_app.py`
- Break large tasks into smaller chunks
- Use task chaining instead of single large task

### Issue: Memory leaks

**Solutions**:
- Reduce `worker_max_tasks_per_child` (currently 100)
- Monitor worker memory: `docker stats avalanche-celery-worker`
- Restart workers periodically in production

### Issue: Tasks stuck in PENDING

**Causes**:
- Worker not consuming from correct queue
- Task routing misconfigured
- Worker crashed before acknowledging task

**Solutions**:
```bash
# Purge all tasks
celery -A src.tasks.celery_app purge

# Restart workers
docker-compose restart celery-worker
```

## Best Practices

1. **Keep tasks idempotent**: Tasks should be safe to retry
2. **Use short time limits**: Prevent runaway tasks
3. **Monitor task queues**: Avoid queue buildup
4. **Log extensively**: Use structured logging
5. **Handle failures gracefully**: Implement retry logic
6. **Use task groups**: For parallel processing
7. **Store large results externally**: Don't overload result backend
8. **Set rate limits**: Prevent overwhelming external services
9. **Use task priorities**: For important tasks
10. **Monitor worker health**: Auto-restart failed workers

## Security Considerations

1. **Flower authentication**: Change default password
2. **Redis authentication**: Enable in production
3. **Task input validation**: Validate all task parameters
4. **Rate limiting**: Prevent task spam
5. **Resource limits**: Prevent DoS via task submission
6. **Secure result backend**: Use encrypted connections
7. **Task serialization**: Use JSON (avoid pickle in production)

## Migration from Synchronous to Asynchronous

### Before (Synchronous):

```python
def analyze_data(dataset_path):
    # Long-running operation blocks UI
    result = run_analysis(dataset_path)
    return result
```

### After (Asynchronous):

```python
def analyze_data(dataset_path):
    # Submit task, return immediately
    task = run_analysis.delay(dataset_path)
    return {'task_id': task.id, 'status': 'submitted'}

def check_analysis_status(task_id):
    # Check status later
    result = AsyncResult(task_id)
    return {
        'status': result.state,
        'progress': result.info if result.state == 'PROGRESS' else None,
        'result': result.result if result.ready() else None
    }
```

## References

- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis Documentation](https://redis.io/documentation)
- [Flower Documentation](https://flower.readthedocs.io/)
- [Task Queue Best Practices](https://docs.celeryq.dev/en/stable/userguide/tasks.html#best-practices)
