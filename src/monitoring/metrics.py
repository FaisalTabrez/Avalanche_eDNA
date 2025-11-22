"""
Prometheus metrics exporter for Avalanche eDNA application

This module exposes application metrics for Prometheus scraping.
"""

import time
import psutil
from datetime import datetime
from functools import wraps
from flask import Flask, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)

from src.database.manager import DatabaseManager


# Create custom registry
registry = CollectorRegistry()

# Application info
app_info = Info('avalanche_application', 'Application information', registry=registry)
app_info.info({
    'version': '1.0.0',
    'name': 'Avalanche eDNA',
    'environment': 'development'
})

# ============================================================================
# HTTP Metrics
# ============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

http_request_size_bytes = Summary(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

http_response_size_bytes = Summary(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

# ============================================================================
# Analysis Metrics
# ============================================================================

analysis_runs_total = Counter(
    'analysis_runs_total',
    'Total number of analysis runs',
    ['analysis_type', 'status'],
    registry=registry
)

analysis_duration_seconds = Histogram(
    'analysis_duration_seconds',
    'Analysis duration in seconds',
    ['analysis_type'],
    buckets=(30, 60, 120, 300, 600, 1800, 3600, 7200),
    registry=registry
)

analysis_sequences_processed = Counter(
    'analysis_sequences_processed_total',
    'Total number of sequences processed',
    ['analysis_type'],
    registry=registry
)

analysis_active_runs = Gauge(
    'analysis_active_runs',
    'Number of currently active analysis runs',
    ['analysis_type'],
    registry=registry
)

# ============================================================================
# Training Metrics
# ============================================================================

training_runs_total = Counter(
    'training_runs_total',
    'Total number of training runs',
    ['model_type', 'status'],
    registry=registry
)

training_duration_seconds = Histogram(
    'training_duration_seconds',
    'Training duration in seconds',
    ['model_type'],
    buckets=(300, 600, 1800, 3600, 7200, 14400, 28800),
    registry=registry
)

training_accuracy = Gauge(
    'training_accuracy',
    'Model training accuracy',
    ['model_type', 'run_id'],
    registry=registry
)

training_loss = Gauge(
    'training_loss',
    'Model training loss',
    ['model_type', 'run_id'],
    registry=registry
)

# ============================================================================
# Download Metrics
# ============================================================================

download_jobs_total = Counter(
    'download_jobs_total',
    'Total number of download jobs',
    ['source', 'status'],
    registry=registry
)

download_duration_seconds = Histogram(
    'download_duration_seconds',
    'Download duration in seconds',
    ['source'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400),
    registry=registry
)

download_bytes_total = Counter(
    'download_bytes_total',
    'Total bytes downloaded',
    ['source'],
    registry=registry
)

download_active_jobs = Gauge(
    'download_active_jobs',
    'Number of currently active downloads',
    ['source'],
    registry=registry
)

# ============================================================================
# Database Metrics
# ============================================================================

database_connections = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=registry
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    registry=registry
)

database_size_bytes = Gauge(
    'database_size_bytes',
    'Database size in bytes',
    ['database'],
    registry=registry
)

database_tables_count = Gauge(
    'database_tables_count',
    'Number of database tables',
    registry=registry
)

# ============================================================================
# System Metrics
# ============================================================================

system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes',
    ['type'],
    registry=registry
)

system_disk_usage_bytes = Gauge(
    'system_disk_usage_bytes',
    'Disk usage in bytes',
    ['path', 'type'],
    registry=registry
)

# ============================================================================
# Backup Metrics
# ============================================================================

backup_last_success_timestamp = Gauge(
    'backup_last_success_timestamp',
    'Timestamp of last successful backup',
    ['backup_type'],
    registry=registry
)

backup_success = Gauge(
    'backup_success',
    'Whether last backup was successful (1=success, 0=failure)',
    ['backup_type'],
    registry=registry
)

backup_duration_seconds = Histogram(
    'backup_duration_seconds',
    'Backup duration in seconds',
    ['backup_type'],
    buckets=(30, 60, 300, 600, 1800, 3600),
    registry=registry
)

backup_size_bytes = Gauge(
    'backup_size_bytes',
    'Backup size in bytes',
    ['backup_type'],
    registry=registry
)


# ============================================================================
# Metric Collection Functions
# ============================================================================

def collect_system_metrics():
    """Collect system metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage_percent.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage_bytes.labels(type='used').set(memory.used)
        system_memory_usage_bytes.labels(type='available').set(memory.available)
        system_memory_usage_bytes.labels(type='total').set(memory.total)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        system_disk_usage_bytes.labels(path='/', type='used').set(disk.used)
        system_disk_usage_bytes.labels(path='/', type='free').set(disk.free)
        system_disk_usage_bytes.labels(path='/', type='total').set(disk.total)
        
    except Exception as e:
        print(f"Error collecting system metrics: {e}")


def collect_database_metrics():
    """Collect database metrics"""
    try:
        db = DatabaseManager()
        
        # Connection pool stats
        if hasattr(db.engine.pool, 'size'):
            database_connections.set(db.engine.pool.size())
        
    except Exception as e:
        print(f"Error collecting database metrics: {e}")


def collect_application_metrics():
    """Collect all application metrics"""
    collect_system_metrics()
    collect_database_metrics()


# ============================================================================
# Decorators for automatic metric tracking
# ============================================================================

def track_request(endpoint):
    """Decorator to track HTTP requests"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method='GET',
                    endpoint=endpoint,
                    status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method='GET',
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator


def track_analysis(analysis_type):
    """Decorator to track analysis operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            analysis_active_runs.labels(analysis_type=analysis_type).inc()
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'failed'
                raise
            finally:
                duration = time.time() - start_time
                analysis_active_runs.labels(analysis_type=analysis_type).dec()
                analysis_runs_total.labels(
                    analysis_type=analysis_type,
                    status=status
                ).inc()
                analysis_duration_seconds.labels(
                    analysis_type=analysis_type
                ).observe(duration)
        
        return wrapper
    return decorator


def track_training(model_type):
    """Decorator to track training operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'failed'
                raise
            finally:
                duration = time.time() - start_time
                training_runs_total.labels(
                    model_type=model_type,
                    status=status
                ).inc()
                training_duration_seconds.labels(
                    model_type=model_type
                ).observe(duration)
        
        return wrapper
    return decorator


def track_download(source):
    """Decorator to track download operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            download_active_jobs.labels(source=source).inc()
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                # Track bytes if available in result
                if isinstance(result, dict) and 'bytes' in result:
                    download_bytes_total.labels(source=source).inc(result['bytes'])
                return result
            except Exception as e:
                status = 'failed'
                raise
            finally:
                duration = time.time() - start_time
                download_active_jobs.labels(source=source).dec()
                download_jobs_total.labels(
                    source=source,
                    status=status
                ).inc()
                download_duration_seconds.labels(
                    source=source
                ).observe(duration)
        
        return wrapper
    return decorator


# ============================================================================
# Metrics Application Class
# ============================================================================

class ApplicationMetrics:
    """Application metrics manager"""
    
    def __init__(self):
        self.registry = registry
    
    def record_analysis(self, analysis_type, duration, status='success', sequences=0):
        """Record analysis metrics"""
        analysis_runs_total.labels(
            analysis_type=analysis_type,
            status=status
        ).inc()
        analysis_duration_seconds.labels(
            analysis_type=analysis_type
        ).observe(duration)
        if sequences > 0:
            analysis_sequences_processed.labels(
                analysis_type=analysis_type
            ).inc(sequences)
    
    def record_training(self, model_type, duration, accuracy=None, loss=None, run_id=None, status='success'):
        """Record training metrics"""
        training_runs_total.labels(
            model_type=model_type,
            status=status
        ).inc()
        training_duration_seconds.labels(
            model_type=model_type
        ).observe(duration)
        
        if accuracy is not None and run_id:
            training_accuracy.labels(
                model_type=model_type,
                run_id=run_id
            ).set(accuracy)
        
        if loss is not None and run_id:
            training_loss.labels(
                model_type=model_type,
                run_id=run_id
            ).set(loss)
    
    def record_download(self, source, duration, bytes_downloaded=0, status='success'):
        """Record download metrics"""
        download_jobs_total.labels(
            source=source,
            status=status
        ).inc()
        download_duration_seconds.labels(
            source=source
        ).observe(duration)
        if bytes_downloaded > 0:
            download_bytes_total.labels(source=source).inc(bytes_downloaded)
    
    def record_backup(self, backup_type, duration, size_bytes=0, success=True):
        """Record backup metrics"""
        backup_success.labels(backup_type=backup_type).set(1 if success else 0)
        if success:
            backup_last_success_timestamp.labels(
                backup_type=backup_type
            ).set(time.time())
        backup_duration_seconds.labels(
            backup_type=backup_type
        ).observe(duration)
        if size_bytes > 0:
            backup_size_bytes.labels(
                backup_type=backup_type
            ).set(size_bytes)


# Global metrics instance
application_metrics = ApplicationMetrics()


# ============================================================================
# Flask Application for Metrics Endpoint
# ============================================================================

metrics_app = Flask(__name__)


@metrics_app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    # Collect latest metrics before serving
    collect_application_metrics()
    
    # Generate metrics in Prometheus format
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)


@metrics_app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}


if __name__ == '__main__':
    # Run metrics server
    metrics_app.run(host='0.0.0.0', port=8000)
