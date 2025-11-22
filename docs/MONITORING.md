# Monitoring & Observability Guide

## Overview

The Avalanche eDNA platform implements comprehensive monitoring and observability using **Prometheus** for metrics collection, **Grafana** for visualization, and **Alertmanager** for alerting.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Monitoring Stack                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐        ┌──────────────┐                   │
│  │  Streamlit   │───────>│ Prometheus   │<───────────┐      │
│  │  (metrics)   │        │  (scraping)  │            │      │
│  └──────────────┘        └──────┬───────┘            │      │
│                                  │                    │      │
│  ┌──────────────┐                │              ┌─────────┐ │
│  │   Celery     │────────────────┤              │Exporters│ │
│  │  (metrics)   │                │              │         │ │
│  └──────────────┘                │              │- Node   │ │
│                                  │              │- Postgres│ │
│  ┌──────────────┐                │              │- Redis  │ │
│  │  PostgreSQL  │<───────────────┤              └─────────┘ │
│  │  (metrics)   │                │                    ▲      │
│  └──────────────┘                │                    │      │
│                                  │                    │      │
│  ┌──────────────┐                │                    │      │
│  │    Redis     │<───────────────┘                    │      │
│  │  (metrics)   │                                     │      │
│  └──────────────┘                                     │      │
│         │                                              │      │
│         └──────────────────────────────────────────────┘      │
│                                                               │
│         ┌──────────────┐        ┌──────────────┐             │
│         │  Grafana     │<───────│ Prometheus   │             │
│         │(dashboards)  │        │   (query)    │             │
│         └──────────────┘        └──────┬───────┘             │
│                                        │                     │
│         ┌──────────────┐               │                     │
│         │Alertmanager  │<──────────────┘                     │
│         │  (alerts)    │                                     │
│         └──────┬───────┘                                     │
│                │                                              │
│         ┌──────▼───────────────────┐                         │
│         │  Email / Slack / Webhook │                         │
│         └──────────────────────────┘                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Prometheus

**Purpose**: Time-series database for metrics collection

**Configuration**: `config/prometheus/prometheus.yml`

**Metrics Sources**:
- Application metrics (port 8000)
- Node Exporter (system metrics, port 9100)
- PostgreSQL Exporter (database metrics, port 9187)
- Redis Exporter (cache metrics, port 9121)
- Celery metrics (task queue, port 9808)

**Web UI**: http://localhost:9090

**Key Features**:
- 15-second scrape interval
- 30-day data retention
- Rule-based alerting
- PromQL query language

### 2. Grafana

**Purpose**: Metrics visualization and dashboarding

**Configuration**: `config/grafana/`

**Web UI**: http://localhost:3000
- **Username**: admin
- **Password**: admin (change in production!)

**Features**:
- Pre-configured dashboards
- Real-time monitoring
- Alert annotations
- Custom panels and queries

### 3. Alertmanager

**Purpose**: Alert routing and notification

**Configuration**: `config/alertmanager/alertmanager.yml`

**Web UI**: http://localhost:9093

**Notification Channels**:
- Email (SMTP)
- Slack (webhooks)
- Webhooks (custom integrations)

### 4. Exporters

#### Node Exporter
- **Purpose**: System metrics (CPU, memory, disk, network)
- **Port**: 9100
- **Metrics**: `node_*`

#### PostgreSQL Exporter
- **Purpose**: Database metrics
- **Port**: 9187
- **Metrics**: `pg_*`

#### Redis Exporter
- **Purpose**: Cache metrics
- **Port**: 9121
- **Metrics**: `redis_*`

## Docker Deployment

### Start Monitoring Stack

```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d

# Verify services
docker-compose ps

# Check logs
docker-compose logs -f prometheus grafana
```

### Stop Monitoring Stack

```bash
docker-compose --profile monitoring down
```

### Services Exposed

| Service | Port | URL |
|---------|------|-----|
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |
| Alertmanager | 9093 | http://localhost:9093 |
| Node Exporter | 9100 | http://localhost:9100/metrics |
| Postgres Exporter | 9187 | http://localhost:9187/metrics |
| Redis Exporter | 9121 | http://localhost:9121/metrics |
| Application Metrics | 8000 | http://localhost:8000/metrics |

## Metrics Categories

### Application Metrics

```promql
# HTTP requests
http_requests_total

# Request duration
http_request_duration_seconds

# Analysis operations
analysis_runs_total
analysis_duration_seconds
analysis_active_runs

# Training operations
training_runs_total
training_duration_seconds
training_accuracy
training_loss

# Downloads
download_jobs_total
download_bytes_total
download_active_jobs

# Backups
backup_success
backup_last_success_timestamp
backup_duration_seconds
```

### System Metrics

```promql
# CPU usage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk usage
(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100

# Network traffic
rate(node_network_receive_bytes_total[5m])
rate(node_network_transmit_bytes_total[5m])
```

### Database Metrics

```promql
# Active connections
pg_stat_activity_count

# Transaction rate
rate(pg_stat_database_xact_commit[5m])

# Database size
pg_database_size_bytes

# Cache hit ratio
pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read)
```

### Redis Metrics

```promql
# Memory usage
redis_memory_used_bytes

# Connected clients
redis_connected_clients

# Operations per second
rate(redis_commands_total[5m])

# Hit rate
rate(redis_keyspace_hits_total[5m]) / (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m]))
```

## Alert Rules

### Critical Alerts

**High Priority** - Immediate attention required:

- `PostgreSQLDown`: Database is offline
- `RedisDown`: Cache is offline
- `CeleryWorkerDown`: Task worker is offline
- `ApplicationDown`: Application is unreachable
- `LowDiskSpace`: < 10% disk space remaining
- `NoWorkersAvailable`: All Celery workers offline

### Warning Alerts

**Medium Priority** - Monitor and plan action:

- `HighCPUUsage`: > 90% for 5 minutes
- `HighMemoryUsage`: > 90% for 5 minutes
- `HighSystemLoad`: Load > 0.8 per CPU
- `PostgreSQLTooManyConnections`: > 80 connections
- `TaskQueueBackup`: > 100 tasks queued
- `NoRecentBackup`: No backup in 24 hours

### Info Alerts

**Low Priority** - Informational:

- `HighRequestRate`: > 100 req/s
- `DatabaseSizeGrowth`: Rapid database growth

## Grafana Dashboards

### System Overview Dashboard

**Metrics**:
- CPU usage per core
- Memory usage (used/available/cached)
- Disk I/O and space
- Network throughput
- System load average

**Use Cases**:
- Monitor resource utilization
- Identify performance bottlenecks
- Capacity planning

### Application Metrics Dashboard

**Metrics**:
- HTTP request rate
- Request duration (p50, p95, p99)
- Error rates by endpoint
- Active sessions
- Top endpoints by traffic

**Use Cases**:
- Application performance monitoring
- User activity tracking
- Error analysis

### Database Performance Dashboard

**Metrics**:
- Connection pool status
- Query performance
- Transaction rates
- Database size
- Cache hit ratios
- Lock contention

**Use Cases**:
- Query optimization
- Connection pool tuning
- Database health monitoring

### Task Queue Dashboard

**Metrics**:
- Active workers
- Task throughput
- Queue lengths
- Task duration
- Failure rates
- Worker resource usage

**Use Cases**:
- Task queue health
- Worker capacity planning
- Failure investigation

## Configuration

### Prometheus Configuration

Edit `config/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s  # How often to scrape
  evaluation_interval: 15s  # How often to evaluate rules

scrape_configs:
  - job_name: 'my-service'
    static_configs:
      - targets: ['my-service:9090']
        labels:
          environment: 'production'
```

### Alert Rules

Add rules to `config/prometheus/alerts/rules.yml`:

```yaml
groups:
  - name: my_alerts
    rules:
      - alert: MyAlert
        expr: metric_name > threshold
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Alert summary"
          description: "Alert description"
```

### Alertmanager Configuration

Edit `config/alertmanager/alertmanager.yml`:

```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'alerts@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'user@gmail.com'
        auth_password: 'password'
```

## Usage Examples

### Query Metrics

#### Prometheus Web UI

1. Open http://localhost:9090
2. Go to Graph tab
3. Enter PromQL query
4. Click "Execute"

**Example Queries**:

```promql
# Average CPU usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Request rate
rate(http_requests_total[5m])

# Failed analyses
analysis_runs_total{status="failed"}

# Database connections
pg_stat_activity_count

# Redis memory
redis_memory_used_bytes / 1024 / 1024  # MB
```

### Create Custom Dashboard

1. Open Grafana (http://localhost:3000)
2. Click "+" → "Dashboard"
3. Click "Add new panel"
4. Enter PromQL query
5. Configure visualization
6. Click "Apply"
7. Click "Save dashboard"

### Set Up Alerts

#### In Prometheus

Add to `config/prometheus/alerts/rules.yml`:

```yaml
- alert: HighAPILatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High API latency detected"
    description: "95th percentile latency is {{ $value }}s"
```

#### In Grafana

1. Edit panel
2. Click "Alert" tab
3. Click "Create Alert"
4. Set conditions
5. Configure notifications
6. Save

## Monitoring Best Practices

### 1. Metric Collection

- **Collect meaningful metrics**: Focus on actionable data
- **Use labels wisely**: Don't create high-cardinality metrics
- **Set appropriate intervals**: Balance resolution vs. storage
- **Monitor metric cardinality**: Avoid label explosion

### 2. Dashboards

- **Keep it simple**: One purpose per dashboard
- **Use templates**: Create reusable dashboards
- **Set proper time ranges**: Match use case
- **Add documentation**: Annotate panels with descriptions

### 3. Alerting

- **Alert on symptoms, not causes**: Focus on user impact
- **Set appropriate thresholds**: Avoid alert fatigue
- **Group related alerts**: Reduce noise
- **Document runbooks**: Include remediation steps
- **Test alert routing**: Verify notifications work

### 4. Performance

- **Use recording rules**: Pre-compute expensive queries
- **Limit query range**: Don't query years of data
- **Use downsampling**: Aggregate old data
- **Monitor Prometheus itself**: Watch scrape duration

## Troubleshooting

### No Metrics in Grafana

**Check**:
1. Prometheus is running: `docker ps | grep prometheus`
2. Targets are up: http://localhost:9090/targets
3. Datasource configured: Grafana → Configuration → Data Sources
4. Query syntax: Test in Prometheus first

### Alerts Not Firing

**Check**:
1. Rules loaded: http://localhost:9090/rules
2. Alert conditions met: Check expression in Prometheus
3. Alertmanager connected: http://localhost:9090/config
4. Notification configured: http://localhost:9093

### High Resource Usage

**Solutions**:
- Increase scrape interval
- Reduce retention time
- Use recording rules
- Add more Prometheus instances

### Missing Data

**Causes**:
- Target down
- Network issues
- Incorrect scrape configuration
- Exporter crashed

**Check**:
```bash
# View Prometheus logs
docker logs avalanche-prometheus

# Check target status
curl http://localhost:9090/api/v1/targets

# Test exporter directly
curl http://localhost:9100/metrics
```

## Maintenance

### Backup Prometheus Data

```bash
# Stop Prometheus
docker-compose stop prometheus

# Backup data directory
tar -czf prometheus-backup.tar.gz config/prometheus/ prometheus-data/

# Restart Prometheus
docker-compose start prometheus
```

### Backup Grafana Dashboards

```bash
# Export dashboards
docker exec avalanche-grafana grafana-cli admin export > dashboards-backup.json

# Or copy dashboard files
cp config/grafana/dashboards/*.json backups/
```

### Update Configuration

```bash
# Edit configuration
vim config/prometheus/prometheus.yml

# Reload configuration (no restart needed)
curl -X POST http://localhost:9090/-/reload
```

### Clean Up Old Data

```bash
# Prometheus automatically removes data after retention period (30 days)
# To manually clean:
docker exec avalanche-prometheus \
  promtool tsdb cleanup --max-block-duration=24h /prometheus
```

## Security Considerations

1. **Change default passwords**:
   - Grafana: admin/admin
   - Alertmanager SMTP credentials

2. **Enable authentication**:
   - Prometheus (use reverse proxy)
   - Grafana (enable auth providers)

3. **Use HTTPS**:
   - Configure TLS in production
   - Use valid certificates

4. **Restrict access**:
   - Firewall rules
   - Network policies
   - VPN/bastion hosts

5. **Secure sensitive data**:
   - Use secrets management
   - Encrypt credentials
   - Rotate keys regularly

## Integration Examples

### Application Code

```python
from src.monitoring.metrics import application_metrics

# Record analysis
application_metrics.record_analysis(
    analysis_type='taxonomic',
    duration=123.45,
    sequences=1000,
    status='success'
)

# Record training
application_metrics.record_training(
    model_type='transformer',
    duration=3600.0,
    accuracy=0.95,
    loss=0.05,
    run_id='run_123'
)

# Record download
application_metrics.record_download(
    source='SRA',
    duration=600.0,
    bytes_downloaded=1024**3,  # 1GB
    status='success'
)
```

### Using Decorators

```python
from src.monitoring.metrics import track_analysis, track_training

@track_analysis('taxonomic')
def run_taxonomic_analysis(data):
    # Analysis code
    return results

@track_training('transformer')
def train_transformer_model(data):
    # Training code
    return model
```

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
- [Grafana Dashboard Examples](https://grafana.com/grafana/dashboards/)
