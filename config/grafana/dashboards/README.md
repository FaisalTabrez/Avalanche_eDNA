# Grafana Dashboards for Avalanche eDNA

This directory contains pre-configured Grafana dashboards for monitoring the Avalanche eDNA platform.

## Available Dashboards

### 1. System Overview
- **File**: `system-overview.json`
- **Description**: High-level system metrics including CPU, memory, disk, and network
- **Panels**:
  - CPU usage (overall and per core)
  - Memory usage (used, available, cached)
  - Disk I/O and space
  - Network traffic
  - System load average

### 2. Application Metrics
- **File**: `application-metrics.json`
- **Description**: Application-specific metrics for Streamlit and API
- **Panels**:
  - HTTP request rate and duration
  - Error rates by endpoint
  - Active connections
  - Request/response sizes
  - Top endpoints by traffic

### 3. Database Performance
- **File**: `database-metrics.json`
- **Description**: PostgreSQL database metrics
- **Panels**:
  - Connection pool status
  - Query performance
  - Transaction rates
  - Database size growth
  - Cache hit ratios
  - Lock statistics
  - Replication lag (if applicable)

### 4. Task Queue Monitor
- **File**: `celery-metrics.json`
- **Description**: Celery task queue monitoring
- **Panels**:
  - Active workers
  - Task rates (submitted, started, completed, failed)
  - Queue lengths by priority
  - Task duration percentiles
  - Worker resource usage
  - Task failure analysis

### 5. Redis Cache
- **File**: `redis-metrics.json`
- **Description**: Redis cache and broker metrics
- **Panels**:
  - Memory usage
  - Connected clients
  - Operations per second
  - Hit/miss ratios
  - Eviction rates
  - Keyspace statistics

### 6. Backup & Maintenance
- **File**: `backup-metrics.json`
- **Description**: Backup job monitoring
- **Panels**:
  - Last backup timestamp
  - Backup success/failure rates
  - Backup sizes over time
  - Backup duration trends
  - Storage usage for backups

## Dashboard Installation

Dashboards are automatically provisioned when Grafana starts (see `../provisioning/dashboards/dashboards.yml`).

### Manual Import

If you need to manually import a dashboard:

1. Access Grafana: http://localhost:3000
2. Login (admin/admin)
3. Click "+" → "Import"
4. Upload JSON file or paste JSON content
5. Select Prometheus datasource
6. Click "Import"

## Customization

To customize dashboards:

1. Open dashboard in Grafana
2. Click gear icon (⚙️) → "Settings"
3. Modify panels, add new queries, adjust time ranges
4. Click "Save dashboard"
5. Export JSON to update file

## Variables

Most dashboards support the following variables:

- **$datasource**: Prometheus datasource
- **$interval**: Query interval (auto-calculated)
- **$instance**: Instance/server filter
- **$job**: Job/service filter

## Alerts

Dashboards include alert panels that change color based on thresholds:

- 🟢 Green: Normal operation
- 🟡 Yellow: Warning level
- 🔴 Red: Critical level

## Best Practices

1. **Use time ranges wisely**:
   - Last 5 minutes for real-time monitoring
   - Last 1 hour for recent trends
   - Last 24 hours for daily patterns
   - Last 7 days for weekly trends

2. **Create custom dashboards** for specific workflows

3. **Share dashboards** with team using JSON exports

4. **Set up annotations** for deployments and incidents

5. **Use templating** for multi-environment monitoring

## Troubleshooting

### Dashboard shows "No data"

- Check Prometheus is scraping targets
- Verify datasource configuration
- Check time range selection
- Verify metric names in queries

### Slow dashboard loading

- Reduce time range
- Increase query interval
- Disable auto-refresh temporarily
- Optimize panel queries

### Missing panels

- Check Prometheus target is up
- Verify exporter is running
- Check firewall rules
- Review Prometheus logs

## Dashboard Refresh

- **Auto-refresh**: Dashboards auto-refresh every 30s-1m (configurable)
- **Manual refresh**: Click 🔄 button in top-right
- **Time range**: Use time picker in top-right corner

## Export & Backup

To backup dashboards:

```bash
# Export all dashboards
docker exec -it avalanche-grafana grafana-cli admin export

# Backup to file
cp config/grafana/dashboards/*.json backups/dashboards/
```

## References

- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
- [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/)
- [Prometheus Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
