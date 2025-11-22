# Monitoring Quick Reference

## 🚀 Quick Start

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Stop monitoring stack
docker-compose --profile monitoring down

# View logs
docker-compose logs -f prometheus grafana alertmanager
```

## 🌐 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | None |
| Alertmanager | http://localhost:9093 | None |
| Application Metrics | http://localhost:8000/metrics | None |

## 📊 Key Metrics

### System Health
```promql
# CPU Usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory Usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk Usage
(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100
```

### Application Performance
```promql
# Request Rate
rate(http_requests_total[5m])

# Error Rate
rate(http_requests_total{status=~"5.."}[5m])

# Response Time (p95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Database Performance
```promql
# Active Connections
pg_stat_activity_count

# Transaction Rate
rate(pg_stat_database_xact_commit[5m])

# Cache Hit Ratio
pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read)
```

### Task Queue
```promql
# Active Workers
celery_workers_active

# Task Rate
rate(celery_tasks_total[5m])

# Queue Length
celery_queue_length
```

## 🔔 Alert Severity

- **🔴 Critical**: Immediate action required (system down, data loss risk)
- **🟡 Warning**: Monitor and plan action (high resource usage, degraded performance)
- **🔵 Info**: Informational (high traffic, successful deployments)

## 🛠️ Common Tasks

### Check Target Status
```bash
# In Prometheus UI
http://localhost:9090/targets

# Or via API
curl http://localhost:9090/api/v1/targets
```

### Reload Configuration
```bash
# Prometheus (no restart needed)
curl -X POST http://localhost:9090/-/reload

# Alertmanager
curl -X POST http://localhost:9093/-/reload
```

### Query Metrics
```bash
# Via API
curl 'http://localhost:9090/api/v1/query?query=up'

# Via PromQL (in web UI)
# Navigate to http://localhost:9090/graph
```

### Export Dashboard
```bash
# Backup all dashboards
docker exec avalanche-grafana grafana-cli admin export > dashboards-backup.json

# Backup specific dashboard
# Go to Grafana UI → Dashboard Settings → JSON Model → Copy
```

## 📈 Useful Queries

### Top 5 Endpoints by Request Count
```promql
topk(5, sum by (endpoint) (rate(http_requests_total[5m])))
```

### Failed Analysis Jobs
```promql
analysis_runs_total{status="failed"}
```

### Database Connection Pool Utilization
```promql
(pg_stat_activity_count / pg_settings_max_connections) * 100
```

### Redis Memory Usage
```promql
redis_memory_used_bytes / redis_memory_max_bytes * 100
```

### Backup Success Rate (last 24h)
```promql
sum(rate(backup_success[24h])) / sum(rate(backup_runs_total[24h])) * 100
```

## 🔧 Troubleshooting

### No Data in Grafana
1. Check Prometheus is scraping: http://localhost:9090/targets
2. Verify datasource configured: Grafana → Configuration → Data Sources
3. Test query in Prometheus first

### Alerts Not Firing
1. Check rules loaded: http://localhost:9090/rules
2. Verify expression evaluates: Test in Prometheus graph
3. Check Alertmanager config: http://localhost:9093

### High Memory Usage
- Reduce retention time (default: 30 days)
- Increase scrape interval (default: 15s)
- Use recording rules for expensive queries

## 📚 Additional Resources

- [Full Monitoring Guide](MONITORING.md)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Tutorial](https://prometheus.io/docs/prometheus/latest/querying/basics/)
