# PostgreSQL Migration Guide

## Overview

This guide covers migrating the Avalanche eDNA system from SQLite (development) to PostgreSQL (production) for improved scalability, concurrent access, and production-grade features.

## Prerequisites

### 1. PostgreSQL Installation

**Docker (Recommended):**
```bash
# Already configured in docker-compose.yml
docker-compose up -d postgres
```

**Manual Installation:**
- **Ubuntu/Debian**: `sudo apt-get install postgresql postgresql-contrib`
- **macOS**: `brew install postgresql`
- **Windows**: Download from https://www.postgresql.org/download/windows/

### 2. Python Dependencies

```bash
pip install psycopg2-binary>=2.9.0
```

### 3. Database Setup

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE avalanche_edna;
CREATE USER avalanche WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE avalanche_edna TO avalanche;

# Exit
\q
```

## Environment Configuration

### Development (.env)

```bash
# Database Configuration
DB_TYPE=sqlite
SQLITE_PATH=data/avalanche.db
```

### Production (.env)

```bash
# Database Configuration
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=avalanche_edna
DB_USER=avalanche
DB_PASSWORD=your_secure_password

# Connection Pool Settings
DB_POOL_MIN=2
DB_POOL_MAX=20
DB_POOL_TIMEOUT=30

# Retry Configuration
DB_MAX_RETRIES=3
DB_RETRY_DELAY=1
```

### Docker Environment

The `docker-compose.yml` already includes PostgreSQL:

```yaml
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: avalanche_edna
    POSTGRES_USER: avalanche
    POSTGRES_PASSWORD: ${DB_PASSWORD}
  ports:
    - "5432:5432"
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

## Migration Process

### Step 1: Test Connections

Verify both SQLite and PostgreSQL are accessible:

```bash
python scripts/migrate_to_postgres.py --test-connection
```

Expected output:
```
DATABASE CONNECTION TEST
========================================
Testing SQLite (data/avalanche.db)...
✓ SQLite connection successful
  Path: data/avalanche.db
  Exists: True

Testing PostgreSQL (avalanche@localhost:5432/avalanche_edna)...
✓ PostgreSQL connection successful
  Host: localhost
  Database: avalanche_edna
  Pool: 2-20
========================================
```

### Step 2: Create PostgreSQL Schema

Create tables and indexes in PostgreSQL:

```bash
python scripts/migrate_to_postgres.py --setup-schema
```

This creates:
- Tables: `datasets`, `analysis_reports`, `organism_profiles`, `similarity_matrices`, `report_comparisons`
- Indexes: Performance indexes on all foreign keys and frequently queried columns
- Foreign key constraints

### Step 3: Migrate Data

Transfer all data from SQLite to PostgreSQL:

```bash
python scripts/migrate_to_postgres.py --migrate
```

The migration process:
1. Reads all data from SQLite
2. Converts data types (TEXT → JSONB, TEXT → TIMESTAMP, etc.)
3. Inserts into PostgreSQL (with conflict handling)
4. Automatically validates row counts

**Migration Order** (respecting foreign keys):
1. datasets
2. analysis_reports
3. organism_profiles
4. similarity_matrices
5. report_comparisons

### Step 4: Validate Migration

Verify all data migrated successfully:

```bash
python scripts/migrate_to_postgres.py --validate
```

Expected output:
```
MIGRATION VALIDATION RESULTS
============================================================
✓ datasets                 SQLite:     12 | PostgreSQL:     12
✓ analysis_reports         SQLite:     45 | PostgreSQL:     45
✓ organism_profiles        SQLite:    523 | PostgreSQL:    523
✓ similarity_matrices      SQLite:     38 | PostgreSQL:     38
✓ report_comparisons       SQLite:      7 | PostgreSQL:      7
============================================================
✓ All tables validated successfully!
```

### Step 5: Switch Application to PostgreSQL

Update environment variable:

```bash
# In .env or docker-compose.yml
DB_TYPE=postgresql
```

Restart application:

```bash
# Docker
docker-compose restart streamlit

# Local
streamlit run streamlit_app.py
```

### Step 6: Verify Application

1. Login to application
2. Check datasets appear correctly
3. Create new analysis
4. Verify results saved to PostgreSQL

## Data Type Conversions

The migration automatically converts SQLite types to PostgreSQL:

| SQLite Type | PostgreSQL Type | Conversion |
|-------------|----------------|------------|
| TEXT (JSON) | JSONB | Parse JSON string |
| TEXT (ISO datetime) | TIMESTAMP | Parse datetime |
| TEXT (arrays) | TEXT[] | Parse JSON array |
| TEXT | VARCHAR/TEXT | Direct copy |
| INTEGER | INTEGER | Direct copy |
| REAL | REAL | Direct copy |
| BLOB | BYTEA | Direct copy |

## Schema Differences

### SQLite Schema
```sql
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metadata TEXT  -- JSON stored as TEXT
)
```

### PostgreSQL Schema
```sql
CREATE TABLE datasets (
    dataset_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    metadata JSONB  -- Native JSON type
)
```

## Performance Improvements

### Indexes Created

```sql
-- Example indexes for optimal query performance
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX idx_reports_dataset_id ON analysis_reports(dataset_id);
CREATE INDEX idx_reports_status ON analysis_reports(status);
CREATE INDEX idx_profiles_organism ON organism_profiles(organism_name);
```

### Connection Pooling

PostgreSQL uses connection pooling for efficiency:

- **Min connections**: 2 (always available)
- **Max connections**: 20 (scales under load)
- **Timeout**: 30 seconds
- **Reuse**: Connections reused across requests

### Query Performance

Expected improvements after migration:

| Operation | SQLite | PostgreSQL | Improvement |
|-----------|--------|------------|-------------|
| List datasets | 50ms | 15ms | 3.3x faster |
| Search organisms | 200ms | 45ms | 4.4x faster |
| Complex joins | 500ms | 80ms | 6.25x faster |
| Concurrent writes | Limited | Excellent | Multi-user |

## Rollback Procedure

If issues occur, rollback to SQLite:

1. **Stop application**:
   ```bash
   docker-compose stop streamlit
   ```

2. **Switch to SQLite**:
   ```bash
   # In .env
   DB_TYPE=sqlite
   ```

3. **Restart application**:
   ```bash
   docker-compose start streamlit
   ```

4. **SQLite data is unchanged** - original database remains intact

## Production Checklist

Before production deployment:

- [ ] PostgreSQL installed and configured
- [ ] Database user created with secure password
- [ ] Connection pooling configured
- [ ] Schema created successfully
- [ ] Data migration completed
- [ ] Validation passed (all row counts match)
- [ ] Application tested with PostgreSQL
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] SSL/TLS enabled for database connections (production)

## Monitoring

### Database Health Check

```python
from src.database import get_database_connection

db = get_database_connection()
info = db.get_database_info()

print(f"Type: {info['type']}")
print(f"Healthy: {info['healthy']}")
```

### Connection Pool Status

```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity 
WHERE datname = 'avalanche_edna';

-- Check slow queries
SELECT pid, now() - query_start as duration, query 
FROM pg_stat_activity 
WHERE state = 'active' 
ORDER BY duration DESC;
```

## Troubleshooting

### Connection Refused

**Error**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
1. Check PostgreSQL is running: `systemctl status postgresql`
2. Verify host/port in .env
3. Check firewall allows port 5432

### Authentication Failed

**Error**: `FATAL: password authentication failed`

**Solution**:
1. Verify DB_PASSWORD in .env
2. Check user exists: `psql -U postgres -c "\du"`
3. Reset password: `ALTER USER avalanche WITH PASSWORD 'new_password';`

### Migration Validation Fails

**Error**: Row count mismatch

**Solution**:
1. Check migration logs for errors
2. Re-run migration: `python scripts/migrate_to_postgres.py --migrate`
3. If persists, check for constraint violations in PostgreSQL logs

### Slow Queries

**Error**: Queries taking longer than expected

**Solution**:
1. Verify indexes created: `\d+ table_name` in psql
2. Analyze tables: `ANALYZE;`
3. Check query plans: `EXPLAIN ANALYZE SELECT ...`
4. Increase connection pool size if needed

## Advanced Configuration

### SSL Connections (Production)

```python
# In .env
DB_SSL_MODE=require
DB_SSL_CERT=/path/to/client-cert.pem
DB_SSL_KEY=/path/to/client-key.pem
DB_SSL_ROOT_CERT=/path/to/server-ca.pem
```

### Read Replicas

For high-traffic deployments:

```python
# Primary (write)
DB_PRIMARY_HOST=primary.db.example.com

# Replica (read)
DB_REPLICA_HOST=replica.db.example.com
```

### Partitioning Large Tables

For datasets with millions of rows:

```sql
-- Partition by date
CREATE TABLE analysis_reports (
    ...
) PARTITION BY RANGE (created_at);

CREATE TABLE reports_2024 PARTITION OF analysis_reports
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

## Backup and Recovery

See `BACKUP_GUIDE.md` (Phase 1.5) for:
- Automated PostgreSQL backups
- Point-in-time recovery
- Cloud backup integration

## Next Steps

After successful migration:

1. **Monitor performance** for first week
2. **Optimize queries** based on usage patterns
3. **Set up automated backups** (Phase 1.5)
4. **Enable replication** for high availability
5. **Configure monitoring alerts** (Phase 2.3)

## Support

For issues:
1. Check logs: `docker-compose logs postgres`
2. Review PostgreSQL logs: `/var/log/postgresql/`
3. Test connection: `python scripts/migrate_to_postgres.py --test-connection`
4. File issue on GitHub with error details
