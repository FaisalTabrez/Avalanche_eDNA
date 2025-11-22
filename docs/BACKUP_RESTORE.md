# Backup and Restore Documentation

This document describes the automated backup system for Avalanche and restoration procedures.

## Overview

Avalanche includes a comprehensive backup system that supports:
- **Database backups** (PostgreSQL and SQLite)
- **File system backups** (datasets, results, configurations)
- **Cloud storage integration** (AWS S3, Azure Blob, Google Cloud Storage)
- **Automated scheduling** (cron jobs)
- **Retention policies** (daily, weekly, monthly)
- **Backup verification** (checksum validation)
- **Compression** (gzip for space efficiency)

## Architecture

### Components

1. **BackupManager** (`scripts/backup/backup_manager.py`)
   - Core backup logic
   - Handles database and file backups
   - Manages compression and cloud uploads
   - Implements retention policies

2. **RestoreManager** (`scripts/backup/restore_manager.py`)
   - Restoration logic
   - Supports database and file restoration
   - Handles decompression

3. **Automated Scripts**
   - `database_backup.sh` - Daily database backups
   - `files_backup.sh` - Weekly file backups
   - `full_backup.sh` - Full system backups

4. **Configuration** (`config/backup.yaml`)
   - Backup settings
   - Retention policies
   - Cloud storage credentials
   - Schedule configuration

## Quick Start

### Manual Backup

```bash
# Backup database only
python scripts/backup/backup_manager.py database

# Backup files only
python scripts/backup/backup_manager.py files

# Full system backup (database + files)
python scripts/backup/backup_manager.py full

# List all backups
python scripts/backup/backup_manager.py list

# Verify a backup
python scripts/backup/backup_manager.py verify --backup-id database_20241122_020000

# Upload backup to cloud
python scripts/backup/backup_manager.py upload --backup-id database_20241122_020000
```

### Manual Restore

```bash
# List available backups
python scripts/backup/restore_manager.py list

# Restore database
python scripts/backup/restore_manager.py database --backup-id database_20241122_020000

# Restore files
python scripts/backup/restore_manager.py files --backup-id files_20241122_030000

# Force restore (skip confirmation)
python scripts/backup/restore_manager.py database --backup-id database_20241122_020000 --force
```

## Configuration

### Basic Setup

1. **Edit configuration** (`config/backup.yaml`):

```yaml
backup:
  root_dir: data/backups
  retention:
    daily: 7
    weekly: 4
    monthly: 12
  compression: true
  cloud:
    enabled: false
```

2. **Set environment variables** (`.env`):

```bash
# Database configuration
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=avalanche_edna
DB_USER=avalanche
DB_PASSWORD=your_secure_password

# Backup settings
BACKUP_CLOUD_ENABLED=false
```

### Cloud Storage Configuration

#### AWS S3

1. **Update configuration** (`config/backup.yaml`):

```yaml
backup:
  cloud:
    enabled: true
    provider: s3
    s3:
      bucket: avalanche-backups
      prefix: production/
      region: us-east-1
```

2. **Set AWS credentials**:

```bash
# In .env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

3. **Install boto3**:

```bash
pip install boto3
```

#### Azure Blob Storage

1. **Update configuration**:

```yaml
backup:
  cloud:
    enabled: true
    provider: azure
    azure:
      container: avalanche-backups
      prefix: production/
```

2. **Set connection string**:

```bash
# In .env
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."
```

3. **Install Azure SDK**:

```bash
pip install azure-storage-blob
```

#### Google Cloud Storage

1. **Update configuration**:

```yaml
backup:
  cloud:
    enabled: true
    provider: gcs
    gcs:
      bucket: avalanche-backups
      prefix: production/
      project_id: your-project-id
```

2. **Set credentials**:

```bash
# In .env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

3. **Install Google Cloud SDK**:

```bash
pip install google-cloud-storage
```

## Automated Backups

### Linux/macOS (Cron)

1. **Make scripts executable**:

```bash
chmod +x scripts/backup/*.sh
```

2. **Edit crontab**:

```bash
crontab -e
```

3. **Add backup schedules**:

```cron
# Database backup - Daily at 2 AM
0 2 * * * /path/to/avalanche/scripts/backup/database_backup.sh

# Files backup - Weekly on Sunday at 3 AM
0 3 * * 0 /path/to/avalanche/scripts/backup/files_backup.sh

# Full backup - Monthly on 1st at 4 AM
0 4 1 * * /path/to/avalanche/scripts/backup/full_backup.sh

# Cleanup old backups - Weekly on Sunday at 5 AM
0 5 * * 0 cd /path/to/avalanche && python3 scripts/backup/backup_manager.py cleanup
```

4. **Verify cron jobs**:

```bash
crontab -l
```

### Windows (Task Scheduler)

1. **Open Task Scheduler**:
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create Task for Database Backup**:
   - Action → Create Basic Task
   - Name: "Avalanche Database Backup"
   - Trigger: Daily at 2:00 AM
   - Action: Start a program
   - Program: `C:\Python\python.exe`
   - Arguments: `scripts\backup\backup_manager.py database`
   - Start in: `C:\Volume D\Avalanche`

3. **Create Task for Files Backup**:
   - Name: "Avalanche Files Backup"
   - Trigger: Weekly on Sunday at 3:00 AM
   - Program: `C:\Python\python.exe`
   - Arguments: `scripts\backup\backup_manager.py files`

4. **Create Task for Cleanup**:
   - Name: "Avalanche Backup Cleanup"
   - Trigger: Weekly on Sunday at 5:00 AM
   - Program: `C:\Python\python.exe`
   - Arguments: `scripts\backup\backup_manager.py cleanup`

### Docker (Cron Container)

Add to `docker-compose.yml`:

```yaml
services:
  backup-scheduler:
    image: alpine:latest
    container_name: avalanche_backup_scheduler
    volumes:
      - ./scripts/backup:/backup
      - ./data:/data
      - ./.env:/app/.env
    command: |
      sh -c "
      apk add --no-cache python3 py3-pip postgresql-client &&
      pip3 install pyyaml &&
      echo '0 2 * * * cd /app && python3 /backup/backup_manager.py database' > /etc/crontabs/root &&
      echo '0 3 * * 0 cd /app && python3 /backup/backup_manager.py files' >> /etc/crontabs/root &&
      echo '0 5 * * 0 cd /app && python3 /backup/backup_manager.py cleanup' >> /etc/crontabs/root &&
      crond -f
      "
    networks:
      - avalanche_network
```

## Backup Strategy

### Recommended Schedule

| Backup Type | Frequency | Retention | Storage |
|-------------|-----------|-----------|---------|
| Database | Daily 2 AM | 7 days | Local + Cloud |
| Files | Weekly (Sunday 3 AM) | 4 weeks | Local + Cloud |
| Full System | Monthly (1st at 4 AM) | 12 months | Cloud only |
| Cleanup | Weekly (Sunday 5 AM) | - | - |

### Retention Policy

The system implements a **grandfather-father-son** retention strategy:

- **Daily backups**: Kept for 7 days
- **Weekly backups**: Kept for 4 weeks (Sunday backups)
- **Monthly backups**: Kept for 12 months (1st of month backups)

Backups are automatically cleaned up based on this policy.

## Backup Structure

```
data/backups/
├── database/
│   ├── database_20241122_020000.sql.gz
│   ├── database_20241122_020000_metadata.json
│   └── ...
├── files/
│   ├── files_20241122_030000.tar.gz
│   ├── files_20241122_030000_metadata.json
│   └── ...
└── logs/
    ├── backup_20241122.log
    └── ...
```

### Metadata Format

Each backup has an associated metadata file:

```json
{
  "backup_id": "database_20241122_020000",
  "timestamp": "2024-11-22T02:00:00",
  "backup_type": "database",
  "size_bytes": 52428800,
  "checksum": "a1b2c3d4...",
  "files_included": ["database_20241122_020000.sql.gz"],
  "compressed": true,
  "encrypted": false,
  "status": "success",
  "error_message": null
}
```

## Restoration Procedures

### Database Restoration

⚠️ **WARNING**: Restoration will overwrite the current database!

1. **List available backups**:

```bash
python scripts/backup/restore_manager.py list --type database
```

2. **Choose backup to restore**:

```bash
python scripts/backup/restore_manager.py database --backup-id database_20241122_020000
```

3. **Confirm restoration**:
   - Review backup details
   - Type `yes` to confirm
   - Wait for completion

4. **Restart application**:

```bash
docker-compose restart app
# or
systemctl restart avalanche
```

### File Restoration

1. **List available backups**:

```bash
python scripts/backup/restore_manager.py list --type files
```

2. **Restore files**:

```bash
python scripts/backup/restore_manager.py files --backup-id files_20241122_030000
```

3. **Verify restoration**:

```bash
# Check restored directories
ls -la data/datasets
ls -la data/reference
```

### Full System Restoration

For complete disaster recovery:

1. **Fresh installation**:
   - Clone repository
   - Install dependencies
   - Setup Docker (if using)

2. **Download backups from cloud** (if needed):

```bash
# AWS S3
aws s3 cp s3://avalanche-backups/production/ data/backups/ --recursive

# Azure
az storage blob download-batch --source avalanche-backups --destination data/backups/

# GCS
gsutil -m cp -r gs://avalanche-backups/production/* data/backups/
```

3. **Restore database**:

```bash
python scripts/backup/restore_manager.py database --backup-id database_20241122_020000 --force
```

4. **Restore files**:

```bash
python scripts/backup/restore_manager.py files --backup-id files_20241122_030000 --force
```

5. **Start application**:

```bash
docker-compose up -d
```

## Backup Verification

### Manual Verification

```bash
# Verify specific backup
python scripts/backup/backup_manager.py verify --backup-id database_20241122_020000

# Verify all recent backups
for backup in $(python scripts/backup/backup_manager.py list | grep backup_id | cut -d: -f2 | tr -d ' '); do
    python scripts/backup/backup_manager.py verify --backup-id "$backup"
done
```

### Automated Verification

Backups are automatically verified after creation:
- Checksum calculation
- Metadata validation
- File integrity check

### Test Restoration

Periodically test restoration to ensure backups are functional:

```bash
# Create test database
createdb avalanche_test

# Restore to test database
DB_NAME=avalanche_test python scripts/backup/restore_manager.py database --backup-id database_20241122_020000 --force

# Verify data
psql avalanche_test -c "SELECT COUNT(*) FROM users;"

# Drop test database
dropdb avalanche_test
```

## Monitoring

### Backup Logs

Logs are stored in `data/backups/logs/`:

```bash
# View recent backup logs
tail -f data/backups/logs/backup_$(date +%Y%m%d).log

# View database backup logs
tail -f data/backups/logs/database_backup_*.log

# Search for errors
grep -i error data/backups/logs/*.log
```

### Backup Status

```bash
# List all backups with status
python scripts/backup/backup_manager.py list

# Check backup sizes
du -sh data/backups/database/*
du -sh data/backups/files/*

# Count backups
echo "Database backups: $(ls -1 data/backups/database/*_metadata.json 2>/dev/null | wc -l)"
echo "File backups: $(ls -1 data/backups/files/*_metadata.json 2>/dev/null | wc -l)"
```

### Email Notifications

Configure email notifications in `config/backup.yaml`:

```yaml
backup:
  notifications:
    enabled: true
    email:
      smtp_host: smtp.gmail.com
      smtp_port: 587
      smtp_user: your-email@gmail.com
      smtp_password: your-app-password
      from_email: avalanche-backup@example.com
      to_emails:
        - admin@example.com
        - ops@example.com
```

## Troubleshooting

### Backup Failures

#### PostgreSQL Connection Failed

```bash
# Test PostgreSQL connection
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT version();"

# Check PostgreSQL is running
docker-compose ps postgres

# Check credentials in .env
cat .env | grep DB_
```

#### Disk Space Issues

```bash
# Check available space
df -h data/backups/

# Check backup sizes
du -sh data/backups/*

# Run cleanup manually
python scripts/backup/backup_manager.py cleanup
```

#### Cloud Upload Failed

```bash
# Test AWS credentials
aws s3 ls s3://avalanche-backups/

# Test Azure connection
az storage container list --connection-string "$AZURE_STORAGE_CONNECTION_STRING"

# Test GCS credentials
gcloud auth list
gsutil ls gs://avalanche-backups/
```

### Restore Failures

#### Database Restore Failed

```bash
# Check backup file exists
ls -lh data/backups/database/database_20241122_020000*

# Verify backup integrity
python scripts/backup/backup_manager.py verify --backup-id database_20241122_020000

# Check PostgreSQL client tools
which pg_dump
which psql
```

#### File Restore Failed

```bash
# Check backup file
file data/backups/files/files_20241122_030000.tar.gz

# Test extraction manually
mkdir -p /tmp/test_restore
tar -tzf data/backups/files/files_20241122_030000.tar.gz | head
tar -xzf data/backups/files/files_20241122_030000.tar.gz -C /tmp/test_restore
```

## Best Practices

### Security

1. **Encrypt backups** for sensitive data
2. **Restrict access** to backup directories
3. **Use IAM roles** for cloud storage (instead of access keys)
4. **Rotate credentials** regularly
5. **Enable audit logging** for backup operations

### Performance

1. **Schedule backups** during low-traffic hours
2. **Use compression** to reduce storage costs
3. **Implement incremental backups** for large datasets
4. **Monitor backup duration** and optimize as needed

### Reliability

1. **Test restorations** quarterly
2. **Store backups off-site** (cloud storage)
3. **Maintain multiple backup copies** (3-2-1 rule)
4. **Document restoration procedures**
5. **Monitor backup success** with alerts

### 3-2-1 Backup Rule

- **3** copies of data (original + 2 backups)
- **2** different storage media (local + cloud)
- **1** off-site backup (cloud storage)

## Dependencies

Required Python packages:

```bash
# Core
pyyaml

# Cloud storage (optional)
boto3  # AWS S3
azure-storage-blob  # Azure Blob
google-cloud-storage  # Google Cloud Storage
```

Required system tools:

```bash
# PostgreSQL
sudo apt-get install postgresql-client  # Ubuntu/Debian
brew install postgresql  # macOS

# Compression
gzip (usually pre-installed)
tar (usually pre-installed)
```

## Support

For backup-related issues:

1. **Check logs**: `data/backups/logs/`
2. **Verify configuration**: `config/backup.yaml`
3. **Test connectivity**: Database and cloud storage
4. **Review documentation**: This guide
5. **Contact support**: Include log files and error messages

## Appendix

### Example Cron Configuration

Complete cron configuration for production:

```cron
# Avalanche Backup Schedule

# Database backup - Daily at 2 AM
0 2 * * * /opt/avalanche/scripts/backup/database_backup.sh >> /var/log/avalanche/cron.log 2>&1

# Files backup - Weekly on Sunday at 3 AM
0 3 * * 0 /opt/avalanche/scripts/backup/files_backup.sh >> /var/log/avalanche/cron.log 2>&1

# Full backup - Monthly on 1st at 4 AM
0 4 1 * * /opt/avalanche/scripts/backup/full_backup.sh >> /var/log/avalanche/cron.log 2>&1

# Cleanup - Weekly on Sunday at 5 AM
0 5 * * 0 cd /opt/avalanche && /usr/bin/python3 scripts/backup/backup_manager.py cleanup >> /var/log/avalanche/cron.log 2>&1

# Backup verification - Daily at 6 AM
0 6 * * * cd /opt/avalanche && /usr/bin/python3 scripts/backup/verify_recent_backups.py >> /var/log/avalanche/cron.log 2>&1
```

### Backup Size Estimates

Typical backup sizes:

| Component | Size (Uncompressed) | Size (Compressed) | Frequency |
|-----------|---------------------|-------------------|-----------|
| PostgreSQL DB | 100-500 MB | 20-100 MB | Daily |
| Datasets | 1-10 GB | 200 MB - 2 GB | Weekly |
| Reference Data | 500 MB - 2 GB | 100-400 MB | Weekly |
| Results | 1-5 GB | 200 MB - 1 GB | Weekly |
| Config | 1-5 MB | < 1 MB | Weekly |

Monthly storage requirement: ~50-200 GB (with retention policy)

### Recovery Time Objectives

| Scenario | RTO (Recovery Time Objective) | RPO (Recovery Point Objective) |
|----------|------------------------------|--------------------------------|
| Database corruption | < 1 hour | Last daily backup (24h) |
| File deletion | < 2 hours | Last weekly backup (7 days) |
| Server failure | < 4 hours | Last backup (24h) |
| Disaster recovery | < 8 hours | Last cloud backup (24h) |
