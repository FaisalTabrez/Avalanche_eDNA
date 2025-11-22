#!/bin/bash
# Full System Backup Script
# Run manually or scheduled monthly

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/data/backups/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_backup_$TIMESTAMP.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================="
log "Starting full system backup"
log "========================================="

# Change to project directory
cd "$PROJECT_ROOT"

# Load environment variables
if [ -f ".env" ]; then
    log "Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run full backup
log "Running full backup (database + files)..."
python3 scripts/backup/backup_manager.py full >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "Full backup completed successfully"
    
    # Get latest backups
    DB_BACKUP_ID=$(ls -t data/backups/database/*_metadata.json 2>/dev/null | head -1 | xargs basename | sed 's/_metadata.json//')
    FILES_BACKUP_ID=$(ls -t data/backups/files/*_metadata.json 2>/dev/null | head -1 | xargs basename | sed 's/_metadata.json//')
    
    # Verify database backup
    if [ -n "$DB_BACKUP_ID" ]; then
        log "Verifying database backup: $DB_BACKUP_ID"
        python3 scripts/backup/backup_manager.py verify --backup-id "$DB_BACKUP_ID" >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            log "Database backup verified"
            
            # Upload to cloud
            if [ "${BACKUP_CLOUD_ENABLED:-false}" = "true" ]; then
                log "Uploading database backup to cloud..."
                python3 scripts/backup/backup_manager.py upload --backup-id "$DB_BACKUP_ID" >> "$LOG_FILE" 2>&1
            fi
        else
            log "ERROR: Database backup verification failed!"
        fi
    fi
    
    # Verify files backup
    if [ -n "$FILES_BACKUP_ID" ]; then
        log "Verifying files backup: $FILES_BACKUP_ID"
        python3 scripts/backup/backup_manager.py verify --backup-id "$FILES_BACKUP_ID" >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            log "Files backup verified"
            
            # Upload to cloud
            if [ "${BACKUP_CLOUD_ENABLED:-false}" = "true" ]; then
                log "Uploading files backup to cloud..."
                python3 scripts/backup/backup_manager.py upload --backup-id "$FILES_BACKUP_ID" >> "$LOG_FILE" 2>&1
            fi
        else
            log "ERROR: Files backup verification failed!"
        fi
    fi
else
    log "ERROR: Full backup failed!"
    exit 1
fi

# Cleanup old backups
log "Running backup cleanup..."
python3 scripts/backup/backup_manager.py cleanup >> "$LOG_FILE" 2>&1

log "========================================="
log "Full backup process completed"
log "========================================="

exit 0
