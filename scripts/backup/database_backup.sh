#!/bin/bash
# Automated Database Backup Script
# Scheduled to run daily at 2 AM via cron

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/data/backups/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/database_backup_$TIMESTAMP.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================="
log "Starting automated database backup"
log "========================================="

# Change to project directory
cd "$PROJECT_ROOT"

# Load environment variables
if [ -f ".env" ]; then
    log "Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run backup
log "Running database backup..."
python3 scripts/backup/backup_manager.py database >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "Database backup completed successfully"
    
    # Get latest backup ID
    BACKUP_ID=$(ls -t data/backups/database/*_metadata.json 2>/dev/null | head -1 | xargs basename | sed 's/_metadata.json//')
    
    if [ -n "$BACKUP_ID" ]; then
        log "Latest backup ID: $BACKUP_ID"
        
        # Verify backup
        log "Verifying backup integrity..."
        python3 scripts/backup/backup_manager.py verify --backup-id "$BACKUP_ID" >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            log "Backup verification successful"
            
            # Upload to cloud (if enabled)
            if [ "${BACKUP_CLOUD_ENABLED:-false}" = "true" ]; then
                log "Uploading backup to cloud storage..."
                python3 scripts/backup/backup_manager.py upload --backup-id "$BACKUP_ID" >> "$LOG_FILE" 2>&1
                
                if [ $? -eq 0 ]; then
                    log "Cloud upload successful"
                else
                    log "WARNING: Cloud upload failed"
                fi
            fi
        else
            log "ERROR: Backup verification failed!"
            exit 1
        fi
    fi
else
    log "ERROR: Database backup failed!"
    exit 1
fi

# Cleanup old backups
log "Running backup cleanup..."
python3 scripts/backup/backup_manager.py cleanup >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "Backup cleanup completed"
else
    log "WARNING: Backup cleanup failed"
fi

log "========================================="
log "Database backup process completed"
log "========================================="

exit 0
