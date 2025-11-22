#!/usr/bin/env python3
"""
Verify Recent Backups - Check integrity of recent backups
Run daily to ensure backups are valid
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path is set
sys.path.insert(0, str(Path(__file__).parent))
from backup_manager import BackupManager


def verify_recent_backups(days: int = 7):
    """Verify backups created in the last N days
    
    Args:
        days: Number of days to look back
    """
    manager = BackupManager()
    
    # Get all backups
    all_backups = manager.list_backups()
    
    # Filter to recent backups
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_backups = [
        b for b in all_backups
        if datetime.fromisoformat(b.timestamp) > cutoff_date
    ]
    
    print(f"\n{'='*60}")
    print(f"Verifying {len(recent_backups)} backups from last {days} days")
    print(f"{'='*60}\n")
    
    failed_backups = []
    
    for backup in recent_backups:
        print(f"Checking: {backup.backup_id}")
        print(f"  Type: {backup.backup_type}")
        print(f"  Date: {backup.timestamp}")
        print(f"  Size: {backup.size_bytes / (1024**2):.2f} MB")
        
        # Verify backup
        is_valid = manager.verify_backup(backup.backup_id)
        
        if is_valid:
            print(f"  Status: ✓ VALID")
        else:
            print(f"  Status: ✗ INVALID")
            failed_backups.append(backup)
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print(f"Verification Summary:")
    print(f"  Total backups checked: {len(recent_backups)}")
    print(f"  Valid: {len(recent_backups) - len(failed_backups)}")
    print(f"  Invalid: {len(failed_backups)}")
    print(f"{'='*60}\n")
    
    if failed_backups:
        print("⚠️  WARNING: The following backups failed verification:")
        for backup in failed_backups:
            print(f"  - {backup.backup_id}")
        print()
        return False
    else:
        print("✓ All recent backups are valid")
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify recent backups')
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to look back (default: 7)'
    )
    
    args = parser.parse_args()
    
    success = verify_recent_backups(days=args.days)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
