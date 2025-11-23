# Independent User Management System

## Overview

The Avalanche eDNA platform now has a completely independent user management system that runs separately from the main application. This provides enhanced security and dedicated administration capabilities.

## Architecture

### Two Independent Applications

1. **Main Avalanche eDNA Application**
   - URL: http://localhost:8501
   - Purpose: eDNA sequence analysis and biodiversity assessment
   - Authentication: Required for all users
   - Access: All roles (admin, analyst, viewer)

2. **User Management Application**
   - URL: http://localhost:8502
   - Purpose: User administration and management
   - Authentication: Admin-only access
   - Access: Administrators only

### Shared Database

Both applications share the same SQLite user database (`data/avalanche_users.db`) via a shared Docker volume named `user-database`. This ensures:

- Single source of truth for user credentials
- Changes in user management immediately reflect in main app
- Consistent authentication across both systems
- Centralized audit logging

## Features

### User Management Application Features

1. **User List Management**
   - View all users with detailed information
   - Real-time metrics (total users, admins, analysts, active users)
   - Role indicators and status badges
   - Search and filter capabilities

2. **User Operations**
   - **Change Role**: Switch between admin, analyst, viewer
   - **Toggle Status**: Activate/deactivate user accounts
   - **Reset Password**: Admin-initiated password resets
   - **Delete User**: Remove users from the system (cannot delete self)

3. **User Creation**
   - Create new users with custom roles
   - Password strength validation
   - Email and username validation
   - Immediate activation

4. **Audit Log**
   - Complete audit trail of all user actions
   - Login/logout tracking
   - User modification history
   - IP address logging
   - Action timestamps

## Access

### Default Admin Credentials

```
Username: admin
Password: Admin@123
```

**⚠️ IMPORTANT**: Change the default password immediately after first login!

### Security Features

- Admin-only authentication
- Session-based security
- Password strength requirements
- Audit trail for all actions
- Independent from main application

## Docker Deployment

### Service Configuration

```yaml
user-management:
  image: avalanche-edna:application
  container_name: avalanche-user-management
  ports:
    - "8502:8502"
  volumes:
    - user-database:/app/data  # Shared with main app
```

### Starting the Service

```bash
# Start only user management
cd docker
docker-compose up -d user-management

# Start both services
docker-compose up -d streamlit user-management

# Start all services
docker-compose up -d
```

### Stopping the Service

```bash
# Stop user management
docker-compose stop user-management

# Remove user management container
docker-compose down user-management
```

### Viewing Logs

```bash
# Real-time logs
docker logs -f avalanche-user-management

# Last 50 lines
docker logs avalanche-user-management --tail 50
```

## Usage Guide

### Accessing User Management

1. Navigate to http://localhost:8502
2. Login with admin credentials
3. Select the desired tab:
   - **User List**: View and manage existing users
   - **Create User**: Add new users
   - **Audit Log**: Review system activity

### Managing Users

#### Change User Role

1. Go to **User List** tab
2. Expand the user card
3. Select new role from dropdown
4. Click **Update Role**

#### Reset User Password

1. Go to **User List** tab
2. Expand the user card
3. Click **Reset Password**
4. Enter new password meeting requirements
5. Click **Confirm Reset**

#### Deactivate User

1. Go to **User List** tab
2. Expand the user card
3. Uncheck **Active** checkbox
4. Click **Update Status**

#### Delete User

1. Go to **User List** tab
2. Expand the user card
3. Click **Delete User** (twice to confirm)

**Note**: Cannot delete yourself while logged in

### Creating New Users

1. Go to **Create User** tab
2. Fill in required fields:
   - Username (unique)
   - Email (valid format)
   - Password (must meet requirements)
   - Confirm Password (must match)
   - Role (admin/analyst/viewer)
3. Click **Create User**

#### Password Requirements

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character

### Reviewing Audit Log

1. Go to **Audit Log** tab
2. View recent actions (last 100 entries)
3. Expand entries for details:
   - Action type
   - User who performed action
   - Timestamp
   - IP address
   - Additional details

## Roles and Permissions

### Admin
- **Permissions**: Full system access
- **Capabilities**:
  - All analyst capabilities
  - User management
  - System configuration
  - Access to both applications

### Analyst
- **Permissions**: Read, write, delete own
- **Capabilities**:
  - Upload sequences
  - Run analyses
  - Create reports
  - Delete own datasets
  - Access main application only

### Viewer
- **Permissions**: Read-only
- **Capabilities**:
  - View existing analyses
  - Download reports
  - Browse datasets
  - Access main application only

## Troubleshooting

### Cannot Access User Management

**Problem**: http://localhost:8502 not responding

**Solutions**:
```bash
# Check if service is running
docker ps | grep user-management

# Restart service
docker-compose restart user-management

# Check logs for errors
docker logs avalanche-user-management
```

### Login Failed

**Problem**: Invalid credentials error

**Solutions**:
1. Verify you're using admin credentials
2. Check if account is active
3. Ensure database is accessible
4. Review audit log for account status

### Changes Not Reflected in Main App

**Problem**: User changes don't appear in main application

**Solutions**:
```bash
# Restart main application
docker-compose restart streamlit

# Verify shared volume
docker volume inspect docker_user-database

# Check both containers use same volume
docker inspect avalanche-streamlit | grep user-database
docker inspect avalanche-user-management | grep user-database
```

### Database Locked Error

**Problem**: "Database is locked" error

**Solutions**:
1. SQLite can handle concurrent reads but limited writes
2. Retry the operation
3. If persistent, restart both services:
```bash
docker-compose restart streamlit user-management
```

## Best Practices

1. **Security**
   - Change default admin password immediately
   - Use strong passwords for all accounts
   - Regularly review audit logs
   - Deactivate unused accounts

2. **User Management**
   - Create users with least privilege
   - Upgrade to analyst/admin only when needed
   - Use descriptive usernames
   - Keep email addresses current

3. **Monitoring**
   - Review audit log weekly
   - Monitor failed login attempts
   - Track user activity patterns
   - Archive old audit entries

4. **Backup**
   - Regularly backup user database
   - Store backups securely
   - Test restoration procedures

## Database Location

### In Docker Container
```
/app/data/avalanche_users.db
```

### Docker Volume
```
docker_user-database
```

### Backup Command
```bash
# Export database
docker exec avalanche-user-management sqlite3 /app/data/avalanche_users.db ".backup /app/data/backup_$(date +%Y%m%d).db"

# Copy to host
docker cp avalanche-user-management:/app/data/backup_20250123.db ./backups/
```

## Integration with Main Application

The user management system is completely independent but shares the user database. This means:

### ✅ What Works

- Users created in user management can immediately login to main app
- Password resets take effect immediately
- Role changes affect main app permissions instantly
- Deactivated users cannot login to main app
- Audit log tracks actions from both applications

### ❌ What Doesn't Work

- User management cannot access main app features (analysis, reports, etc.)
- Main app users cannot access user management (admin role required)
- Sessions are independent (login to both separately)

## Port Configuration

| Service | Port | URL |
|---------|------|-----|
| Main Application | 8501 | http://localhost:8501 |
| User Management | 8502 | http://localhost:8502 |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |

## Environment Variables

User management uses minimal environment variables:

```yaml
PYTHONPATH: /app
ENVIRONMENT: production
LOG_LEVEL: INFO
```

No database credentials needed (uses SQLite file directly).

## Future Enhancements

Planned features for user management:

- [ ] Bulk user import/export
- [ ] Password expiration policies
- [ ] Two-factor authentication
- [ ] Session management (view active sessions)
- [ ] Advanced audit log filtering
- [ ] User groups and teams
- [ ] API access for automation
- [ ] Email notifications for password resets

## Support

For issues or questions:

1. Check logs: `docker logs avalanche-user-management`
2. Review audit log in the application
3. Verify database accessibility
4. Consult main documentation: `README.md`

---

**Last Updated**: 2025-11-23
**Version**: 1.0.0
**Application**: Avalanche eDNA Platform
