# Authentication System Documentation

## Overview

The Avalanche eDNA Biodiversity Assessment System includes a comprehensive authentication and authorization system to secure access to analysis features and manage users.

## Features

- **User Authentication**: Secure login/logout with session management
- **Role-Based Access Control (RBAC)**: Three user roles with different permissions
- **Password Security**: PBKDF2-SHA256 hashing with strong password requirements
- **Session Management**: Automatic session timeout and renewal
- **Account Lockout**: Protection against brute-force attacks (5 failed attempts → 15-minute lockout)
- **Audit Logging**: Track all user actions for security and compliance
- **User Management**: Admin interface for user CRUD operations

## User Roles

### Admin
**Permissions**: `read`, `write`, `delete`, `manage_users`, `manage_system`

- Full system access
- Can create, update, and delete users
- Can change user roles and permissions
- Access to audit logs and system configuration

### Analyst
**Permissions**: `read`, `write`, `delete_own`

- Can create and analyze datasets
- Can upload and process sequences
- Can view and download results
- Can delete their own data
- Cannot manage users or system settings

### Viewer
**Permissions**: `read`

- Read-only access to the system
- Can browse existing results
- Can view taxonomic data
- Cannot create, modify, or delete data

## Getting Started

### First-Time Setup

1. **Start the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Default admin account**:
   On first run, a default admin account is automatically created:
   - **Username**: `admin`
   - **Password**: `Admin@123`
   - **⚠️ IMPORTANT**: Change this password immediately after first login!

3. **Login**:
   - Navigate to the "Login" page
   - Enter the admin credentials
   - Click "Login"

4. **Change default password**:
   - Go to User Management (admin only)
   - Click "Reset Password" for the admin user
   - Enter a strong new password

### Creating Users

**Admin only**

1. Navigate to **User Management** page
2. Go to **Create User** tab
3. Fill in user details:
   - Username (unique)
   - Email (unique)
   - Password (must meet strength requirements)
   - Role (admin, analyst, viewer)
4. Click **Create User**

### Password Requirements

All passwords must meet these criteria:
- At least 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character: `!@#$%^&*(),.?":{}|<>`
- Not a common weak password

## Usage

### For Users

#### Login
1. Navigate to **Login** page
2. Enter username and password
3. Click **Login**

#### Logout
1. Click **Logout** button in the sidebar

#### Change Password
1. Go to **User Management** (if admin) or contact an admin
2. Request password reset
3. Enter new password meeting strength requirements

### For Administrators

#### User Management
- **View Users**: Browse all registered users with their roles and status
- **Create Users**: Add new users with specific roles
- **Update Users**: Change user roles, email, or active status
- **Reset Passwords**: Force password reset for any user
- **Delete Users**: Remove user accounts (cannot delete yourself)
- **View Audit Log**: Monitor all user actions

#### Managing Roles
1. Go to **User Management** → **User List**
2. Find the user to modify
3. Select new role from dropdown
4. Click **Update Role**

#### Deactivating Accounts
1. Go to **User Management** → **User List**
2. Find the user to deactivate
3. Uncheck **Active** checkbox
4. Click **Update Status**

## Security Features

### Password Hashing
- Algorithm: PBKDF2-SHA256
- Iterations: 100,000 (OWASP recommended)
- Salt: 32-byte random salt per password
- Constant-time comparison to prevent timing attacks

### Session Security
- Session timeout: 1 hour (configurable)
- Automatic session invalidation on logout
- Session token: 32-byte cryptographically secure random token
- Periodic cleanup of expired sessions

### Brute-Force Protection
- Failed login attempts tracked per user
- Account locked for 15 minutes after 5 failed attempts
- Automatic unlock after timeout
- Admin can manually reset lockout

### Audit Logging
All security-relevant actions are logged:
- User creation, updates, deletion
- Login successes and failures
- Password changes
- Role changes
- Permission checks

View audit logs: **User Management** → **Audit Log** tab

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login TEXT,
    is_active INTEGER DEFAULT 1,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TEXT
)
```

### Sessions Table
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
)
```

### Audit Log Table
```sql
CREATE TABLE audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,
    resource TEXT,
    details TEXT,
    ip_address TEXT
)
```

## API Usage (For Developers)

### Basic Authentication

```python
from src.auth import get_auth_manager

# Get authentication manager
auth = get_auth_manager()

# Login
success, message = auth.login("username", "password")

# Check if authenticated
if auth.is_authenticated():
    user = auth.get_current_user()
    print(f"Logged in as {user['username']}")

# Logout
auth.logout()
```

### Decorators for Access Control

```python
from src.auth import require_auth, require_role, require_permission

# Require authentication
@require_auth()
def protected_function():
    # Only accessible to authenticated users
    pass

# Require specific role
@require_role('admin')
def admin_function():
    # Only accessible to admins
    pass

# Require specific permission
@require_permission('write')
def write_function():
    # Only accessible to users with write permission
    pass

# Require any of multiple roles
from src.auth.decorators import require_any_role

@require_any_role('admin', 'analyst')
def analyst_or_admin_function():
    # Accessible to admins or analysts
    pass
```

### Manual Permission Checks

```python
from src.auth import get_auth_manager

auth = get_auth_manager()

# Check role
if auth.has_role('admin'):
    # Admin-only code
    pass

# Check permission
if auth.has_permission('write'):
    # Write-enabled code
    pass
```

### User Management

```python
from src.auth import get_auth_manager

auth = get_auth_manager()

# Create user
success, user_id = auth.user_manager.create_user(
    username="newuser",
    email="user@example.com",
    password="SecurePass123!",
    role="analyst"
)

# List users
users = auth.user_manager.list_users()

# Update user
success, message = auth.user_manager.update_user(
    user_id="user-id",
    role="admin",
    is_active=True
)

# Delete user
success, message = auth.user_manager.delete_user("user-id")

# Change password
success, message = auth.user_manager.change_password(
    user_id="user-id",
    new_password="NewSecurePass456!"
)
```

## Configuration

### Session Timeout

Edit `src/auth/authenticator.py`:

```python
# Default: 3600 seconds (1 hour)
auth = AuthManager(session_timeout=7200)  # 2 hours
```

### Password Iterations

Edit `src/auth/password_utils.py`:

```python
class PasswordHasher:
    ITERATIONS = 100000  # Increase for more security (slower)
```

### Database Location

Edit `src/auth/user_manager.py`:

```python
# Default: data/avalanche_users.db
user_manager = UserManager(db_path="custom/path/users.db")
```

## Troubleshooting

### Cannot Login - Account Locked
**Cause**: Too many failed login attempts

**Solution**:
1. Wait 15 minutes for automatic unlock
2. Or contact admin to reset your account

### Forgot Password
**Cause**: Password not remembered

**Solution**:
1. Contact system administrator
2. Admin can reset your password via User Management

### Session Expired
**Cause**: Session timeout (default 1 hour of inactivity)

**Solution**:
1. Log in again
2. Sessions auto-renew on activity

### Cannot Access Feature
**Cause**: Insufficient permissions

**Solution**:
1. Check your role in sidebar
2. Contact admin to upgrade role if needed

### Default Admin Not Working
**Cause**: Admin already created or database issue

**Solution**:
1. Check for existing users in User Management
2. If database corrupted, delete `data/avalanche_users.db` and restart

## Security Best Practices

### For Administrators

1. **Change default password immediately** after first login
2. **Use strong passwords** (16+ characters with complexity)
3. **Review audit logs regularly** for suspicious activity
4. **Deactivate unused accounts** to reduce attack surface
5. **Grant minimum required roles** (principle of least privilege)
6. **Backup user database** regularly (`data/avalanche_users.db`)
7. **Monitor failed login attempts** in audit logs

### For Users

1. **Never share passwords** with anyone
2. **Use unique passwords** not used elsewhere
3. **Log out after use** especially on shared computers
4. **Report suspicious activity** to administrators
5. **Change password periodically** (every 90 days recommended)

## Future Enhancements

Planned improvements (see DEPLOYMENT_ROADMAP.md):

- [ ] Multi-factor authentication (MFA/2FA)
- [ ] OAuth/SSO integration (Google, GitHub, Azure AD)
- [ ] Password reset via email
- [ ] User profile management
- [ ] API keys for programmatic access
- [ ] IP-based access control
- [ ] Advanced audit log filtering
- [ ] Rate limiting per user
- [ ] Session management dashboard

## Support

For issues or questions:
1. Check this documentation
2. Review audit logs (admins only)
3. Contact system administrator
4. File issue on GitHub repository
