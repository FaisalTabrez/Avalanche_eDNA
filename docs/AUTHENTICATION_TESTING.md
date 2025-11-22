# Testing the Authentication System

## Quick Test

1. **Start the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Check for default admin creation**:
   - Look for success message in sidebar: "🔑 First-time setup complete!"
   - Default credentials should be displayed

3. **Test login**:
   - Navigate to Login page
   - Username: `admin`
   - Password: `Admin@123`
   - Click "Login"
   - Should see welcome message and user info in sidebar

4. **Test navigation**:
   - Verify you can access all pages
   - Check that "User Management" appears (admin only)

5. **Test user management** (admin):
   - Go to User Management
   - Create a test user (analyst role)
   - Verify user appears in User List
   - Update user role to viewer
   - Test password reset

6. **Test logout**:
   - Click "Logout" in sidebar
   - Verify you're logged out
   - Verify restricted pages show login prompt

7. **Test new user login**:
   - Login with the test user credentials
   - Verify appropriate pages are visible
   - Verify User Management is NOT visible (not admin)

## Comprehensive Testing Checklist

### Authentication Tests

- [ ] First-time setup creates default admin
- [ ] Login with correct credentials succeeds
- [ ] Login with incorrect credentials fails
- [ ] Login with non-existent user fails
- [ ] Logout clears session
- [ ] Session persists across page navigation
- [ ] Session expires after timeout (wait 1 hour or modify timeout)

### Registration Tests

- [ ] Registration with valid data succeeds
- [ ] Registration with weak password fails
- [ ] Registration with mismatched passwords fails
- [ ] Registration with duplicate username fails
- [ ] Registration with duplicate email fails
- [ ] New user can login after registration

### Password Security Tests

- [ ] Password < 8 characters rejected
- [ ] Password without uppercase rejected
- [ ] Password without lowercase rejected
- [ ] Password without digit rejected
- [ ] Password without special char rejected
- [ ] Common passwords (password, 12345678) rejected

### Brute-Force Protection Tests

- [ ] Failed login increments attempt counter
- [ ] 5 failed logins lock account
- [ ] Locked account shows appropriate message
- [ ] Account auto-unlocks after 15 minutes
- [ ] Successful login resets failed attempts

### Role-Based Access Tests

#### Admin Role
- [ ] Can access all pages
- [ ] Can see User Management
- [ ] Can create users
- [ ] Can update user roles
- [ ] Can delete users (except self)
- [ ] Can reset passwords
- [ ] Can view audit logs

#### Analyst Role
- [ ] Can access analysis features
- [ ] Cannot access User Management
- [ ] Can create datasets
- [ ] Can view/download results
- [ ] Can upload data

#### Viewer Role
- [ ] Can view pages
- [ ] Cannot access User Management
- [ ] Cannot create/modify data
- [ ] Can browse existing results

### User Management Tests

- [ ] User list displays all users
- [ ] User creation works with all roles
- [ ] User update (role) works
- [ ] User update (status) works
- [ ] User update (email) works
- [ ] Password reset works
- [ ] User deletion works
- [ ] Cannot delete self
- [ ] Audit log displays actions

### Session Management Tests

- [ ] Session created on login
- [ ] Session validated on each request
- [ ] Expired session redirects to login
- [ ] Multiple sessions for same user
- [ ] Session cleanup removes expired sessions

### UI/UX Tests

- [ ] Login page displays correctly
- [ ] Registration tab works
- [ ] Password requirements shown
- [ ] Error messages are clear
- [ ] Success messages are clear
- [ ] User info appears in sidebar
- [ ] Logout button works
- [ ] Page access controlled by auth status

### Database Tests

- [ ] User database created on first run
- [ ] Users table has correct schema
- [ ] Sessions table has correct schema
- [ ] Audit log table has correct schema
- [ ] Constraints enforce uniqueness
- [ ] Foreign keys work correctly

### Audit Logging Tests

- [ ] User creation logged
- [ ] Login success logged
- [ ] Login failure logged
- [ ] Password change logged
- [ ] User update logged
- [ ] User deletion logged
- [ ] Audit log searchable/filterable

## Manual Test Script

```bash
# 1. Clean start (delete existing database)
rm data/avalanche_users.db

# 2. Start application
streamlit run streamlit_app.py

# 3. Verify default admin creation
# - Check sidebar for admin credentials message

# 4. Login as admin
# - Navigate to Login page
# - Enter: admin / Admin@123
# - Verify login success

# 5. Create test users
# - Go to User Management → Create User
# - Create analyst: test_analyst / Test@Analyst123 / analyst
# - Create viewer: test_viewer / Test@Viewer123 / viewer

# 6. Test analyst permissions
# - Logout
# - Login as test_analyst
# - Verify can access analysis pages
# - Verify cannot access User Management

# 7. Test viewer permissions
# - Logout
# - Login as test_viewer
# - Verify can only view data
# - Verify cannot access User Management

# 8. Test admin features
# - Logout
# - Login as admin
# - Go to User Management
# - Change test_analyst role to viewer
# - Verify role updated
# - Reset test_viewer password
# - Verify can login with new password
# - Delete test_analyst
# - Verify user removed

# 9. Test brute-force protection
# - Logout
# - Try login with wrong password 5 times
# - Verify account locked
# - Wait 15 minutes or modify timeout in code
# - Verify can login again

# 10. Test session expiration
# - Login
# - Wait for session timeout (1 hour or modify)
# - Try to navigate
# - Verify redirected to login
```

## Expected Behaviors

### Successful Login
```
✅ Welcome, admin!
👤 admin (admin) appears in sidebar
Navigation shows all pages including User Management
```

### Failed Login
```
❌ Invalid username or password. Account will be locked after 5 failed attempts.
User not logged in
Limited page navigation
```

### Account Locked
```
❌ Invalid username or password. Account will be locked after 5 failed attempts.
(Even with correct password, login fails until timeout)
```

### Session Expired
```
⚠️ Please log in to access this page.
Redirected to login page
```

### Insufficient Permissions
```
🚫 Access denied. This page requires admin role.
Redirected to home page
```

## Database Inspection

Check database contents:

```bash
sqlite3 data/avalanche_users.db
```

```sql
-- View all users
SELECT user_id, username, email, role, is_active, failed_login_attempts 
FROM users;

-- View sessions
SELECT session_id, user_id, created_at, last_activity 
FROM sessions;

-- View recent audit log
SELECT timestamp, username, action, details 
FROM audit_log 
ORDER BY timestamp DESC 
LIMIT 10;

-- Check for locked accounts
SELECT username, locked_until, failed_login_attempts 
FROM users 
WHERE locked_until IS NOT NULL;
```

## Performance Testing

### Load Test (create many users)

```python
from src.auth import get_auth_manager

auth = get_auth_manager()

# Create 100 test users
for i in range(100):
    auth.user_manager.create_user(
        username=f"testuser{i}",
        email=f"testuser{i}@example.com",
        password="TestPass123!",
        role="viewer"
    )

# List all users (should be fast)
users = auth.user_manager.list_users()
print(f"Created {len(users)} users")
```

### Session Cleanup Performance

```python
from src.auth.password_utils import SessionManager

sm = SessionManager(timeout_seconds=1)

# Create many expired sessions
for i in range(1000):
    sm.create_session(f"user{i}", f"username{i}", "viewer")

# Wait for expiration
import time
time.sleep(2)

# Cleanup (should be fast)
sm.cleanup_expired_sessions()
print(f"Active sessions: {len(sm.sessions)}")
```

## Common Issues and Solutions

### Issue: Cannot login with default admin
**Solution**: 
1. Check if database exists: `ls data/avalanche_users.db`
2. Delete database and restart: `rm data/avalanche_users.db && streamlit run streamlit_app.py`
3. Check sidebar for admin credentials

### Issue: Session expired immediately
**Solution**:
1. Check session timeout setting in `src/auth/authenticator.py`
2. Increase timeout: `AuthManager(session_timeout=7200)`

### Issue: Cannot access pages after login
**Solution**:
1. Check sidebar for user info
2. Verify `st.session_state.authenticated` is True
3. Check browser console for errors
4. Clear browser cache and reload

### Issue: User Management not showing
**Solution**:
1. Verify logged in as admin
2. Check sidebar shows "admin (admin)"
3. Check router.py includes user_management

### Issue: Database locked
**Solution**:
1. Close all connections to database
2. Restart Streamlit
3. If persists: `rm data/avalanche_users.db` and recreate

## Integration Testing

Test with actual analysis workflows:

1. **As Analyst**:
   - Login as analyst
   - Go to Analysis page
   - Upload FASTA file
   - Run analysis
   - View results
   - Verify results saved under analyst's user_id

2. **As Viewer**:
   - Login as viewer
   - Browse Biodiversity Results
   - Attempt to upload (should fail)
   - Verify can view but not modify

3. **As Admin**:
   - Login as admin
   - View all users' results
   - Manage users while analysis running
   - Check audit log for analysis actions

## Automated Testing (Future)

Create `tests/test_auth.py`:

```python
import pytest
from src.auth import get_auth_manager

def test_login_success():
    auth = get_auth_manager()
    success, msg = auth.login("admin", "Admin@123")
    assert success == True

def test_login_failure():
    auth = get_auth_manager()
    success, msg = auth.login("admin", "wrongpassword")
    assert success == False

def test_create_user():
    auth = get_auth_manager()
    success, user_id = auth.user_manager.create_user(
        username="testuser",
        email="test@example.com",
        password="TestPass123!",
        role="viewer"
    )
    assert success == True

# Add more tests...
```

Run tests:
```bash
pytest tests/test_auth.py -v
```

## Success Criteria

Phase 1.2 is complete when:

- [x] All authentication components created
- [x] Login/logout works correctly
- [x] User registration works
- [x] Role-based access control enforced
- [x] Password security requirements met
- [x] Brute-force protection active
- [x] Session management functional
- [x] User management UI operational
- [x] Audit logging captures actions
- [x] Documentation complete
- [x] Manual testing passed
- [ ] No security vulnerabilities
- [ ] Performance acceptable

**Status**: ✅ **READY FOR TESTING**
