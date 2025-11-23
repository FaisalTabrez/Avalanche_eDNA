"""
Standalone User Management Application
Independent Streamlit app for managing Avalanche eDNA users
Using PostgreSQL and Redis for optimal performance
"""
import streamlit as st
from pathlib import Path
from src.auth.postgres_user_manager import PostgresUserManager
from src.auth.password_utils import validate_password_strength, PasswordHasher


# Page configuration
st.set_page_config(
    page_title="Avalanche User Management",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def init_session_state():
    """Initialize session state variables"""
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    if 'admin_username' not in st.session_state:
        st.session_state.admin_username = None


def admin_login():
    """Simple admin login for user management access"""
    st.title("🔐 Avalanche User Management")
    st.markdown("---")
    
    st.info("This is an independent user management system. Login with admin credentials.")
    
    with st.form("admin_login"):
        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("Please provide both username and password")
                return
            
            # Initialize PostgreSQL user manager
            try:
                user_manager = PostgresUserManager()
            except Exception as e:
                st.error(f"Failed to connect to database: {e}")
                return
            
            # Authenticate
            success, result = user_manager.authenticate(username, password)
            
            if success and result.get('role') == 'admin':
                st.session_state.admin_authenticated = True
                st.session_state.admin_username = username
                st.session_state.admin_user_id = result.get('user_id')
                st.rerun()
            elif success:
                st.error("Access denied. Only administrators can access this system.")
            else:
                st.error("Invalid credentials")


def logout():
    """Logout admin user"""
    st.session_state.admin_authenticated = False
    st.session_state.admin_username = None
    st.session_state.admin_user_id = None
    st.rerun()


def render_user_list(user_manager):
    """Render list of all users"""
    st.subheader("👥 All Users")
    
    users = user_manager.list_users()
    
    if not users:
        st.info("No users found in the system")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(users))
    with col2:
        admin_count = sum(1 for u in users if u['role'] == 'admin')
        st.metric("Admins", admin_count)
    with col3:
        analyst_count = sum(1 for u in users if u['role'] == 'analyst')
        st.metric("Analysts", analyst_count)
    with col4:
        active_count = sum(1 for u in users if u['is_active'])
        st.metric("Active Users", active_count)
    
    st.markdown("---")
    
    # Display users in expandable cards
    for user in users:
        status_emoji = "✅" if user['is_active'] else "❌"
        role_emoji = "👑" if user['role'] == 'admin' else ("🔬" if user['role'] == 'analyst' else "👁️")
        
        with st.expander(f"{role_emoji} {user['username']} ({user['role']}) {status_emoji}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**User ID:** `{user['user_id']}`")
                st.write(f"**Username:** {user['username']}")
                st.write(f"**Email:** {user['email']}")
            
            with col2:
                st.write(f"**Role:** {user['role']}")
                st.write(f"**Status:** {'🟢 Active' if user['is_active'] else '🔴 Inactive'}")
                st.write(f"**Created:** {user['created_at'][:10]}")
            
            if user['last_login']:
                st.write(f"**Last Login:** {user['last_login']}")
            
            # Actions
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Change role
                st.write("**Change Role**")
                new_role = st.selectbox(
                    "Select new role",
                    options=['admin', 'analyst', 'viewer'],
                    index=['admin', 'analyst', 'viewer'].index(user['role']),
                    key=f"role_{user['user_id']}",
                    label_visibility="collapsed"
                )
                
                if new_role != user['role']:
                    if st.button("Update Role", key=f"update_role_{user['user_id']}"):
                        success, message = user_manager.update_user(
                            user['user_id'],
                            role=new_role
                        )
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            with col2:
                # Toggle active status
                st.write("**Active Status**")
                new_status = st.checkbox(
                    "Active",
                    value=bool(user['is_active']),
                    key=f"active_{user['user_id']}"
                )
                
                if new_status != bool(user['is_active']):
                    if st.button("Update Status", key=f"update_status_{user['user_id']}"):
                        success, message = user_manager.update_user(
                            user['user_id'],
                            is_active=new_status
                        )
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            with col3:
                # Reset password
                st.write("**Reset Password**")
                if st.button("Reset Password", key=f"reset_{user['user_id']}"):
                    st.session_state[f'reset_password_{user["user_id"]}'] = True
                
                if st.session_state.get(f'reset_password_{user["user_id"]}', False):
                    new_password = st.text_input(
                        "New Password",
                        type="password",
                        key=f"new_pwd_{user['user_id']}"
                    )
                    
                    if new_password:
                        is_valid, error_msg = validate_password_strength(new_password)
                        
                        if not is_valid:
                            st.error(f"❌ {error_msg}")
                        elif st.button("Confirm Reset", key=f"confirm_reset_{user['user_id']}"):
                            success, message = user_manager.change_password(
                                user['user_id'],
                                new_password
                            )
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.session_state[f'reset_password_{user["user_id"]}'] = False
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
            
            with col4:
                # Delete user
                st.write("**Delete User**")
                if user['user_id'] != st.session_state.get('admin_user_id'):
                    if st.button("Delete User", key=f"delete_{user['user_id']}", type="primary"):
                        if st.session_state.get(f'confirm_delete_{user["user_id"]}', False):
                            success, message = user_manager.delete_user(user['user_id'])
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.session_state[f'confirm_delete_{user["user_id"]}'] = False
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.session_state[f'confirm_delete_{user["user_id"]}'] = True
                            st.warning("⚠️ Click again to confirm deletion")
                else:
                    st.info("Cannot delete yourself")


def render_create_user(user_manager):
    """Render form to create a new user"""
    st.subheader("➕ Create New User")
    
    with st.form("create_user_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        
        col1, col2 = st.columns(2)
        
        with col1:
            password = st.text_input("Password", type="password")
        
        with col2:
            confirm_password = st.text_input("Confirm Password", type="password")
        
        role = st.selectbox("Role", options=['viewer', 'analyst', 'admin'])
        
        st.caption("**Password requirements:**")
        st.caption("• At least 8 characters")
        st.caption("• At least one uppercase, lowercase, digit, and special character")
        
        submit = st.form_submit_button("Create User", use_container_width=True, type="primary")
        
        if submit:
            if not username or not email or not password or not confirm_password:
                st.error("❌ Please fill in all fields")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                # Validate password strength
                is_valid, error_msg = validate_password_strength(password)
                
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                else:
                    success, result = user_manager.create_user(
                        username=username,
                        email=email,
                        password=password,
                        role=role
                    )
                    
                    if success:
                        st.success(f"✅ User created successfully with ID: {result}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")


def render_audit_log(user_manager):
    """Render audit log"""
    st.subheader("📋 Audit Log")
    
    # Get recent audit entries
    conn = user_manager._get_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                al.timestamp,
                u.username,
                al.action,
                al.details,
                al.ip_address
            FROM audit_log al
            LEFT JOIN users u ON al.user_id = u.user_id
            ORDER BY al.timestamp DESC
            LIMIT 100
        """)
        
        entries = cursor.fetchall()
    finally:
        user_manager._return_connection(conn)
    
    if not entries:
        st.info("No audit log entries found")
        return
    
    # Display in table format
    st.write(f"Showing last {len(entries)} entries")
    
    for entry in entries:
        timestamp, username, action, details, ip = entry
        
        # Color code by action type
        if 'login' in action.lower():
            badge_color = "🟢"
        elif 'logout' in action.lower():
            badge_color = "🔵"
        elif 'delete' in action.lower() or 'fail' in action.lower():
            badge_color = "🔴"
        else:
            badge_color = "🟡"
        
        with st.expander(f"{badge_color} {timestamp} - {username or 'Unknown'} - {action}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Action:** {action}")
                st.write(f"**User:** {username or 'Unknown'}")
            with col2:
                st.write(f"**Time:** {timestamp}")
                st.write(f"**IP:** {ip or 'N/A'}")
            
            if details:
                st.write(f"**Details:** {details}")


def main_app():
    """Main user management application"""
    st.title("👥 Avalanche User Management System")
    
    # Sidebar
    with st.sidebar:
        st.write(f"**Logged in as:** {st.session_state.admin_username}")
        st.write(f"**Role:** Administrator")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        st.markdown("---")
        st.caption("Independent User Management")
        st.caption("PostgreSQL + Redis")
        st.caption("Avalanche eDNA Platform")
    
    # Initialize PostgreSQL user manager
    try:
        user_manager = PostgresUserManager()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info("Please ensure PostgreSQL and Redis services are running.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["👥 User List", "➕ Create User", "📋 Audit Log"])
    
    with tab1:
        render_user_list(user_manager)
    
    with tab2:
        render_create_user(user_manager)
    
    with tab3:
        render_audit_log(user_manager)


def main():
    """Main entry point"""
    init_session_state()
    
    if not st.session_state.admin_authenticated:
        admin_login()
    else:
        main_app()


if __name__ == "__main__":
    main()
