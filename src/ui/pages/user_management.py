"""
User management page for administrators
"""
import streamlit as st
from src.auth import get_auth_manager, require_role
from src.auth.password_utils import validate_password_strength


@require_role('admin')
def render():
    """Render the user management page"""
    st.title("User Management")
    
    auth = get_auth_manager()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["User List", "Create User", "Audit Log"])
    
    with tab1:
        render_user_list(auth)
    
    with tab2:
        render_create_user(auth)
    
    with tab3:
        render_audit_log()


def render_user_list(auth):
    """Render list of all users"""
    st.subheader("All Users")
    
    users = auth.user_manager.list_users()
    
    if not users:
        st.info("No users found")
        return
    
    # Display users in expandable cards
    for user in users:
        with st.expander(f"{user['username']} ({user['role']})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**User ID:** {user['user_id']}")
                st.write(f"**Username:** {user['username']}")
                st.write(f"**Email:** {user['email']}")
            
            with col2:
                st.write(f"**Role:** {user['role']}")
                st.write(f"**Status:** {'Active' if user['is_active'] else 'Inactive'}")
                st.write(f"**Created:** {user['created_at'][:10]}")
            
            if user['last_login']:
                st.write(f"**Last Login:** {user['last_login']}")
            
            # Actions
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Change role
                new_role = st.selectbox(
                    "Change Role",
                    options=['admin', 'analyst', 'viewer'],
                    index=['admin', 'analyst', 'viewer'].index(user['role']),
                    key=f"role_{user['user_id']}"
                )
                
                if st.button("Update Role", key=f"update_role_{user['user_id']}"):
                    success, message = auth.user_manager.update_user(
                        user['user_id'],
                        role=new_role
                    )
                    
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            with col2:
                # Toggle active status
                new_status = st.checkbox(
                    "Active",
                    value=bool(user['is_active']),
                    key=f"active_{user['user_id']}"
                )
                
                if new_status != bool(user['is_active']):
                    if st.button("Update Status", key=f"update_status_{user['user_id']}"):
                        success, message = auth.user_manager.update_user(
                            user['user_id'],
                            is_active=new_status
                        )
                        
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            with col3:
                # Reset password
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
                            st.error(error_msg)
                        elif st.button("Confirm Reset", key=f"confirm_reset_{user['user_id']}"):
                            success, message = auth.user_manager.change_password(
                                user['user_id'],
                                new_password
                            )
                            
                            if success:
                                st.success(message)
                                st.session_state[f'reset_password_{user["user_id"]}'] = False
                                st.rerun()
                            else:
                                st.error(message)
            
            with col4:
                # Delete user
                current_user = auth.get_current_user()
                if user['user_id'] != current_user['user_id']:
                    if st.button("Delete User", key=f"delete_{user['user_id']}", type="primary"):
                        if st.session_state.get(f'confirm_delete_{user["user_id"]}', False):
                            success, message = auth.user_manager.delete_user(user['user_id'])
                            
                            if success:
                                st.success(message)
                                st.session_state[f'confirm_delete_{user["user_id"]}'] = False
                                st.rerun()
                            else:
                                st.error(message)
                        else:
                            st.session_state[f'confirm_delete_{user["user_id"]}'] = True
                            st.warning("Click again to confirm deletion")
                else:
                    st.info("Cannot delete yourself")


def render_create_user(auth):
    """Render form to create a new user"""
    st.subheader("Create New User")
    
    with st.form("create_user_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        
        col1, col2 = st.columns(2)
        
        with col1:
            password = st.text_input("Password", type="password")
        
        with col2:
            confirm_password = st.text_input("Confirm Password", type="password")
        
        role = st.selectbox("Role", options=['viewer', 'analyst', 'admin'])
        
        st.caption("Password requirements:")
        st.caption("- At least 8 characters")
        st.caption("- At least one uppercase, lowercase, digit, and special character")
        
        submit = st.form_submit_button("Create User", use_container_width=True)
        
        if submit:
            if not username or not email or not password or not confirm_password:
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                # Validate password strength
                is_valid, error_msg = validate_password_strength(password)
                
                if not is_valid:
                    st.error(error_msg)
                else:
                    success, result = auth.user_manager.create_user(
                        username=username,
                        email=email,
                        password=password,
                        role=role
                    )
                    
                    if success:
                        st.success(f"User created successfully with ID: {result}")
                        st.rerun()
                    else:
                        st.error(result)


def render_audit_log():
    """Render audit log of user actions"""
    st.subheader("Audit Log")
    
    import sqlite3
    from pathlib import Path
    
    db_path = Path("data/avalanche_users.db")
    
    if not db_path.exists():
        st.info("No audit log available")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get recent audit log entries
    cursor.execute("""
        SELECT 
            audit_log.*,
            users.username
        FROM audit_log
        LEFT JOIN users ON audit_log.user_id = users.user_id
        ORDER BY timestamp DESC
        LIMIT 100
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        st.info("No audit log entries")
        return
    
    # Display as table
    import pandas as pd
    
    df = pd.DataFrame([dict(row) for row in rows])
    
    # Format timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select columns to display
    display_columns = ['timestamp', 'username', 'action', 'details', 'ip_address']
    df_display = df[display_columns].copy()
    df_display.columns = ['Timestamp', 'User', 'Action', 'Details', 'IP Address']
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
