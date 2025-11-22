"""
Login page for user authentication
"""
import streamlit as st
from src.auth import get_auth_manager


def render():
    """Render the login page"""
    st.title("Login to Avalanche")
    
    auth = get_auth_manager()
    
    # Check if already authenticated
    if auth.is_authenticated():
        user = auth.get_current_user()
        st.success(f"Already logged in as {user['username']}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Go to Home", use_container_width=True):
                st.switch_page("pages/home.py")
        with col2:
            if st.button("Logout", use_container_width=True):
                auth.logout()
                st.rerun()
        return
    
    # Create tabs for login and register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        render_login_form(auth)
    
    with tab2:
        render_register_form(auth)


def render_login_form(auth):
    """Render the login form"""
    st.subheader("Sign In")
    
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            submit = st.form_submit_button("Login", use_container_width=True)
        
        with col2:
            if st.form_submit_button("Forgot Password?", use_container_width=True):
                st.info("Please contact your administrator to reset your password.")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                success, message = auth.login(username, password)
                
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


def render_register_form(auth):
    """Render the registration form"""
    st.subheader("Create Account")
    
    st.info("Registration creates a viewer account by default. Contact an administrator to upgrade your role.")
    
    with st.form("register_form"):
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm")
        
        st.caption("Password requirements:")
        st.caption("- At least 8 characters")
        st.caption("- At least one uppercase letter")
        st.caption("- At least one lowercase letter")
        st.caption("- At least one digit")
        st.caption("- At least one special character (!@#$%^&*(),.?\":{}|<>)")
        
        submit = st.form_submit_button("Register", use_container_width=True)
        
        if submit:
            if not username or not email or not password or not confirm_password:
                st.error("Please fill in all fields")
            else:
                success, message = auth.register_user(
                    username=username,
                    email=email,
                    password=password,
                    confirm_password=confirm_password,
                    role='viewer'
                )
                
                if success:
                    st.success(message)
                    st.info("You can now log in with your credentials.")
                else:
                    st.error(message)


if __name__ == "__main__":
    render()
