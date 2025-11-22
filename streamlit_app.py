
#!/usr/bin/env python3
"""
eDNA Biodiversity Assessment System - Streamlit GUI
A comprehensive web interface for biological sequence analysis
"""

import streamlit as st
import plotly.io as pio
import plotly.express as px
from pathlib import Path
import sys

# Add project modules
sys.path.append(str(Path(__file__).parent))

# UI Components
from src.ui.components import header, sidebar
from src.ui import router, state

# Authentication
from src.auth import get_auth_manager

# Page configuration
st.set_page_config(
    page_title="eDNA Biodiversity Assessment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotly dark theme and high-contrast palette
pio.templates.default = "plotly_dark"
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = (
    px.colors.qualitative.Bold + px.colors.qualitative.Set3 + px.colors.qualitative.Vivid
)

def initialize_auth():
    """Initialize authentication and create default admin if needed"""
    auth = get_auth_manager()
    
    # Create default admin on first run
    created, message = auth.create_default_admin()
    
    if created:
        # Show admin credentials in sidebar for first-time setup
        if 'admin_created_shown' not in st.session_state:
            st.sidebar.success("🔑 First-time setup complete!")
            st.sidebar.info(message)
            st.sidebar.warning("⚠️ Please change the default password immediately!")
            st.session_state.admin_created_shown = True

def main():
    """Main application function"""
    
    # Initialize session state
    state.init_session_state()
    
    # Initialize authentication
    initialize_auth()

    # Header (includes global CSS)
    header.render()

    # Sidebar navigation
    PAGES = router.get_pages_config()
    
    # Show user info in sidebar if authenticated
    auth = get_auth_manager()
    if auth.is_authenticated():
        user = auth.get_current_user()
        st.sidebar.divider()
        st.sidebar.write(f"👤 **{user['username']}** ({user['role']})")
        if st.sidebar.button("Logout", use_container_width=True):
            auth.logout()
            st.rerun()
    
    # Render sidebar and get selection
    current_key = state.get('current_page_key')
    selected_key = sidebar.render(current_key, PAGES)

    # Update session state when selection changes
    if selected_key != current_key:
        state.set('current_page_key', selected_key)
        st.rerun()

    # Render the selected page
    router.render_page(state.get('current_page_key'))

if __name__ == "__main__":
    main()
