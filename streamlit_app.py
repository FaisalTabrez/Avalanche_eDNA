
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

def main():
    """Main application function"""
    
    # Initialize session state
    state.init_session_state()

    # Header (includes global CSS)
    header.render()

    # Sidebar navigation
    PAGES = router.get_pages_config()
    
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
