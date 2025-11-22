"""
Run Browser Page
"""
import streamlit as st
from pathlib import Path
from src.ui.components.run_selector import run_selector

def render():
    """Browse and select runs using the centralized data manager"""
    st.title("Run Browser")
    
    st.markdown("""
    Browse and select analysis runs. The selected run will be used across all pages 
    (Results Viewer, Taxonomy Viewer, etc.).
    """)
    
    st.markdown("---")
    
    # Use the centralized run selector component
    selected_run = run_selector(
        key_prefix="run_browser",
        show_filters=True,
        auto_select_recent=False
    )
    
    if selected_run:
        st.success(f"✓ Run selected: **{selected_run.dataset}** / `{selected_run.run_id}`")
        st.info("Navigate to **Results Viewer**, **Taxonomy Viewer**, or other pages to view this run's data.")
