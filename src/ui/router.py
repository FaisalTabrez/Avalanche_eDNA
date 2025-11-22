"""
Page Router
"""
import streamlit as st
from src.ui.pages import (
    home,
    analysis,
    training,
    sra_browser,
    analysis_outputs,
    taxonomy,
    about
)

def render_page(page_key):
    """Render the requested page content"""
    
    if page_key == "home":
        home.render()
    elif page_key == "analysis":
        analysis.render()
    elif page_key == "training":
        training.render()
    elif page_key == "sra_browser":
        sra_browser.render()
    elif page_key == "analysis_outputs":
        analysis_outputs.render()
    elif page_key == "taxonomy":
        taxonomy.render()
    elif page_key == "about":
        about.render()
    else:
        st.error(f"Page not found: {page_key}")

def get_pages_config():
    """Return the list of available pages"""
    return [
        {"key": "home", "label": "Home"},
        {"key": "analysis", "label": "Dataset Analysis"},
        {"key": "training", "label": "Model Training"},
        {"key": "sra_browser", "label": "SRA Browser"},
        {"key": "analysis_outputs", "label": "Analysis Outputs"},
        {"key": "taxonomy", "label": "Taxonomy Viewer"},
        {"key": "about", "label": "About"},
    ]
