"""
Page Router
"""
import streamlit as st
from src.ui.pages import (
    home,
    analysis,
    training,
    sra_browser,
    biodiversity_results,
    taxonomy,
    about,
    login,
    user_management
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
    elif page_key == "biodiversity_results":
        biodiversity_results.render()
    elif page_key == "taxonomy":
        taxonomy.render()
    elif page_key == "about":
        about.render()
    elif page_key == "login":
        login.render()
    elif page_key == "user_management":
        user_management.render()
    else:
        st.error(f"Page not found: {page_key}")

def get_pages_config():
    """Return the list of available pages"""
    from src.auth import get_auth_manager
    
    auth = get_auth_manager()
    user = auth.get_current_user()
    
    # Base pages available to all
    pages = [
        {"key": "home", "label": "Home"},
        {"key": "login", "label": "Login"},
    ]
    
    # Pages requiring authentication
    if user:
        pages.extend([
            {"key": "analysis", "label": "Dataset Analysis"},
            {"key": "training", "label": "Model Training"},
            {"key": "sra_browser", "label": "SRA Browser"},
            {"key": "biodiversity_results", "label": "Biodiversity Results"},
            {"key": "taxonomy", "label": "Taxonomy Viewer"},
            {"key": "about", "label": "About"},
        ])
        
        # Admin-only pages
        if user['role'] == 'admin':
            pages.append({"key": "user_management", "label": "User Management"})
    
    return pages
