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
    progress_updates,
    model_training_dashboard,
    dynamic_scaling_config,
    about
)
from src.ui.workflow import render_workflow_hub

def render_page(page_key):
    """Render the requested page content"""
    
    if page_key == "home":
        home.render()
    elif page_key == "workflow_hub":
        render_workflow_hub()
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
    elif page_key == "progress_updates":
        progress_updates.render()
    elif page_key == "model_training_dashboard":
        model_training_dashboard.main()
    elif page_key == "dynamic_scaling_config":
        dynamic_scaling_config.render()
    elif page_key == "about":
        about.render()
    else:
        st.error(f"Page not found: {page_key}")

def get_pages_config():
    """Return the list of available pages"""
    return [
        {"key": "home", "label": "Home"},
        {"key": "workflow_hub", "label": "🧬 Workflow Hub", "icon": "🧬"},
        {"key": "analysis", "label": "Dataset Analysis"},
        {"key": "progress_updates", "label": "Pipeline Progress"},
        {"key": "training", "label": "Model Training"},
        {"key": "model_training_dashboard", "label": "Training Dashboard"},
        {"key": "dynamic_scaling_config", "label": "🚀 Dynamic Scaling"},
        {"key": "sra_browser", "label": "SRA Browser"},
        {"key": "biodiversity_results", "label": "Biodiversity Results"},
        {"key": "taxonomy", "label": "Taxonomy Viewer"},
        {"key": "about", "label": "About"},
    ]
