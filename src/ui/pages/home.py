"""
Home Page
"""
import streamlit as st
from pathlib import Path
from src.utils.config import config as app_config

def render():
    """Display the home page with navigation and quick links"""
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="color: #64FFDA; font-size: 2rem; margin-bottom: 1rem;">
            Deep Sea Environmental DNA Analysis
        </h2>
        <p style="font-size: 1.2rem; color: #8892b0; max-width: 800px; margin: 0 auto;">
            Advanced biodiversity assessment using next-generation sequencing and machine learning.
            Uncover the secrets of the deep ocean with precision and speed.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Cards
    st.markdown("### Core Modules")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("#### Analysis")
            st.info("Process raw sequencing data, perform quality control, and generate taxonomic profiles.")
            if st.button("Launch Analysis", use_container_width=True, key="btn_analysis"):
                st.session_state.current_page_key = "analysis"
                st.rerun()
            
            st.markdown("#### Taxonomy")
            st.info("Explore taxonomic classifications, visualize diversity, and resolve conflicts.")
            if st.button("View Taxonomy", use_container_width=True, key="btn_taxonomy"):
                st.session_state.current_page_key = "taxonomy"
                st.rerun()

    with col2:
        with st.container():
            st.markdown("#### Model Training")
            st.info("Train custom DNA embedding models using Contrastive Learning or Autoencoders.")
            if st.button("Train Models", use_container_width=True, key="btn_training"):
                st.session_state.current_page_key = "training"
                st.rerun()

            st.markdown("#### Biodiversity Results")
            st.info("Browse all stored analysis runs, view results, and upload external datasets.")
            if st.button("View Results", use_container_width=True, key="btn_outputs"):
                st.session_state.current_page_key = "biodiversity_results"
                st.rerun()

    with col3:
        with st.container():
            st.markdown("#### Quick Actions")
            st.success("Ready to start? Begin a new analysis workflow immediately.")
            if st.button("Start New Run", type="primary", use_container_width=True, key="btn_start"):
                st.session_state.current_page_key = "analysis"
                st.rerun()
            
            st.markdown("#### SRA Browser")
            st.info("Search and download sequences from the NCBI Sequence Read Archive.")
            if st.button("Browse SRA", use_container_width=True, key="btn_sra"):
                st.session_state.current_page_key = "sra_browser"
                st.rerun()
    
    # Recent runs quick links using centralized data manager
    st.markdown("---")
    st.markdown("### Recent Activity")
    
    from src.ui.data_manager import get_data_manager
    dm = get_data_manager()
    
    try:
        recent_runs = dm.get_recent_runs(limit=6)
        if recent_runs:
            cols = st.columns(3)
            for idx, run in enumerate(recent_runs):
                with cols[idx % 3]:
                    label = f"{run.dataset} / {run.run_id}"
                    if st.button(f"{label}", key=f"recent_{idx}", use_container_width=True):
                        dm.set_current_run(run.path)
                        st.session_state.current_page_key = "biodiversity_results"
                        st.rerun()
        else:
            st.info("No runs found. Run an analysis to get started!")
    except Exception as e:
        st.warning(f"Could not list recent runs: {e}")
    
    # Feature overview
    st.markdown("---")
    st.markdown("## Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### Sequence Analysis
        - Multi-format support (FASTA, FASTQ, Swiss-Prot)
        - Automatic format detection
        - Quality assessment
        - Composition analysis
        """)
    with col2:
        st.markdown("""
        ### Biodiversity Metrics
        - Shannon and Simpson indices
        - Species richness
        - Evenness measures
        """)
    with col3:
        st.markdown("""
        ### Advanced Features
        - Interactive visualizations
        - Batch analysis (planned)
        - Export capabilities
        """)
