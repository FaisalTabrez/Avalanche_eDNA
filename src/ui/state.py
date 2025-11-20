"""
Session State Management
"""
import streamlit as st

def init_session_state():
    """Initialize session state variables"""
    if 'current_page_key' not in st.session_state:
        st.session_state.current_page_key = 'home'
    
    if 'sra_downloaded_file' not in st.session_state:
        # Used to pass downloaded file from SRA browser to Analysis page
        pass 
        
    if 'prefill_results_dir' not in st.session_state:
        st.session_state.prefill_results_dir = None
        
    if 'sra_batch_queue' not in st.session_state:
        st.session_state.sra_batch_queue = []
        
    if 'downloaded_sra' not in st.session_state:
        st.session_state.downloaded_sra = []

def get(key, default=None):
    """Get a value from session state"""
    return st.session_state.get(key, default)

def set(key, value):
    """Set a value in session state"""
    st.session_state[key] = value
