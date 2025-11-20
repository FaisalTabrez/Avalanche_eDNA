"""
Sidebar Component
"""
import streamlit as st

def render(current_page_key, pages):
    """
    Render the sidebar navigation
    
    Args:
        current_page_key: The key of the currently selected page
        pages: List of page dictionaries with 'key' and 'label'
        
    Returns:
        The selected page key
    """
    st.sidebar.markdown("### Navigation")

    # Find current index
    key_to_index = {p["key"]: i for i, p in enumerate(pages)}
    current_index = key_to_index.get(current_page_key, 0)

    selection = st.sidebar.radio(
        "Navigation",
        pages,
        index=current_index,
        format_func=lambda p: p["label"],
        key="page_selector",
        label_visibility="collapsed"
    )
    
    return selection["key"]
