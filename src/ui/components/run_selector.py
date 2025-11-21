"""
Reusable UI components for data selection across pages
"""
import streamlit as st
from pathlib import Path
from typing import Optional
from src.ui.data_manager import get_data_manager, RunInfo


def run_selector(
    key_prefix: str = "run_selector",
    show_filters: bool = True,
    auto_select_recent: bool = False,
    on_select_callback: Optional[callable] = None
) -> Optional[RunInfo]:
    """
    Unified run selector component for consistent UI across pages.
    
    Args:
        key_prefix: Unique prefix for widget keys
        show_filters: Show dataset filter and search
        auto_select_recent: Auto-select most recent run if nothing selected
        on_select_callback: Optional callback when run is selected
        
    Returns:
        Selected RunInfo or None
    """
    dm = get_data_manager()
    
    # Get current selection from session state
    current_run_path = dm.get_current_run()
    
    # Filters section
    dataset_filter = "All"
    search_query = ""
    
    if show_filters:
        filter_cols = st.columns([2, 2, 1])
        
        with filter_cols[0]:
            datasets = ["All"] + dm.get_datasets()
            dataset_filter = st.selectbox(
                "Dataset",
                datasets,
                key=f"{key_prefix}_dataset_filter"
            )
        
        with filter_cols[1]:
            search_query = st.text_input(
                "Search",
                placeholder="Search dataset or run ID...",
                key=f"{key_prefix}_search"
            )
        
        with filter_cols[2]:
            if st.button("🔄 Refresh", key=f"{key_prefix}_refresh"):
                st.cache_data.clear()
                st.rerun()
    
    # Get filtered runs
    runs = dm.search_runs(search_query, dataset_filter)
    
    if not runs:
        st.info("No runs found. Run an analysis from the Analysis page to create data.")
        return None
    
    # Auto-select most recent if requested and nothing selected
    if auto_select_recent and current_run_path is None:
        dm.set_current_run(runs[0].path)
        current_run_path = runs[0].path
    
    # Display run list with selection
    st.markdown("### Available Runs")
    
    for i, run in enumerate(runs[:20]):  # Limit to 20 for UI performance
        is_selected = current_run_path and run.path.resolve() == Path(current_run_path).resolve()
        
        with st.container():
            cols = st.columns([3, 2, 1, 1, 1, 1])
            
            # Run info
            status_icon = "✓" if is_selected else "○"
            cols[0].markdown(f"{status_icon} **{run.dataset}** / `{run.run_id}`")
            cols[1].text(run.modified)
            
            # Indicators
            cols[2].markdown("📊" if run.has_pipeline else "—")
            cols[3].markdown("🔬" if run.has_taxonomy else "—")
            cols[4].markdown("✨" if run.has_novelty else "—")
            
            # Select button
            btn_label = "Selected" if is_selected else "Select"
            btn_type = "secondary" if is_selected else "primary"
            
            if cols[5].button(btn_label, key=f"{key_prefix}_select_{i}", type=btn_type, disabled=is_selected):
                dm.set_current_run(run.path)
                if on_select_callback:
                    on_select_callback(run)
                st.rerun()
    
    # Show more indicator
    if len(runs) > 20:
        st.caption(f"Showing 20 of {len(runs)} runs. Use filters to narrow down.")
    
    # Return currently selected run
    if current_run_path:
        return dm.get_run_by_path(str(current_run_path))
    
    return None


def current_run_header(show_clear: bool = True) -> Optional[RunInfo]:
    """
    Display header showing currently selected run with option to clear.
    
    Args:
        show_clear: Show clear selection button
        
    Returns:
        Currently selected RunInfo or None
    """
    dm = get_data_manager()
    current_run_path = dm.get_current_run()
    
    if not current_run_path:
        st.info("ℹ️ No run selected. Use the Run Browser to select a run.")
        return None
    
    run = dm.get_run_by_path(str(current_run_path))
    
    if not run:
        st.warning(f"⚠️ Selected run not found: {current_run_path}")
        dm.clear_current_run()
        return None
    
    # Display header
    header_cols = st.columns([4, 1] if show_clear else [1])
    
    with header_cols[0]:
        st.markdown(
            f"### 📂 Current Run: **{run.dataset}** / `{run.run_id}`\n\n"
            f"📍 `{run.path}`  \n"
            f"🕐 {run.modified}"
        )
        
        # Quick status badges
        badges = []
        if run.has_pipeline:
            badges.append("📊 Pipeline")
        if run.has_taxonomy:
            badges.append("🔬 Taxonomy")
        if run.has_novelty:
            badges.append("✨ Novelty")
        if run.has_clustering:
            badges.append("🎯 Clustering")
        
        if badges:
            st.caption(" | ".join(badges))
    
    if show_clear and len(header_cols) > 1:
        with header_cols[1]:
            if st.button("Clear Selection", key="clear_current_run"):
                dm.clear_current_run()
                st.rerun()
    
    return run


def quick_run_picker(
    key_prefix: str = "quick_picker",
    label: str = "Select Run"
) -> Optional[RunInfo]:
    """
    Compact dropdown run picker for pages with limited space.
    
    Args:
        key_prefix: Unique prefix for widget keys
        label: Label for the selector
        
    Returns:
        Selected RunInfo or None
    """
    dm = get_data_manager()
    runs = dm.discover_runs()
    
    if not runs:
        st.info("No runs found.")
        return None
    
    # Create options list
    options = ["(Select a run)"] + [
        f"{r.dataset} / {r.run_id} ({r.modified})"
        for r in runs[:50]  # Limit for performance
    ]
    
    # Get current selection index
    current_run_path = dm.get_current_run()
    current_idx = 0
    
    if current_run_path:
        for i, run in enumerate(runs[:50]):
            if run.path.resolve() == Path(current_run_path).resolve():
                current_idx = i + 1
                break
    
    # Dropdown selector
    selected_idx = st.selectbox(
        label,
        range(len(options)),
        format_func=lambda i: options[i],
        index=current_idx,
        key=f"{key_prefix}_dropdown"
    )
    
    # Handle selection
    if selected_idx == 0:
        return None
    
    selected_run = runs[selected_idx - 1]
    
    # Update session state if changed
    if not current_run_path or Path(current_run_path).resolve() != selected_run.path.resolve():
        dm.set_current_run(selected_run.path)
    
    return selected_run
