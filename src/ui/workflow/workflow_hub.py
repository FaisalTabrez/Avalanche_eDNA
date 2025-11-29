"""
Unified Workflow Hub - Main orchestration page
Single-page interface for dataset → configure → execute → results
"""

import streamlit as st
from typing import Optional, Dict, Any
import time
from datetime import datetime

from src.ui.task_manager import get_task_manager, TaskStatus, TaskType


def init_workflow_state():
    """Initialize workflow session state"""
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 1  # 1=Dataset, 2=Config, 3=Execute, 4=Results
    
    if 'workflow_dataset' not in st.session_state:
        st.session_state.workflow_dataset = None
    
    if 'workflow_dataset_type' not in st.session_state:
        st.session_state.workflow_dataset_type = None  # 'upload', 'existing', 'sra'
    
    if 'workflow_config' not in st.session_state:
        st.session_state.workflow_config = {
            'preset': 'full_edna',
            'analysis_settings': {},
            'model_settings': {},
            'scaling_settings': {}
        }
    
    if 'workflow_current_task_id' not in st.session_state:
        st.session_state.workflow_current_task_id = None
    
    if 'workflow_results' not in st.session_state:
        st.session_state.workflow_results = None
    
    if 'task_panel_expanded' not in st.session_state:
        st.session_state.task_panel_expanded = True


def render_workflow_hub():
    """Main workflow hub page"""
    init_workflow_state()
    
    # Page header
    st.markdown("# 🧬 Analysis & Training Hub")
    
    # Show current active task if any
    task_manager = get_task_manager()
    current_task_id = st.session_state.workflow_current_task_id
    
    if current_task_id:
        task = task_manager.get_task(current_task_id)
        if task and task.status == TaskStatus.RUNNING:
            st.success(f"🟢 Current: {task.name} - {task.progress:.1f}% complete")
        elif task and task.status == TaskStatus.COMPLETED:
            st.info(f"✅ Completed: {task.name}")
    
    st.divider()
    
    # Main content area with sidebar navigation
    col_nav, col_content = st.columns([1, 4])
    
    with col_nav:
        render_workflow_steps()
    
    with col_content:
        # Render appropriate step based on current workflow step
        current_step = st.session_state.workflow_step
        
        if current_step == 1:
            render_dataset_step()
        elif current_step == 2:
            render_configure_step()
        elif current_step == 3:
            render_execute_step()
        elif current_step == 4:
            render_results_step()
    
    st.divider()
    
    # Task panel at bottom
    render_task_panel()


def render_workflow_steps():
    """Render workflow step navigation sidebar"""
    st.markdown("### Workflow Steps")
    
    current_step = st.session_state.workflow_step
    
    # Step indicators
    steps = [
        (1, "Dataset", "📂"),
        (2, "Configure", "⚙️"),
        (3, "Execute", "▶️"),
        (4, "Results", "📊")
    ]
    
    for step_num, step_name, icon in steps:
        if step_num == current_step:
            st.markdown(f"**→ {step_num}. {icon} {step_name}**")
        elif step_num < current_step:
            if st.button(f"✓ {step_num}. {step_name}", key=f"nav_step_{step_num}", use_container_width=True):
                st.session_state.workflow_step = step_num
                st.rerun()
        else:
            st.markdown(f"○ {step_num}. {icon} {step_name}")
    
    st.divider()
    
    # Quick actions
    st.markdown("### Quick Actions")
    if st.button("🔄 New Analysis", use_container_width=True):
        # Reset workflow
        st.session_state.workflow_step = 1
        st.session_state.workflow_dataset = None
        st.session_state.workflow_current_task_id = None
        st.session_state.workflow_results = None
        st.rerun()


def render_dataset_step():
    """Step 1: Dataset Selection"""
    st.markdown("## Step 1: Select Dataset")
    
    # Import step component
    from src.ui.workflow.step_1_dataset import render_dataset_selection
    
    render_dataset_selection()


def render_configure_step():
    """Step 2: Configuration"""
    st.markdown("## Step 2: Configure Analysis & Training")
    
    # Show selected dataset
    if st.session_state.workflow_dataset:
        st.success(f"✓ Dataset: {st.session_state.workflow_dataset.get('name', 'Unknown')}")
    
    # Import step component
    from src.ui.workflow.step_2_configure import render_configuration
    
    render_configuration()


def render_execute_step():
    """Step 3: Execute & Monitor"""
    st.markdown("## Step 3: Execute & Monitor")
    
    # Import step component
    from src.ui.workflow.step_3_execute import render_execution
    
    render_execution()


def render_results_step():
    """Step 4: Results Dashboard"""
    st.markdown("## Step 4: Results Dashboard")
    
    # Import step component
    from src.ui.workflow.step_4_results import render_results
    
    render_results()


def render_task_panel():
    """Render persistent task panel at bottom"""
    task_manager = get_task_manager()
    active_tasks = task_manager.get_active_tasks()
    all_tasks = task_manager.get_all_tasks()
    
    # Panel header
    col1, col2 = st.columns([4, 1])
    with col1:
        if st.session_state.task_panel_expanded:
            st.markdown("### 📋 Active Tasks")
        else:
            st.markdown("### 📋 Active Tasks (Collapsed)")
    
    with col2:
        if st.button("▲ Collapse" if st.session_state.task_panel_expanded else "▼ Expand", 
                     key="toggle_task_panel"):
            st.session_state.task_panel_expanded = not st.session_state.task_panel_expanded
            st.rerun()
    
    if not st.session_state.task_panel_expanded:
        # Show minimal info
        if active_tasks:
            st.info(f"{len(active_tasks)} active task(s)")
        return
    
    # Show all tasks
    if not all_tasks:
        st.info("No tasks yet. Start an analysis or training to see tasks here.")
        return
    
    # Clear completed button
    completed_count = len([t for t in all_tasks if t.status in [
        TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED
    ]])
    
    if completed_count > 0:
        if st.button(f"🗑️ Clear {completed_count} Completed", key="clear_completed"):
            task_manager.clear_completed_tasks()
            st.rerun()
    
    # Render each task
    for task in reversed(all_tasks):  # Most recent first
        render_task_card(task)


def render_task_card(task):
    """Render a single task card"""
    task_manager = get_task_manager()
    
    # Status emoji
    status_emoji = {
        TaskStatus.QUEUED: "⏳",
        TaskStatus.RUNNING: "🟢",
        TaskStatus.PAUSED: "🟡",
        TaskStatus.COMPLETED: "✅",
        TaskStatus.FAILED: "❌",
        TaskStatus.STOPPED: "⏹️"
    }
    
    emoji = status_emoji.get(task.status, "○")
    
    with st.container():
        st.markdown(f"#### {emoji} {task.name}")
        
        # Progress bar for running/paused tasks
        if task.status in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
            st.progress(task.progress / 100.0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Stage: {task.stage}")
            with col2:
                elapsed = format_duration(task.elapsed_time)
                st.caption(f"Elapsed: {elapsed}")
            with col3:
                if task.estimated_time_remaining:
                    eta = format_duration(task.estimated_time_remaining)
                    st.caption(f"ETA: {eta}")
        
        # Message
        st.caption(task.message)
        
        # Action buttons
        button_cols = st.columns([1, 1, 1, 1, 4])
        
        with button_cols[0]:
            if task.status == TaskStatus.RUNNING:
                if st.button("⏸️ Pause", key=f"pause_{task.task_id}"):
                    try:
                        task_manager.pause_task(task.task_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with button_cols[1]:
            if task.status in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
                if st.button("⏹️ Stop", key=f"stop_{task.task_id}"):
                    try:
                        task_manager.stop_task(task.task_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with button_cols[2]:
            if task.status == TaskStatus.COMPLETED:
                if st.button("📊 View", key=f"view_{task.task_id}"):
                    st.session_state.workflow_current_task_id = task.task_id
                    st.session_state.workflow_step = 4
                    st.session_state.workflow_results = task.results
                    st.rerun()
        
        with button_cols[3]:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED]:
                if st.button("🗑️", key=f"remove_{task.task_id}"):
                    try:
                        task_manager.remove_task(task.task_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.divider()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
