import streamlit as st
import pandas as pd
import time
from src.ui.core.task_manager import TaskManager

def render():
    st.title("🖥️ System Monitor")
    
    tm = TaskManager()
    
    # 1. Resource Usage
    st.subheader("Resource Usage")
    metrics = tm.get_system_metrics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Usage", f"{metrics['cpu']}%")
        st.progress(metrics['cpu'] / 100)
    with col2:
        st.metric("Memory Usage", f"{metrics['memory']}%")
        st.progress(metrics['memory'] / 100)
    with col3:
        st.metric("Disk Usage", f"{metrics['disk']}%")
        st.progress(metrics['disk'] / 100)
        
    st.markdown("---")
    
    # 2. Task Management
    st.subheader("Task Manager")
    
    # Refresh button
    if st.button("🔄 Refresh Tasks"):
        st.rerun()
        
    tasks = tm.get_all_tasks()
    
    if not tasks:
        st.info("No tasks found.")
    else:
        # Convert to DataFrame for display
        df = pd.DataFrame(tasks)
        
        # Select columns to display
        display_cols = ["name", "status", "start_time", "end_time", "id"]
        st.dataframe(
            df[display_cols].style.applymap(
                lambda x: "color: green" if x == "completed" else 
                          ("color: red" if x == "failed" else 
                           ("color: blue" if x == "running" else "color: gray")),
                subset=["status"]
            ),
            use_container_width=True
        )
        
        # Task Actions
        st.subheader("Task Actions")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_task_id = st.selectbox(
                "Select Task", 
                options=[t['id'] for t in tasks],
                format_func=lambda x: f"{next((t['name'] for t in tasks if t['id'] == x), 'Unknown')} ({x})"
            )
        
        with c2:
            selected_task = next((t for t in tasks if t['id'] == selected_task_id), None)
            if selected_task and selected_task['status'] == 'running':
                if st.button("🛑 Stop Task", type="primary"):
                    if tm.stop_task(selected_task_id):
                        st.success(f"Task {selected_task_id} stopped.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to stop task.")
            else:
                st.button("🛑 Stop Task", disabled=True)
                
        # Task Details
        if selected_task:
            with st.expander("Task Details", expanded=True):
                st.json(selected_task)

