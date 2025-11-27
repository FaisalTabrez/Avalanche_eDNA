import streamlit as st
import time
from src.ui.core.task_manager import TaskManager
from src.ui.core.state_manager import StateManager

def render():
    st.title("🚀 Mission Control")
    st.markdown("Welcome to the Avalanche eDNA System.")
    
    # Initialize TaskManager
    tm = TaskManager()
    metrics = tm.get_system_metrics()
    
    # Update State
    StateManager.update_nested("system_status", "cpu", metrics["cpu"])
    StateManager.update_nested("system_status", "memory", metrics["memory"])
    StateManager.update_nested("system_status", "disk", metrics["disk"])
    
    # Auto-refresh for dashboard
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="avalanche-card">
            <h3>System Status</h3>
            <div class="metric-value">Online</div>
            <div class="metric-label">All Systems Go</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        active_tasks = len([t for t in tm.get_all_tasks() if t['status'] == 'running'])
        st.markdown(f"""
        <div class="avalanche-card">
            <h3>Active Jobs</h3>
            <div class="metric-value">{active_tasks}</div>
            <div class="metric-label">Running</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="avalanche-card">
            <h3>Storage</h3>
            <div class="metric-value">{metrics['disk']}%</div>
            <div class="metric-label">Used</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="avalanche-card">
            <h3>Database</h3>
            <div class="metric-value">Connected</div>
            <div class="metric-label">PostgreSQL</div>
        </div>
        """, unsafe_allow_html=True)

    # Recent Activity Section
    st.subheader("Recent Activity")
    tasks = tm.get_all_tasks()
    # Sort by start time descending
    tasks.sort(key=lambda x: x['start_time'], reverse=True)
    
    if not tasks:
        st.info("No recent activity.")
    else:
        for task in tasks[:5]:
            status_color = {
                "completed": "🟢",
                "running": "🔵",
                "failed": "🔴",
                "stopped": "⚪"
            }.get(task['status'], "⚪")
            
            with st.expander(f"{status_color} {task['name']} - {task['status'].upper()}"):
                st.write(f"**ID:** {task['id']}")
                st.write(f"**Started:** {task['start_time']}")
                if task['end_time']:
                    st.write(f"**Ended:** {task['end_time']}")
                if task['error']:
                    st.error(f"Error: {task['error']}")
                if task.get('result'):
                    st.success(f"Result: {task['result']}")
