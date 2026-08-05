"""
Step 3: Execute & Monitor Component
Real-time progress monitoring for analysis and training
"""

import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from src.ui.task_manager import TaskStatus, TaskType, get_task_manager


def render_execution():
    """Render execution and monitoring interface"""

    # Check if we have dataset and config
    if not st.session_state.workflow_dataset:
        st.warning("⚠️ No dataset selected. Please go back to Step 1.")
        if st.button("← Back to Dataset Selection"):
            st.session_state.workflow_step = 1
            st.rerun()
        return

    dataset = st.session_state.workflow_dataset
    config = st.session_state.workflow_config

    # Show configuration summary
    st.markdown("### Pipeline Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", dataset.get("name", "Unknown"))
        st.caption(f"{dataset.get('size_mb', 0):.2f} MB")

    with col2:
        preset_name = config.get("preset", "custom")
        from src.ui.workflow.step_2_configure import PRESETS

        preset_info = PRESETS.get(preset_name, {})
        st.metric("Preset", preset_info.get("name", "Custom"))
        st.caption(preset_info.get("estimated_time", "Unknown"))

    with col3:
        st.metric("Memory", preset_info.get("memory_usage", "Unknown"))

    # Navigation back
    col_back, col_space = st.columns([1, 4])
    with col_back:
        if st.button("← Configure"):
            st.session_state.workflow_step = 2
            st.rerun()

    st.divider()

    # Check if task already running
    task_manager = get_task_manager()
    current_task_id = st.session_state.workflow_current_task_id

    if current_task_id:
        task = task_manager.get_task(current_task_id)
        if task:
            if task.status == TaskStatus.RUNNING:
                render_live_progress(task)
            elif task.status == TaskStatus.COMPLETED:
                st.success("✅ Pipeline completed successfully!")
                st.info("View results in Step 4")

                if st.button("View Results →", type="primary"):
                    st.session_state.workflow_step = 4
                    st.session_state.workflow_results = task.results
                    st.rerun()
            elif task.status == TaskStatus.FAILED:
                st.error(f"❌ Pipeline failed: {task.error}")

                if st.button("🔄 Try Again"):
                    st.session_state.workflow_current_task_id = None
                    st.rerun()
            elif task.status == TaskStatus.PAUSED:
                st.warning("⏸️ Pipeline paused")
                render_live_progress(task)
            elif task.status == TaskStatus.STOPPED:
                st.warning("⏹️ Pipeline stopped")

                if st.button("🔄 Start New Run"):
                    st.session_state.workflow_current_task_id = None
                    st.rerun()
            return

    # No task running - show start button
    st.markdown("### Ready to Execute")

    st.info(
        "Click the button below to start the pipeline. You can monitor progress in real-time."
    )

    if st.button(
        "🚀 Start Pipeline Execution", type="primary", use_container_width=True
    ):
        start_pipeline_execution()


def start_pipeline_execution():
    """Start the pipeline execution"""
    task_manager = get_task_manager()

    dataset = st.session_state.workflow_dataset
    config = st.session_state.workflow_config

    # Create task
    task_id = task_manager.create_task(
        task_type=TaskType.ANALYSIS,
        name=f"Analysis: {dataset.get('name', 'Unknown')}",
        dataset_name=dataset.get("name"),
        config=config,
    )

    # Store task ID
    st.session_state.workflow_current_task_id = task_id

    # Start task (simplified - in production would call actual analysis function)
    # For now, just start a demo task
    task_manager.start_task(
        task_id=task_id,
        target_func=demo_analysis_task,
        kwargs={"dataset_path": dataset.get("file_path"), "config": config},
    )

    st.rerun()


def demo_analysis_task(progress_queue, stop_event, task_id, dataset_path, config):
    """
    Demo analysis task for testing
    In production, this would call the actual analysis pipeline
    """
    import random
    import time

    stages = [
        "Loading dataset",
        "Preprocessing sequences",
        "Quality analysis",
        "Generating embeddings",
        "Clustering analysis",
        "Taxonomy classification",
        "Computing diversity metrics",
        "Training model",
        "Generating results",
    ]

    total_stages = len(stages)

    for i, stage in enumerate(stages):
        if stop_event.is_set():
            break

        progress = (i / total_stages) * 100

        progress_queue.put(
            {
                "type": "progress",
                "progress": progress,
                "stage": stage,
                "message": f"Processing {stage.lower()}...",
                "cpu_percent": random.uniform(40, 90),
                "memory_mb": random.uniform(1000, 4000),
                "gpu_memory_mb": random.uniform(2000, 6000),
                "eta": (total_stages - i - 1) * 15,  # 15 seconds per stage
            }
        )

        # Simulate work
        time.sleep(3)

    # Complete
    if not stop_event.is_set():
        progress_queue.put(
            {
                "type": "complete",
                "result": {
                    "sequences_analyzed": 45120,
                    "clusters_found": 127,
                    "novel_taxa": 23,
                    "accuracy": 94.2,
                },
            }
        )


def render_live_progress(task):
    """Render real-time progress monitoring"""

    st.markdown("### Pipeline Progress")

    # Overall progress
    st.progress(task.progress / 100.0)

    # Status row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", task.stage)

    with col2:
        elapsed = format_duration(task.elapsed_time)
        st.metric("Elapsed", elapsed)

    with col3:
        if task.estimated_time_remaining:
            eta = format_duration(task.estimated_time_remaining)
            st.metric("ETA", eta)
        else:
            st.metric("ETA", "Calculating...")

    with col4:
        st.metric("Progress", f"{task.progress:.1f}%")

    st.divider()

    # Current stage details
    st.markdown("#### Current Stage")

    with st.container():
        st.markdown(f"**{task.stage}**")
        st.caption(task.message)

    # Resource monitoring
    st.markdown("#### Resource Usage")

    col1, col2, col3 = st.columns(3)

    with col1:
        cpu_pct = task.cpu_percent
        st.metric("CPU", f"{cpu_pct:.1f}%")
        st.progress(min(cpu_pct / 100.0, 1.0))

    with col2:
        mem_gb = task.memory_mb / 1024
        st.metric("Memory", f"{mem_gb:.2f} GB")
        # Assume 16GB total for progress bar
        st.progress(min(mem_gb / 16.0, 1.0))

    with col3:
        gpu_gb = task.gpu_memory_mb / 1024
        st.metric("GPU Memory", f"{gpu_gb:.2f} GB")
        # Assume 8GB GPU for progress bar
        st.progress(min(gpu_gb / 8.0, 1.0))

    st.divider()

    # Live log (simplified)
    st.markdown("#### Live Log")

    log_container = st.container()
    with log_container:
        # In production, would stream actual logs
        st.code(
            f"""
[{datetime.now().strftime('%H:%M:%S')}] {task.stage}
[{datetime.now().strftime('%H:%M:%S')}] {task.message}
[{datetime.now().strftime('%H:%M:%S')}] Progress: {task.progress:.1f}%
[{datetime.now().strftime('%H:%M:%S')}] CPU: {task.cpu_percent:.1f}%, Memory: {task.memory_mb:.0f} MB
        """.strip(),
            language="log",
        )

    # Action buttons
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if task.status == TaskStatus.RUNNING:
            if st.button("⏸️ Pause"):
                task_manager = get_task_manager()
                try:
                    task_manager.pause_task(task.task_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if task.status in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
            if st.button("⏹️ Stop"):
                task_manager = get_task_manager()
                try:
                    task_manager.stop_task(task.task_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # Auto-refresh for live updates
    time.sleep(2)
    st.rerun()


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
