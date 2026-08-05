import time
from pathlib import Path

import streamlit as st

from src.ui.core.state_manager import StateManager
from src.ui.core.task_manager import TaskManager


def render():
    st.title("🧬 Analysis Hub")

    # Initialize Wizard State
    if StateManager.get("analysis_step") is None:
        StateManager.set("analysis_step", 1)
        StateManager.set(
            "analysis_config",
            {
                "input_type": "Upload Files",
                "files": [],
                "sra_id": "",
                "database": "Silva 138",
                "identity": 97.0,
                "evalue": 1e-5,
                "threads": 4,
            },
        )

    current_step = StateManager.get("analysis_step")

    # Wizard Progress
    steps = ["Input Data", "Configuration", "Review & Run", "Execution"]
    progress = (current_step - 1) / (len(steps) - 1)
    st.progress(progress)

    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        if i + 1 == current_step:
            cols[i].markdown(f"**{i+1}. {step}**")
        elif i + 1 < current_step:
            cols[i].markdown(f"✅ {step}")
        else:
            cols[i].markdown(f"{i+1}. {step}")

    st.markdown("---")

    # Render Current Step
    if current_step == 1:
        render_step_1()
    elif current_step == 2:
        render_step_2()
    elif current_step == 3:
        render_step_3()
    elif current_step == 4:
        render_step_4()


def render_step_1():
    st.header("Step 1: Input Data")

    config = StateManager.get("analysis_config")

    input_type = st.radio(
        "Select Input Method",
        ["Upload Files", "SRA Accession"],
        index=0 if config["input_type"] == "Upload Files" else 1,
    )

    # Update config immediately on change
    if input_type != config["input_type"]:
        config["input_type"] = input_type
        StateManager.set("analysis_config", config)
        st.rerun()

    if input_type == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload FASTQ/FASTA files",
            accept_multiple_files=True,
            type=["fastq", "fasta", "fq", "fa", "gz"],
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} files selected.")
            # In a real app, we'd save these to a temp dir here

    else:
        sra_id = st.text_input(
            "Enter SRA Accession ID",
            value=config["sra_id"],
            placeholder="e.g., SRR12345678",
        )
        if sra_id:
            config["sra_id"] = sra_id
            StateManager.set("analysis_config", config)

    col1, col2 = st.columns([1, 5])
    with col2:
        if st.button("Next: Configuration ➡️"):
            # Validation
            if input_type == "SRA Accession" and not config["sra_id"]:
                st.error("Please enter an SRA ID.")
            else:
                StateManager.set("analysis_step", 2)
                st.rerun()


def render_step_2():
    st.header("Step 2: Configuration")

    config = StateManager.get("analysis_config")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Database Settings")
        db = st.selectbox(
            "Reference Database",
            ["Silva 138", "PR2", "Midori", "Custom"],
            index=["Silva 138", "PR2", "Midori", "Custom"].index(config["database"]),
        )
        config["database"] = db

    with col2:
        st.subheader("Algorithm Parameters")
        identity = st.slider(
            "Identity Threshold (%)",
            min_value=80.0,
            max_value=100.0,
            value=config["identity"],
        )
        config["identity"] = identity

        evalue = st.number_input("E-Value Cutoff", value=config["evalue"], format="%e")
        config["evalue"] = evalue

        threads = st.number_input(
            "CPU Threads", min_value=1, max_value=32, value=config["threads"]
        )
        config["threads"] = threads

    StateManager.set("analysis_config", config)

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("⬅️ Back"):
            StateManager.set("analysis_step", 1)
            st.rerun()
    with c3:
        if st.button("Next: Review ➡️"):
            StateManager.set("analysis_step", 3)
            st.rerun()


def render_step_3():
    st.header("Step 3: Review & Run")

    config = StateManager.get("analysis_config")

    st.info("Please review your settings before starting the analysis.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📂 Input")
        st.write(f"**Type:** {config['input_type']}")
        if config["input_type"] == "SRA Accession":
            st.write(f"**ID:** {config['sra_id']}")
        else:
            st.write("**Files:** (Selected in Step 1)")

    with col2:
        st.markdown("### ⚙️ Configuration")
        st.write(f"**Database:** {config['database']}")
        st.write(f"**Identity:** {config['identity']}%")
        st.write(f"**E-Value:** {config['evalue']}")
        st.write(f"**Threads:** {config['threads']}")

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("⬅️ Back"):
            StateManager.set("analysis_step", 2)
            st.rerun()
    with c3:
        if st.button("🚀 Start Analysis", type="primary"):
            # Submit Task
            tm = TaskManager()
            task_id = tm.submit_task(
                name=f"Analysis-{int(time.time())}",
                target_func=mock_analysis_task,  # We'll define this mock for now
                kwargs=config,
            )

            StateManager.set("current_analysis_id", task_id)
            StateManager.set("analysis_step", 4)
            st.rerun()


def render_step_4():
    st.header("Step 4: Execution")

    task_id = StateManager.get("current_analysis_id")
    tm = TaskManager()
    task = tm.get_task_status(task_id)

    if not task:
        st.error("Task not found.")
        if st.button("Start Over"):
            StateManager.set("analysis_step", 1)
            st.rerun()
        return

    st.markdown(f"### Task ID: `{task_id}`")

    status = task["status"]
    if status == "queued":
        st.info("⏳ Job is queued...")
    elif status == "running":
        st.info("🏃 Analysis in progress...")
        st.progress(50)  # Mock progress
    elif status == "completed":
        st.success("✅ Analysis completed successfully!")
        st.progress(100)
        if st.button("View Results in Data Explorer"):
            StateManager.set("current_page", "Data Explorer")
            st.rerun()
    elif status == "failed":
        st.error(f"❌ Analysis failed: {task.get('error')}")

    if status in ["queued", "running"]:
        time.sleep(2)
        st.rerun()

    if st.button("Start New Analysis"):
        StateManager.set("analysis_step", 1)
        st.rerun()


# Mock function to simulate backend work
def mock_analysis_task(**kwargs):
    import time

    time.sleep(5)  # Simulate work
    return {"total_reads": 15000, "identified_species": 42}
