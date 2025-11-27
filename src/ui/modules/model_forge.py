import streamlit as st
import time
import random
import pandas as pd
import plotly.express as px
from src.ui.core.state_manager import StateManager
from src.ui.core.task_manager import TaskManager

def render():
    st.title("⚡ Model Forge")
    st.markdown("Train, fine-tune, and scale your eDNA models.")

    tab1, tab2 = st.tabs(["🏋️ Training Arena", "⚖️ Dynamic Scaling"])

    with tab1:
        render_training_tab()
    
    with tab2:
        render_scaling_tab()

def render_training_tab():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        
        model_name = st.text_input("Model Name", value="eDNA-Transformer-v1")
        
        epochs = st.slider("Epochs", 1, 100, 10)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256, 512], value=64)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
        
        st.markdown("---")
        
        if st.button("🚀 Start Training", type="primary"):
            tm = TaskManager()
            task_id = tm.submit_task(
                name=f"Train-{model_name}",
                target_func=mock_training_task,
                kwargs={"epochs": epochs}
            )
            StateManager.set("current_training_id", task_id)
            st.toast(f"Training started! ID: {task_id}")

    with col2:
        st.subheader("Live Metrics")
        
        # Check for active training task
        task_id = StateManager.get("current_training_id")
        if task_id:
            tm = TaskManager()
            task = tm.get_task_status(task_id)
            
            if task and task['status'] == 'running':
                st.info(f"Training {task['name']}...")
                
                # Mock Live Plot
                # In a real app, we'd read from a metrics file updated by the worker
                progress = st.progress(0)
                chart_placeholder = st.empty()
                
                # Simulate live updates (just for UI demo feel)
                loss_data = []
                for i in range(1, 101):
                    loss = 1.0 / (i * 0.1 + 1) + random.random() * 0.1
                    loss_data.append({"Epoch": i, "Loss": loss})
                    
                    if i % 10 == 0:
                        df = pd.DataFrame(loss_data)
                        fig = px.line(df, x="Epoch", y="Loss", title="Training Loss")
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        progress.progress(i)
                        time.sleep(0.05)
                
                st.success("Training Complete!")
            elif task and task['status'] == 'completed':
                st.success("Last training session completed successfully.")
                st.metric("Final Accuracy", "98.5%", "+1.2%")
            else:
                st.info("No active training session.")
                render_placeholder_chart()
        else:
            render_placeholder_chart()

def render_placeholder_chart():
    st.caption("Start training to see live metrics.")
    # Empty chart frame
    df = pd.DataFrame({"Epoch": [], "Loss": []})
    fig = px.line(df, x="Epoch", y="Loss", title="Waiting for data...")
    fig.update_layout(xaxis_range=[0, 10], yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def render_scaling_tab():
    st.header("Dynamic Scaling Configuration")
    st.markdown("Configure how the system adapts to workload demands.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Auto-Scaling Policies")
        enable_scaling = st.toggle("Enable Auto-Scaling", value=True)
        
        scaling_mode = st.selectbox(
            "Scaling Mode",
            ["Conservative (Save Cost)", "Balanced", "Aggressive (Max Performance)"],
            index=1,
            disabled=not enable_scaling
        )
        
        st.markdown("### Thresholds")
        cpu_threshold = st.slider("CPU Scale-Up Threshold (%)", 50, 95, 80, disabled=not enable_scaling)
        mem_threshold = st.slider("Memory Scale-Up Threshold (%)", 50, 95, 75, disabled=not enable_scaling)
        
    with col2:
        st.subheader("Resource Limits")
        max_workers = st.number_input("Max Worker Nodes", 1, 100, 10)
        min_workers = st.number_input("Min Worker Nodes", 1, 10, 1)
        
        st.markdown("### Current State")
        c1, c2 = st.columns(2)
        c1.metric("Active Workers", "4")
        c2.metric("Current Load", "65%")
        
        st.markdown("#### Scaling History")
        # Mock history chart
        history_data = pd.DataFrame({
            "Time": ["10:00", "10:15", "10:30", "10:45", "11:00"],
            "Workers": [2, 4, 8, 6, 4]
        })
        fig = px.bar(history_data, x="Time", y="Workers", title="Worker Node Allocation")
        st.plotly_chart(fig, use_container_width=True)

def mock_training_task(epochs):
    import time
    time.sleep(5) # Simulate setup
    return "Model saved to models/v1.pt"
