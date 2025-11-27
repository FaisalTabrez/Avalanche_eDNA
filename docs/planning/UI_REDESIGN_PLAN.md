# Avalanche eDNA System - UI Redesign Plan

## 1. Vision
To create a modern, professional, and intuitive "Mission Control" interface for the Avalanche eDNA pipeline. The new UI will move away from a disjointed page structure to a cohesive, workflow-driven application that guides users from raw data to actionable insights.

## 2. Design Philosophy
- **Dashboard-First**: Immediate visibility into system status and recent activities upon login.
- **Wizard-Driven Workflows**: Complex processes (like Analysis pipelines) are broken down into guided steps (Setup -> Configuration -> Execution -> Review).
- **Real-Time Feedback**: Live updates for long-running tasks (training, database migration, analysis) using asynchronous polling.
- **Unified Configuration**: Tightly coupled settings (e.g., Model Training + Dynamic Scaling) are presented together contextually.

## 3. Proposed Architecture

### Directory Structure
```
src/
  ui/
    __init__.py
    app.py                # Main entry point (cleaner than streamlit_app.py)
    styles/
      theme.css           # Custom CSS for polished look
    components/           # Reusable UI widgets
      cards.py            # Metric cards
      charts.py           # Plotly wrappers
      navigation.py       # Custom sidebar
      status.py           # Progress bars & status indicators
    modules/              # Feature-specific logic
      dashboard.py        # Home view
      analysis.py         # The core eDNA workflow
      data_explorer.py    # Result viewer
      model_forge.py      # Training + Scaling
      system_monitor.py   # Resources & Logs
    core/
      state_manager.py    # Robust session_state wrapper
      task_runner.py      # Interface to backend TaskManager
```

## 4. Core Modules (Pages)

### A. Mission Control (Dashboard)
- **System Health**: CPU/Memory usage, Database status.
- **Quick Actions**: "New Analysis", "View Reports".
- **Recent Activity**: List of last 5 runs with status (Success/Fail).
- **Storage Metrics**: Disk usage for datasets and references.

### B. Analysis Hub (The Workflow)
A linear, guided experience replacing the previous "Workflow Hub".
1.  **Input**: File upload (FASTQ/Fasta) or SRA selection.
2.  **Configuration**: Select Reference DB, Parameters (e-value, identity).
3.  **Execution**: Live log streaming, progress bar.
4.  **Results Preview**: Immediate summary of findings.

### C. Data Explorer
- **Taxonomy Viewer**: Interactive Sunburst/Sankey diagrams for species distribution.
- **Sequence Browser**: Searchable table of identified sequences with BLAST stats.
- **Filtering**: Advanced filters (Confidence score, Taxon level).

### D. Model Forge (Training & Scaling)
*Combines previous "Training" and "Dynamic Scaling" pages.*
- **Configuration Panel**:
    - Model Hyperparameters (Epochs, Batch Size).
    - **Dynamic Scaling**: Toggle "Auto-Scale", set Memory Limits, Cluster thresholds.
- **Training Arena**:
    - Real-time loss curves.
    - Live resource usage monitoring (showing Dynamic Scaling in action).
- **Model Registry**: List available models, compare performance.

### E. System Monitor
- **Task Manager**: View/Kill running background tasks.
- **Logs**: Searchable application logs.
- **Configuration Editor**: Visual editor for `config.yaml`.

## 5. Implementation Strategy

### Phase 1: Foundation
- Set up `src/ui/app.py` and `src/ui/core/state_manager.py`.
- Implement the custom Navigation and Layout shell.
- Apply `theme.css`.

### Phase 2: The Dashboard & Monitor
- Build `Mission Control` to display static/mock data.
- Connect `System Monitor` to existing `TaskManager`.

### Phase 3: The Core Workflow
- Re-implement the Analysis pipeline using the "Wizard" pattern.
- Ensure file uploads and SRA downloads work seamlessly.

### Phase 4: Advanced Features
- Build `Model Forge` with the integrated Scaling controls.
- Implement `Data Explorer` with Plotly.

## 6. Next Steps
1.  Approve this plan.
2.  Initialize the new directory structure.
3.  Begin Phase 1 implementation.
