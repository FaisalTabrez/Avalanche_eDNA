# Implementation Roadmap - Unified Workflow System

## Quick Start Implementation (This Week)

### **Day 1-2: Prototype Core Structure**

#### 1. Create Basic Workflow Hub
**File:** `src/ui/pages/workflow_hub.py`

```python
import streamlit as st
from pathlib import Path
import time

def render():
    """Main unified workflow hub"""
    
    # Initialize session state for workflows
    init_workflow_state()
    
    # Header with current task indicator
    render_header()
    
    # Main layout: Sidebar steps + Content area
    col_steps, col_content = st.columns([1, 3])
    
    with col_steps:
        current_step = render_workflow_steps()
    
    with col_content:
        render_step_content(current_step)
    
    # Bottom task manager panel
    render_task_panel()

def init_workflow_state():
    """Initialize workflow session state"""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'dataset'
    
    if 'workflow_data' not in st.session_state:
        st.session_state.workflow_data = {
            'dataset': None,
            'config': {},
            'task_id': None,
            'results': None
        }
    
    if 'active_tasks' not in st.session_state:
        st.session_state.active_tasks = {}

def render_header():
    """Render page header with current task"""
    st.title("🧬 Analysis & Training Hub")
    
    # Show active task if any
    if st.session_state.active_tasks:
        task = list(st.session_state.active_tasks.values())[0]
        status_emoji = "🟢" if task['status'] == 'running' else "🟡"
        st.info(f"{status_emoji} Current: {task['name']} - {task['progress']*100:.0f}% complete")

def render_workflow_steps():
    """Render workflow navigation sidebar"""
    st.markdown("### Workflow Steps")
    
    steps = [
        ('dataset', '📂 Dataset', st.session_state.workflow_data['dataset'] is not None),
        ('configure', '⚙️ Configure', bool(st.session_state.workflow_data['config'])),
        ('execute', '▶️ Execute', st.session_state.workflow_data['task_id'] is not None),
        ('results', '📊 Results', st.session_state.workflow_data['results'] is not None)
    ]
    
    for step_key, step_label, completed in steps:
        status = "✓" if completed else "○"
        active = "→" if st.session_state.current_step == step_key else " "
        
        if st.button(f"{active} {status} {step_label}", key=f"step_{step_key}",
                    use_container_width=True):
            st.session_state.current_step = step_key
            st.rerun()
    
    return st.session_state.current_step

def render_step_content(step):
    """Render content for current step"""
    if step == 'dataset':
        render_dataset_step()
    elif step == 'configure':
        render_configure_step()
    elif step == 'execute':
        render_execute_step()
    elif step == 'results':
        render_results_step()

def render_dataset_step():
    """Step 1: Dataset Selection"""
    st.header("1. Select Dataset")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "💾 Existing", "🌐 SRA"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose a file", 
                                        type=['fasta', 'fa', 'fastq', 'fq', 'gz'])
        if uploaded_file:
            st.session_state.workflow_data['dataset'] = {
                'type': 'upload',
                'file': uploaded_file,
                'name': uploaded_file.name
            }
            st.success(f"Selected: {uploaded_file.name}")
            
            if st.button("Next: Configure →", type="primary"):
                st.session_state.current_step = 'configure'
                st.rerun()
    
    with tab2:
        datasets_dir = Path("data/datasets")
        if datasets_dir.exists():
            files = list(datasets_dir.glob("*.fasta")) + list(datasets_dir.glob("*.fa"))
            if files:
                selected = st.selectbox("Select dataset", files, 
                                       format_func=lambda x: x.name)
                if st.button("Use this dataset"):
                    st.session_state.workflow_data['dataset'] = {
                        'type': 'existing',
                        'path': str(selected),
                        'name': selected.name
                    }
                    st.session_state.current_step = 'configure'
                    st.rerun()
    
    with tab3:
        st.info("SRA integration - Enter accession or search")
        accession = st.text_input("SRA Accession", placeholder="e.g., SRR1553606")
        if accession and st.button("Download and Use"):
            # Trigger SRA download
            st.session_state.workflow_data['dataset'] = {
                'type': 'sra',
                'accession': accession,
                'name': accession
            }
            st.session_state.current_step = 'configure'
            st.rerun()

def render_configure_step():
    """Step 2: Unified Configuration"""
    st.header("2. Configure Analysis & Training")
    
    if not st.session_state.workflow_data['dataset']:
        st.warning("⚠️ Please select a dataset first")
        if st.button("← Back to Dataset Selection"):
            st.session_state.current_step = 'dataset'
            st.rerun()
        return
    
    st.success(f"Dataset: {st.session_state.workflow_data['dataset']['name']}")
    
    # Configuration Presets
    preset = st.selectbox("Configuration Preset", [
        "Quick Analysis (Fast scan)",
        "Full eDNA Pipeline (Complete analysis + training)",
        "Training Only",
        "Custom Configuration"
    ])
    
    config = {}
    
    if "Custom" in preset:
        # Expandable configuration sections
        with st.expander("📊 Analysis Settings", expanded=True):
            config['analysis_type'] = st.selectbox("Analysis Type", 
                                                   ["Quick", "Standard", "Comprehensive"])
            config['max_sequences'] = st.number_input("Max Sequences (0 = all)", 
                                                      value=0, min_value=0)
            config['enable_quality'] = st.checkbox("Quality Analysis", value=True)
            config['enable_diversity'] = st.checkbox("Diversity Analysis", value=True)
            config['enable_taxonomy'] = st.checkbox("Taxonomy Classification", value=True)
        
        with st.expander("🧬 Model Settings"):
            config['use_model'] = st.checkbox("Enable Model Training/Inference")
            if config['use_model']:
                config['model_mode'] = st.radio("Model Mode", 
                                               ["Use Pre-trained", "Train New", "Fine-tune"])
                if "Train" in config['model_mode'] or "Fine" in config['model_mode']:
                    config['epochs'] = st.number_input("Epochs", value=50, min_value=1)
                    config['batch_size'] = st.number_input("Batch Size", value=32, min_value=2)
        
        with st.expander("⚡ Dynamic Scaling"):
            config['enable_scaling'] = st.checkbox("Enable Dynamic Scaling", value=True)
            if config['enable_scaling']:
                config['auto_scale'] = st.checkbox("Auto-detect (Recommended)", value=True)
    else:
        # Load preset configuration
        config = load_preset_config(preset)
        st.info(f"Using preset: {preset}")
        with st.expander("View Configuration"):
            st.json(config)
    
    st.session_state.workflow_data['config'] = config
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Dataset"):
            st.session_state.current_step = 'dataset'
            st.rerun()
    with col2:
        if st.button("Start Execution →", type="primary"):
            st.session_state.current_step = 'execute'
            st.rerun()

def render_execute_step():
    """Step 3: Execute and Monitor"""
    st.header("3. Execute & Monitor")
    
    if not st.session_state.workflow_data['config']:
        st.warning("⚠️ Please configure the analysis first")
        return
    
    # Start execution button
    if 'task_id' not in st.session_state.workflow_data or \
       st.session_state.workflow_data['task_id'] is None:
        
        if st.button("▶️ Start Pipeline", type="primary"):
            task_id = start_pipeline_task(
                st.session_state.workflow_data['dataset'],
                st.session_state.workflow_data['config']
            )
            st.session_state.workflow_data['task_id'] = task_id
            st.rerun()
    else:
        # Show live progress
        task_id = st.session_state.workflow_data['task_id']
        task = st.session_state.active_tasks.get(task_id)
        
        if task:
            render_live_progress(task)
            
            # Auto-advance to results when complete
            if task['status'] == 'completed':
                st.session_state.workflow_data['results'] = task['results']
                st.balloons()
                time.sleep(2)
                st.session_state.current_step = 'results'
                st.rerun()

def render_results_step():
    """Step 4: View Results"""
    st.header("4. Results Dashboard")
    
    if not st.session_state.workflow_data['results']:
        st.info("No results yet. Complete an analysis first.")
        return
    
    results = st.session_state.workflow_data['results']
    
    # Quick summary (always visible)
    render_results_summary(results)
    
    # Tabbed detailed results
    tabs = st.tabs(["📊 Overview", "🧬 Diversity", "🔬 Taxonomy", 
                    "📈 Quality", "🤖 Model", "⚡ Scaling"])
    
    with tabs[0]:
        render_overview_tab(results)
    with tabs[1]:
        render_diversity_tab(results)
    with tabs[2]:
        render_taxonomy_tab(results)
    with tabs[3]:
        render_quality_tab(results)
    with tabs[4]:
        render_model_tab(results)
    with tabs[5]:
        render_scaling_tab(results)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 New Analysis"):
            st.session_state.workflow_data = {
                'dataset': None, 'config': {}, 'task_id': None, 'results': None
            }
            st.session_state.current_step = 'dataset'
            st.rerun()
    with col2:
        if st.button("📥 Export Results"):
            export_results(results)
    with col3:
        if st.button("🔗 Share Results"):
            generate_share_link(results)

def render_task_panel():
    """Render bottom task manager panel"""
    if not st.session_state.active_tasks:
        return
    
    st.markdown("---")
    
    with st.expander("📋 Active Tasks", expanded=True):
        for task_id, task in st.session_state.active_tasks.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                status_emoji = {
                    'running': '🟢',
                    'paused': '🟡',
                    'completed': '✅',
                    'failed': '❌'
                }.get(task['status'], '⚪')
                
                st.write(f"{status_emoji} **{task['name']}** - {task['progress']*100:.0f}%")
                st.progress(task['progress'])
            
            with col2:
                if task['status'] == 'running':
                    if st.button("⏸️ Pause", key=f"pause_{task_id}"):
                        pause_task(task_id)
                elif task['status'] == 'paused':
                    if st.button("▶️ Resume", key=f"resume_{task_id}"):
                        resume_task(task_id)
            
            with col3:
                if st.button("🗑️ Stop", key=f"stop_{task_id}"):
                    stop_task(task_id)

# Helper functions (stubs for now)
def load_preset_config(preset):
    """Load preset configuration"""
    # TODO: Load from config file
    return {'preset': preset}

def start_pipeline_task(dataset, config):
    """Start a background pipeline task"""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Create task in session state
    st.session_state.active_tasks[task_id] = {
        'id': task_id,
        'name': f"Analysis: {dataset['name']}",
        'status': 'running',
        'progress': 0.0,
        'stage': 'preprocessing',
        'started_at': time.time(),
        'results': None
    }
    
    # TODO: Start actual background process
    return task_id

def render_live_progress(task):
    """Render live progress for a task"""
    st.subheader(f"Pipeline: {task['name']}")
    
    # Overall progress
    st.progress(task['progress'])
    st.write(f"Stage: {task['stage']} - {task['progress']*100:.0f}% complete")
    
    # Stage details
    with st.expander("📋 Pipeline Stages", expanded=True):
        stages = [
            ('preprocessing', 'Preprocessing', 0.2),
            ('embedding', 'Embedding Generation', 0.5),
            ('training', 'Model Training', 0.3),
            ('analysis', 'Analysis', 0.15),
            ('results', 'Results Generation', 0.05)
        ]
        
        for stage_key, stage_name, weight in stages:
            if task['stage'] == stage_key:
                st.write(f"→ **{stage_name}** (in progress)")
            elif task['progress'] > sum(w for _, _, w in stages[:stages.index((stage_key, stage_name, weight))]):
                st.write(f"✓ {stage_name}")
            else:
                st.write(f"⏳ {stage_name}")

def render_results_summary(results):
    """Render quick results summary"""
    st.info("📊 Analysis Complete!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sequences", "45,120")
    with col2:
        st.metric("Clusters", "127")
    with col3:
        st.metric("Novel Taxa", "23")
    with col4:
        st.metric("Quality", "8.5/10")

# Tab render functions (stubs)
def render_overview_tab(results):
    st.write("Overview content")

def render_diversity_tab(results):
    st.write("Diversity analysis")

def render_taxonomy_tab(results):
    st.write("Taxonomy results")

def render_quality_tab(results):
    st.write("Quality metrics")

def render_model_tab(results):
    st.write("Model training results")

def render_scaling_tab(results):
    st.write("Dynamic scaling metrics")

def export_results(results):
    st.success("Export functionality coming soon")

def generate_share_link(results):
    st.success("Share link generation coming soon")

def pause_task(task_id):
    st.session_state.active_tasks[task_id]['status'] = 'paused'

def resume_task(task_id):
    st.session_state.active_tasks[task_id]['status'] = 'running'

def stop_task(task_id):
    del st.session_state.active_tasks[task_id]
```

#### 2. Update Router
**File:** `src/ui/router.py`

```python
# Add to imports
from src.ui.pages import workflow_hub

# Add to render_page function
elif page_key == "workflow_hub":
    workflow_hub.render()

# Add to get_pages_config
{"key": "workflow_hub", "label": "🚀 Workflow Hub"},
```

#### 3. Update __init__.py
**File:** `src/ui/pages/__init__.py`

```python
from . import workflow_hub
```

### **Day 3-4: Background Task Management**

#### Create Task Manager
**File:** `src/ui/utils/task_manager.py`

```python
import multiprocessing as mp
import queue
import time
import json
from pathlib import Path
from typing import Dict, Optional, Callable
import uuid

class TaskManager:
    """Manage background analysis/training tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.processes: Dict[str, mp.Process] = {}
        self.queues: Dict[str, mp.Queue] = {}
    
    def create_task(self, name: str, task_type: str, config: Dict) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            'id': task_id,
            'name': name,
            'type': task_type,
            'status': 'queued',
            'progress': 0.0,
            'stage': 'initialized',
            'config': config,
            'created_at': time.time(),
            'started_at': None,
            'completed_at': None,
            'results': None,
            'error': None
        }
        
        # Create progress queue
        self.queues[task_id] = mp.Queue()
        
        return task_id
    
    def start_task(self, task_id: str, worker_func: Callable):
        """Start executing a task in background"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        task['status'] = 'running'
        task['started_at'] = time.time()
        
        # Create and start process
        process = mp.Process(
            target=worker_func,
            args=(task['config'], self.queues[task_id])
        )
        process.start()
        self.processes[task_id] = process
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get current task status"""
        if task_id not in self.tasks:
            return None
        
        # Check for progress updates from queue
        if task_id in self.queues:
            try:
                while True:
                    update = self.queues[task_id].get_nowait()
                    self.tasks[task_id].update(update)
            except queue.Empty:
                pass
        
        return self.tasks[task_id]
    
    def pause_task(self, task_id: str):
        """Pause a running task"""
        if task_id in self.processes:
            # Send pause signal
            self.tasks[task_id]['status'] = 'paused'
    
    def resume_task(self, task_id: str):
        """Resume a paused task"""
        self.tasks[task_id]['status'] = 'running'
    
    def stop_task(self, task_id: str):
        """Stop and remove a task"""
        if task_id in self.processes:
            self.processes[task_id].terminate()
            self.processes[task_id].join(timeout=5)
            del self.processes[task_id]
        
        if task_id in self.queues:
            del self.queues[task_id]
        
        if task_id in self.tasks:
            del self.tasks[task_id]
    
    def save_state(self, filepath: Path):
        """Save task state to disk"""
        state = {
            task_id: {
                k: v for k, v in task.items()
                if k not in ['results']  # Don't serialize large results
            }
            for task_id, task in self.tasks.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: Path):
        """Load task state from disk"""
        if not filepath.exists():
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        for task_id, task_data in state.items():
            self.tasks[task_id] = task_data

# Worker function example
def analysis_worker(config: Dict, progress_queue: mp.Queue):
    """Background worker for analysis tasks"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    
    try:
        # Send progress updates
        progress_queue.put({'progress': 0.1, 'stage': 'loading_data'})
        
        # Do actual work here
        # ...
        
        progress_queue.put({'progress': 0.5, 'stage': 'analyzing'})
        
        # ...
        
        progress_queue.put({
            'progress': 1.0,
            'stage': 'completed',
            'status': 'completed',
            'results': {'summary': 'Analysis complete'}
        })
    
    except Exception as e:
        progress_queue.put({
            'status': 'failed',
            'error': str(e)
        })
```

### **Day 5: Integration & Testing**

1. Connect workflow_hub to existing analysis code
2. Test background task execution
3. Verify progress updates work
4. Test navigation persistence
5. Handle edge cases (errors, cancellation, etc.)

---

## Week 2-4: Full Implementation

Follow the detailed plan in UI_UX_REDESIGN_PLAN.md

---

## Testing Checklist

- [ ] Can upload dataset and see it in workflow
- [ ] Configuration presets load correctly
- [ ] Can start analysis and see progress
- [ ] Progress updates in real-time
- [ ] Navigate to other pages and back without losing progress
- [ ] Task panel shows all active tasks
- [ ] Can pause/resume/stop tasks
- [ ] Results display correctly after completion
- [ ] Can start multiple tasks simultaneously
- [ ] Tasks persist after browser refresh
- [ ] Errors handled gracefully
- [ ] Memory usage acceptable
- [ ] Performance acceptable for large datasets

---

## Rollout Plan

1. **Week 1**: Deploy prototype to dev environment
2. **Week 2**: Internal testing and feedback
3. **Week 3**: Beta release with feature flag
4. **Week 4**: Full rollout with documentation

---

## Support & Documentation

- User guide: How to use unified workflow
- Developer guide: How to extend with new steps
- Troubleshooting guide: Common issues
- Video tutorial: Walkthrough of complete workflow
