# Streamlit UI/UX Redesign Plan - Complete Workflow Streamlining

## Current Problems Identified

### 1. **Fragmented User Journey**
- Dataset upload → Configuration → Start Analysis → Navigate to different page → Check progress
- User loses context when switching pages
- No automatic progress tracking in current view
- Background processes invisible when navigating away

### 2. **Disconnected Pages**
- Analysis page (dataset upload + basic analysis)
- Training page (separate model training)
- Progress Updates page (separate monitoring)
- Model Training Dashboard (separate visualization)
- Dynamic Scaling Config (separate configuration)
- Results scattered across multiple pages

### 3. **State Management Issues**
- Process runs in CLI/background when navigating away
- No persistence of running tasks across page navigation
- No real-time updates in current view
- Lost progress visibility

### 4. **Configuration Complexity**
- Analysis settings scattered
- Training settings separate
- Dynamic scaling settings isolated
- No unified configuration view

---

## Proposed Solution: Unified Workflow System

### **Core Concept: Single-Page Workflow with Persistent Task Manager**

Transform the experience into a **unified, step-by-step workflow** where:
1. All configurations happen in one place
2. Progress is always visible (sticky sidebar/bottom panel)
3. Tasks persist across navigation
4. Results are organized hierarchically

---

## New Architecture

### **1. Unified "Analysis & Training Hub" Page**
*Replaces: Analysis, Training, Progress Updates pages*

#### **Layout Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: Analysis & Training Hub                            │
│  Current Task: [Marine Dataset Analysis] [70% Complete]     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────┬───────────────────────────────────────────┐
│  WORKFLOW STEPS │  MAIN CONTENT AREA                        │
│  (Left Sidebar) │                                           │
│                 │                                           │
│  ✓ 1. Dataset   │  [Step content rendered here]            │
│  → 2. Configure │                                           │
│    3. Execute   │  [Live progress/results shown inline]    │
│    4. Results   │                                           │
│                 │                                           │
└─────────────────┴───────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  TASK MONITOR (Collapsible Bottom Panel)                    │
│  🟢 Running: Marine Analysis (70%) | Model Training (45%)   │
│  [Expand for details] [View Logs] [Stop All]                │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Flow

### **Step 1: Dataset Selection**

**UI Elements:**
- **Data Source Tabs**: [Upload File | Select Existing | Download from SRA]
- **Quick Preview**: Show first 5 sequences, file stats
- **Recent Datasets**: Quick access to previously used datasets
- **Batch Selection**: Select multiple datasets for comparison

**New Features:**
- Dataset tagging (marine, soil, air, etc.)
- Metadata input (location, date, environment)
- Automatic format detection with confidence score

---

### **Step 2: Unified Configuration**

**Instead of separate pages, use expandable sections:**

```
📊 Analysis Configuration
├─ Analysis Type: [Quick Scan | Full Analysis | Custom]
├─ Sequence Filters: [Min length | Max length | Quality threshold]
├─ Analysis Modules: ☑ Diversity ☑ Quality ☑ Taxonomy ☑ Novelty
└─ Output Options: [Report format | Visualizations]

🧬 Model Configuration (Optional)
├─ Model Selection: [Pre-trained | Train New | Fine-tune Existing]
├─ If Training:
│   ├─ Architecture: [Contrastive | Transformer | Autoencoder]
│   ├─ Training Params: [Epochs | Batch size | Learning rate]
│   └─ Continual Learning: ☑ Enable ☑ Use EWC ☑ Use LoRA
└─ If Pre-trained: [Select from registry]

⚡ Dynamic Scaling Configuration (Auto-configured by default)
├─ Auto-detect: ☑ Enable (Recommended)
├─ Manual Override:
│   ├─ Cluster Range: [10 - 1000]
│   ├─ Memory Budget: [8 GB]
│   └─ Strategy: [Hierarchical | Flat]
└─ Refinements: ☑ Temperature ☑ Reservoir ☑ Mini-retrieval

💾 Execution Settings
├─ Execution Mode: [Standard | Fast | Comprehensive]
├─ Resource Allocation: [CPU Only | GPU if available]
├─ Save Options: [Embeddings | Models | Reports | All]
└─ Notifications: ☑ On completion ☑ On errors
```

**Key Innovation: Configuration Presets**
- "Quick Analysis" - Fast scan with minimal config
- "Full eDNA Pipeline" - Complete analysis + training
- "Training Only" - Just model training
- "Comparison Mode" - Multiple datasets side-by-side
- "Custom" - Full manual control

---

### **Step 3: Execute & Monitor**

**Unified Execution View:**

```
┌─────────────────────────────────────────────────────┐
│  EXECUTION OVERVIEW                                 │
├─────────────────────────────────────────────────────┤
│  Pipeline: Full eDNA Analysis + Model Training      │
│  Dataset: Marine_Sample_2025.fasta (45,120 seqs)   │
│  Started: 2025-11-27 10:30:15                       │
│  Estimated completion: 15 minutes                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  PIPELINE STAGES (Auto-expanding current stage)     │
├─────────────────────────────────────────────────────┤
│  ✓ 1. Preprocessing [100%] (2 min)                  │
│  → 2. Embedding Generation [67%] (5 min elapsed)    │
│     └─ Loaded: DNABERT-2 (3B params)                │
│     └─ Processed: 30,230 / 45,120 sequences         │
│     └─ Batch 945/1410 (batch_size=32)               │
│     └─ Speed: 120 seq/sec                           │
│     └─ GPU Memory: 6.2 / 8.0 GB                     │
│                                                      │
│     [Live Log Tail - Last 5 lines]                  │
│     > Processing batch 945/1410...                  │
│     > Embedding shape: (32, 768)                    │
│     > Saving checkpoint batch_900...                │
│                                                      │
│  ⏳ 3. Model Training [0%] (Waiting...)              │
│  ⏳ 4. Clustering & Analysis [0%] (Waiting...)       │
│  ⏳ 5. Results Generation [0%] (Waiting...)          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  DYNAMIC SCALING STATUS (Real-time)                 │
├─────────────────────────────────────────────────────┤
│  Current Configuration:                             │
│  • Clusters: 127 (auto-scaled from 50)              │
│  • Memory usage: 2.3 GB / 8.0 GB                    │
│  • Buffer: Hybrid (exemplars: 1,270, recent: 380)   │
│  • Adaptations: 3 scaling events                    │
│                                                      │
│  [View Scaling Timeline] [View Buffer Evolution]    │
└─────────────────────────────────────────────────────┘

[Pause] [Stop] [View Full Logs] [Export Progress]
```

**Key Features:**
- **Auto-expanding sections**: Current stage shows details, others collapsed
- **Live metrics**: Update every 1-2 seconds via WebSocket/polling
- **Inline logs**: Last N lines visible, expandable for full view
- **Resource monitoring**: GPU/CPU/Memory in real-time
- **Checkpoint indicators**: Show when checkpoints are saved
- **Estimated time**: Dynamically updated based on current speed

---

### **Step 4: Results Dashboard**

**Hierarchical Results View:**

```
┌─────────────────────────────────────────────────────┐
│  ANALYSIS RESULTS: Marine_Sample_2025               │
│  Completed: 2025-11-27 10:47:23 (17m 8s)            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  QUICK SUMMARY (Always visible)                     │
├─────────────────────────────────────────────────────┤
│  📊 45,120 sequences | 127 clusters | 23 novel taxa │
│  🧬 Mean length: 324bp | GC: 47.3% | Quality: 8.5/10│
│  ⚡ Dynamic scaling: 3 adaptations | Peak: 187 clust│
│  🎯 Model trained: 50 epochs | Val acc: 94.2%       │
└─────────────────────────────────────────────────────┘

[Tabs: Overview | Diversity | Taxonomy | Quality | Model | Scaling]

TAB: Overview
├─ Executive Summary (text)
├─ Key Metrics (cards)
├─ Top Organisms (table)
└─ Quick Visualizations (3-4 key charts)

TAB: Diversity
├─ Alpha Diversity Metrics
├─ Beta Diversity Comparison
├─ Rarefaction Curves
└─ Diversity Heatmaps

TAB: Taxonomy
├─ Taxonomic Tree (interactive)
├─ Classification Confidence
├─ Novel Taxa Candidates
└─ Phylogenetic Placement

TAB: Quality
├─ Quality Score Distribution
├─ Length Distribution
├─ GC Content Analysis
└─ Sequence Complexity

TAB: Model (if trained)
├─ Training Curves (loss, accuracy)
├─ Confusion Matrix
├─ Embedding Visualization (t-SNE/UMAP)
└─ Model Registry Entry

TAB: Scaling (if enabled)
├─ Scaling Timeline
├─ Buffer Evolution
├─ Memory Usage Over Time
├─ Configuration Diff History
└─ Performance Metrics
```

**Key Features:**
- **Persistent quick summary**: Always visible at top
- **Tabbed interface**: Organized by analysis type
- **Export options**: Per-tab or complete report
- **Comparison mode**: Side-by-side with previous runs
- **Share results**: Generate shareable link/PDF

---

## Persistent Task Manager

### **Concept: Background Task Orchestrator**

**Session State Structure:**
```python
st.session_state.tasks = {
    'task_id_1': {
        'type': 'analysis',
        'dataset': 'Marine_Sample_2025.fasta',
        'status': 'running',  # queued, running, paused, completed, failed
        'progress': 0.67,
        'current_stage': 'embedding_generation',
        'started_at': '2025-11-27 10:30:15',
        'estimated_completion': '2025-11-27 10:47:00',
        'logs': [...],
        'results': {...},
        'process_id': 12345,  # For background process management
        'config': {...}
    },
    'task_id_2': {
        'type': 'training',
        ...
    }
}
```

**Task Manager UI (Bottom Panel):**
```
┌─────────────────────────────────────────────────────┐
│ 🟢 2 Tasks Running | 1 Completed | 0 Failed         │
├─────────────────────────────────────────────────────┤
│ [Expand All] [Collapse All] [Clear Completed]       │
├─────────────────────────────────────────────────────┤
│ ▼ Marine Analysis (Running - 67%)                   │
│   └─ Stage: Embedding Generation | ETA: 8m 15s      │
│   └─ [View Details] [Pause] [Stop] [Logs]           │
│                                                      │
│ ▼ Deep Sea Training (Running - 45%)                 │
│   └─ Epoch: 23/50 | Loss: 0.142 | ETA: 12m 30s      │
│   └─ [View Details] [Pause] [Stop] [Logs]           │
│                                                      │
│ ▸ Soil Dataset (Completed - 100%) ✓                 │
│   └─ Finished: 2025-11-27 10:15:23                  │
│   └─ [View Results] [Re-run] [Delete]               │
└─────────────────────────────────────────────────────┘
```

**Key Features:**
- **Persistent across page navigation**: Task state saved in session
- **Background process management**: Uses Python multiprocessing
- **Real-time updates**: WebSocket or polling (st.experimental_rerun)
- **Task queue**: Queue multiple analyses
- **Auto-cleanup**: Clear old completed tasks
- **Crash recovery**: Save task state to disk, resume on restart

---

## Technical Implementation Plan

### **Phase 1: Core Infrastructure (Week 1)**

1. **Create Unified Workflow Page**
   - `src/ui/pages/workflow_hub.py` - Main unified page
   - Replace analysis.py, training.py, progress_updates.py entry points

2. **Background Task Manager**
   - `src/ui/utils/task_manager.py` - Task orchestration
   - `src/ui/utils/background_runner.py` - Process management
   - Use `multiprocessing.Process` for background execution
   - Implement task serialization for persistence

3. **Enhanced State Management**
   - Extend `src/ui/state.py` with task management
   - Add WebSocket support or efficient polling mechanism
   - Implement task lifecycle (create, run, pause, resume, stop)

### **Phase 2: Workflow Steps (Week 2)**

4. **Step Components**
   - `src/ui/components/step_dataset.py` - Dataset selection
   - `src/ui/components/step_configure.py` - Unified config
   - `src/ui/components/step_execute.py` - Execution monitor
   - `src/ui/components/step_results.py` - Results dashboard

5. **Configuration Presets**
   - `config/workflow_presets.yaml` - Predefined configurations
   - Load/save custom presets
   - Smart defaults based on dataset size

### **Phase 3: Real-time Monitoring (Week 3)**

6. **Live Progress Components**
   - `src/ui/components/live_progress.py` - Real-time metrics
   - `src/ui/components/resource_monitor.py` - CPU/GPU/Memory
   - `src/ui/components/log_viewer.py` - Live log tail
   - Integrate dynamic scaling visualization

7. **Task Persistence**
   - Save task state to SQLite or JSON
   - Resume tasks on app restart
   - Handle crashed/zombie processes

### **Phase 4: Results & Integration (Week 4)**

8. **Unified Results Dashboard**
   - Merge biodiversity_results.py into results component
   - Integrate taxonomy viewer
   - Add model training results
   - Add dynamic scaling visualizations

9. **Navigation Updates**
   - Update router to use new workflow hub
   - Add breadcrumb navigation
   - Maintain backward compatibility for direct page access

10. **Testing & Polish**
    - End-to-end workflow testing
    - Performance optimization
    - Error handling and recovery
    - Documentation

---

## File Structure Changes

### **New Files:**
```
src/ui/
├── pages/
│   ├── workflow_hub.py (NEW - Main unified page)
│   └── legacy/ (OLD - Keep for backward compat)
│       ├── analysis.py
│       ├── training.py
│       └── progress_updates.py
├── components/
│   ├── workflow/
│   │   ├── step_dataset.py (NEW)
│   │   ├── step_configure.py (NEW)
│   │   ├── step_execute.py (NEW)
│   │   └── step_results.py (NEW)
│   ├── task_panel.py (NEW - Bottom task manager)
│   ├── live_progress.py (NEW - Real-time metrics)
│   └── resource_monitor.py (NEW - System resources)
└── utils/
    ├── task_manager.py (NEW - Task orchestration)
    ├── background_runner.py (NEW - Process management)
    └── workflow_presets.py (NEW - Config presets)

config/
└── workflow_presets.yaml (NEW - Preset configurations)

data/
└── tasks/
    └── task_state.db (NEW - Persistent task state)
```

### **Modified Files:**
```
src/ui/
├── router.py - Add workflow_hub route
├── state.py - Enhanced task management
└── pages/
    ├── model_training_dashboard.py - Integrate into results tab
    ├── dynamic_scaling_config.py - Integrate into config step
    ├── biodiversity_results.py - Integrate into results tab
    └── taxonomy.py - Integrate into results tab
```

---

## User Experience Improvements

### **Before (Current):**
```
1. Upload dataset on Analysis page
2. Configure analysis settings
3. Click "Run Analysis"
4. Navigate to "Progress Updates" page
5. Wait while watching progress
6. Navigate to "Biodiversity Results"
7. View some results
8. Navigate to "Taxonomy" for more
9. Go back to upload another dataset
10. Lost track of first analysis progress
```

### **After (New):**
```
1. Open "Workflow Hub"
2. Select dataset (upload/existing/SRA)
3. Choose preset or configure (all in one view)
4. Click "Execute Pipeline"
5. Watch real-time progress inline
6. Results appear in same page when done
7. Bottom panel shows all running tasks
8. Navigate anywhere, tasks persist
9. Return to view any task's progress
10. Compare results side-by-side
```

---

## Advanced Features (Future Enhancements)

### **1. Workflow Templates**
- Save entire workflows (dataset + config + execution)
- Share workflows with team
- Version control for workflows

### **2. Batch Processing**
- Select multiple datasets
- Apply same configuration
- Parallel execution with resource management

### **3. Comparison Mode**
- Side-by-side dataset comparison
- Diff view for configurations
- Statistical comparison of results

### **4. Notifications**
- Browser notifications on completion
- Email alerts for long-running tasks
- Slack/Teams integration

### **5. Collaborative Features**
- Share results with read-only links
- Comments on results
- Export for publications

### **6. Automation**
- Schedule analyses
- Auto-process new SRA datasets
- Trigger training on new data

---

## Migration Strategy

### **Backward Compatibility:**
- Keep existing pages in `legacy/` folder
- Maintain old routes with deprecation warning
- Gradual migration with feature flags

### **User Migration:**
```python
# Feature flag in config
ENABLE_UNIFIED_WORKFLOW = True

# Router logic
if config.get('ENABLE_UNIFIED_WORKFLOW', False):
    # Use new workflow hub
    workflow_hub.render()
else:
    # Use old analysis page
    analysis.render()
```

### **Data Migration:**
- Convert existing run outputs to new format
- Migrate session state structure
- Import historical results into new dashboard

---

## Success Metrics

1. **User Engagement**
   - Time to complete analysis: -50%
   - Page navigation per analysis: -70%
   - User errors/confusion: -80%

2. **Task Visibility**
   - Tasks lost when navigating: 0%
   - Background process awareness: 100%
   - Progress tracking accuracy: >95%

3. **Configuration Efficiency**
   - Time to configure: -60%
   - Preset usage: >70%
   - Configuration errors: -90%

4. **Results Access**
   - Time to find results: -80%
   - Results comparison usage: +200%
   - Export/share actions: +150%

---

## Implementation Priority

### **P0 (Must Have - MVP):**
- Unified workflow page with 4 steps
- Basic task manager (bottom panel)
- Real-time progress monitoring
- Task persistence across navigation

### **P1 (Should Have - v1.0):**
- Configuration presets
- Live resource monitoring
- Enhanced results dashboard
- Batch task support

### **P2 (Nice to Have - v1.1):**
- Workflow templates
- Comparison mode
- Browser notifications
- Advanced filtering/search

### **P3 (Future - v2.0):**
- Collaborative features
- Automation/scheduling
- Integration APIs
- Mobile responsive design

---

## Next Steps

1. **Review & Approve Plan** - Get stakeholder buy-in
2. **Prototype Core UI** - Build basic workflow hub mockup
3. **Implement Task Manager** - Background process orchestration
4. **Migrate One Feature** - Start with analysis workflow
5. **User Testing** - Validate with real users
6. **Iterate & Expand** - Add features based on feedback

---

## Conclusion

This redesign transforms the Avalanche eDNA platform from a **multi-page, fragmented experience** into a **unified, streamlined workflow** that:

✅ Keeps users in context throughout the entire pipeline
✅ Makes background processes visible and manageable
✅ Consolidates configuration into one intuitive interface
✅ Provides real-time progress without page switching
✅ Organizes results hierarchically for easy access
✅ Enables power users with advanced features
✅ Maintains simplicity for quick analyses

The result: **A professional, production-ready bioinformatics platform that researchers will love to use.**
