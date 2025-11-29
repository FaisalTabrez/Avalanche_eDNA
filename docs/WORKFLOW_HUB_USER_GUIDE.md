# Workflow Hub User Guide

## Quick Start

1. **Access the Workflow Hub**
   - Open http://localhost:8504
   - Click "🧬 Workflow Hub" in the sidebar

2. **Follow the 4-Step Process**
   - Step 1: Select Dataset
   - Step 2: Configure
   - Step 3: Execute & Monitor
   - Step 4: View Results

---

## Step-by-Step Guide

### Step 1: Select Dataset

You have 3 options for selecting a dataset:

#### Option A: Upload File
1. Click the "📤 Upload File" tab
2. Drag and drop or click to browse for your file
3. Supported formats: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL (.gz compressed supported)
4. Max file size: 10GB (recommended <1GB for web upload)
5. Enter a dataset name
6. Click "Use This File"

#### Option B: Use Existing Dataset
1. Click the "💾 Existing Datasets" tab
2. Browse the list of previously uploaded datasets
3. Select a dataset from the dropdown
4. Click "Use This Dataset"

#### Option C: Download from SRA
1. Click the "🌐 Download from SRA" tab
2. Enter an SRA accession (e.g., SRR12345678, SRX31101225)
3. Click "📥 Download"
4. Wait for download to complete
5. Dataset will be automatically selected

**Tips:**
- Recently uploaded/downloaded datasets appear at the top
- File size and modification date are shown
- For very large files, consider using command-line tools

---

### Step 2: Configure Analysis

Choose a preset or customize your analysis:

#### Presets:

**⚡ Quick Analysis** (2-5 minutes, 1-2 GB)
- Fast scan for basic statistics
- Quality metrics only
- No model training
- Perfect for: Quick data validation

**🧬 Full eDNA Pipeline** (15-30 minutes, 4-8 GB) ⭐ Recommended
- Complete analysis suite
- Model training (50 epochs)
- Dynamic scaling enabled
- Perfect for: Production analysis

**🤖 Training Only** (10-20 minutes, 3-6 GB)
- Focus on model training
- 100 epochs
- Basic quality analysis
- Perfect for: Model development

**⚙️ Custom** (Varies)
- Full manual control
- Configure every option
- Perfect for: Advanced users

#### Advanced Options (Optional)

Click "Show Advanced Options" to customize:

**Analysis Settings:**
- Max sequences (0 = all)
- Quality threshold (0-40)
- Enable/disable: Quality, Diversity, Taxonomy, Novelty

**Model Settings:**
- Mode: None, Use Existing, Train New
- Epochs (1-500)
- Batch size (16, 32, 64, 128)
- Learning rate (0.0001-0.01)
- Architecture (Contrastive, Supervised)

**Dynamic Scaling:**
- Enable/disable scaling
- Auto-configure (recommended)
- Manual: Min/max clusters

---

### Step 3: Execute & Monitor

Watch your analysis run in real-time:

#### What You'll See:

**Overall Progress**
- Progress bar (0-100%)
- Current stage
- Elapsed time
- Estimated time remaining

**Current Stage Details**
- Stage name (e.g., "Generating embeddings")
- Detailed message
- Sub-progress if available

**Resource Usage** (Real-time)
- CPU usage (%)
- Memory usage (GB)
- GPU memory (GB) if applicable

**Live Log**
- Last few log lines
- Timestamp for each entry
- Auto-scrolling

#### Controls:

- **⏸️ Pause**: Temporarily pause execution (can resume later)
- **⏹️ Stop**: Stop execution permanently
- **Auto-refresh**: Page updates every 2 seconds automatically

**What Happens:**
- Analysis runs in background process
- You can navigate to other pages (task continues)
- Progress is saved (survives app restart)
- When complete, automatically moves to results

---

### Step 4: View Results

Explore your analysis results:

#### Quick Summary (Always Visible)
- Total sequences analyzed
- Clusters identified
- Novel taxa detected
- Model accuracy

#### Tabs:

**📊 Overview**
- Executive summary
- Key metrics grid
- Top organisms table
- Main visualizations

**🧬 Diversity**
- Alpha diversity (Shannon, Simpson, Richness)
- Beta diversity (Bray-Curtis, Jaccard)
- Rarefaction curves
- Species accumulation

**🔬 Taxonomy**
- Taxonomic distribution (pie chart)
- Classification by phylum
- Novel taxa list
- Detailed tables

**📈 Quality**
- Mean/min/max quality scores
- Quality distribution histogram
- Filtering statistics
- Sequences filtered

**🤖 Model**
- Validation accuracy
- Training loss curves
- Epoch-by-epoch progress
- Model size and training time

**⚡ Scaling**
- Cluster evolution over time
- Scaling events timeline
- Memory savings
- Buffer configuration history

#### Actions:
- **🔄 New Analysis**: Start fresh with new dataset
- **📥 Export Report**: Download complete report (coming soon)
- **🔗 Share Results**: Share via link (coming soon)

---

## Task Panel (Bottom of Screen)

### What It Shows:
- All active, paused, and completed tasks
- Real-time progress for running tasks
- Task status (🟢 running, 🟡 paused, ✅ completed, ❌ failed)

### Task Card Information:
- Task name and dataset
- Progress bar and percentage
- Current stage
- Elapsed time and ETA
- Resource usage

### Actions Per Task:
- **📊 View**: Jump to results (completed tasks)
- **⏸️ Pause**: Pause execution (running tasks)
- **▶️ Resume**: Resume execution (paused tasks)
- **⏹️ Stop**: Stop execution (running/paused tasks)
- **🗑️ Remove**: Delete task (completed/failed tasks)
- **📋 Logs**: View detailed logs (all tasks)

### Panel Controls:
- **▲ Collapse / ▼ Expand**: Hide/show task details
- **🗑️ Clear N Completed**: Remove all finished tasks

### Key Feature:
**Tasks persist across navigation!** You can:
- Start an analysis in the workflow hub
- Navigate to other pages
- Come back later
- Task is still visible and running
- No more lost processes!

---

## Tips & Tricks

### For Best Performance:
1. Use "Quick Analysis" preset first to validate data
2. For large datasets (>1M sequences), set max sequences limit
3. Enable dynamic scaling for memory efficiency
4. Monitor resource usage during execution

### Troubleshooting:
- **Upload fails**: File too large or network timeout → Try smaller file
- **Task stuck**: Check logs for errors → May need to stop and restart
- **Results missing**: Wait for task to complete → Check task panel
- **App slow**: Too many completed tasks → Clear completed tasks

### Advanced Usage:
1. **Compare datasets**: Run multiple analyses, keep tasks in panel
2. **Experiment with configs**: Try different presets on same dataset
3. **Monitor resources**: Check CPU/GPU usage to optimize batch size
4. **Save configurations**: Note successful configs for future use

---

## Keyboard Shortcuts (Planned)

- `Ctrl+N`: New analysis
- `Ctrl+R`: View results
- `Space`: Pause/resume current task
- `Esc`: Stop current task
- `Tab`: Next step
- `Shift+Tab`: Previous step

---

## FAQ

**Q: Can I run multiple analyses at once?**  
A: Yes! Each analysis is a separate task. All show in the task panel.

**Q: What happens if I close the browser?**  
A: Tasks continue running. When you reopen, all tasks are restored from disk.

**Q: Can I pause and resume later?**  
A: Yes! Use the pause button. Tasks can be resumed later.

**Q: Where are results saved?**  
A: Results are stored in the task state and can be viewed anytime via the task panel.

**Q: Can I export results?**  
A: Export feature coming soon! For now, results are viewable in the dashboard.

**Q: What's the difference between presets?**  
A: Presets are optimized configurations:
- Quick = Fast validation
- Full = Production analysis  
- Training = Model development
- Custom = Full control

**Q: Why is my task not showing progress?**  
A: Check the task panel. If status is "queued", it's waiting to start. If "running", refresh the page.

---

## Getting Help

- **Documentation**: See `docs/UI_UX_REDESIGN_PLAN.md` for architecture
- **Implementation**: See `docs/IMPLEMENTATION_ROADMAP.md` for technical details
- **Issues**: Check task logs for error messages
- **Support**: Contact the development team

---

**Happy Analyzing! 🧬**
