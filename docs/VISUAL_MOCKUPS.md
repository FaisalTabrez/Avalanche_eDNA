# Visual Mockups - Unified Workflow System

## Main Workflow Hub Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│  🧬 Analysis & Training Hub                              [User] [Help] │
│  🟢 Current: Marine Analysis - 67% complete                            │
└────────────────────────────────────────────────────────────────────────┘
┌──────────────┬─────────────────────────────────────────────────────────┐
│              │                                                         │
│ Workflow     │  [Selected Step Content Rendered Here]                 │
│ Steps        │                                                         │
│              │                                                         │
│ ○ 1. Dataset │  ┌───────────────────────────────────────────────────┐ │
│              │  │  Step 1: Select Dataset                            │ │
│ → 2. Config  │  │                                                    │ │
│              │  │  [Upload] [Existing] [SRA]                         │ │
│ ○ 3. Execute │  │                                                    │ │
│              │  │  [File uploader component]                         │ │
│ ○ 4. Results │  │                                                    │ │
│              │  │  [Next: Configure →]                               │ │
│              │  └───────────────────────────────────────────────────┘ │
│              │                                                         │
└──────────────┴─────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│ 📋 Active Tasks (Collapsible)                    [Expand] [Clear All]  │
│ ├─ 🟢 Marine Analysis (Running - 67%)            [Pause] [Stop] [Logs] │
│ ├─ 🟡 Deep Sea Training (Paused - 45%)          [Resume] [Stop] [Logs] │
│ └─ ✅ Soil Dataset (Completed)              [View Results] [Re-run] ✗  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Dataset Selection

```
┌────────────────────────────────────────────────────────────────────────┐
│  Step 1: Select Dataset                                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [📤 Upload File] [💾 Existing Datasets] [🌐 Download from SRA]       │
│  ════════════════════════════════════════════════════════════════      │
│                                                                        │
│  📤 Upload File                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Drag and drop file here, or click to browse                    │ │
│  │                                                                  │ │
│  │  Supported: FASTA, FASTQ, GenBank, Swiss-Prot, EMBL, GZ         │ │
│  │  Max size: 10 GB                                                 │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  📋 Recent Datasets:                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ marine_sample_2025.fasta  |  45,120 seqs  |  Used 2 hrs ago  [→]│ │
│  │ deep_sea_data.fasta       |  23,456 seqs  |  Used 1 day ago  [→]│ │
│  │ soil_microbiome.fasta     |  78,901 seqs  |  Used 3 days ago [→]│ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ℹ️ Quick Tips:                                                        │
│  • For files >1GB, consider using Fast Mode in configuration          │
│  • SRA downloads are cached for 7 days                                │
│  • You can compare multiple datasets in batch mode                    │
│                                                                        │
│                                      [Next: Configure →]               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Step 2: Unified Configuration

```
┌────────────────────────────────────────────────────────────────────────┐
│  Step 2: Configure Analysis & Training                                │
├────────────────────────────────────────────────────────────────────────┤
│  ✓ Dataset: marine_sample_2025.fasta (45,120 sequences)               │
│                                                                        │
│  Configuration Preset:                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  ▼ Full eDNA Pipeline (Complete analysis + training)            │ │
│  │                                                                  │ │
│  │  Quick Analysis (Fast scan)                                     │ │
│  │  Full eDNA Pipeline (Complete analysis + training)        ← ✓   │ │
│  │  Training Only                                                  │ │
│  │  Custom Configuration                                           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  Estimated Runtime: 15-20 minutes  |  Resources: 6.2 GB Memory        │
│                                                                        │
│  [View Preset Details ▼]                                              │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Preset includes:                                                 │ │
│  │ • Full quality analysis with filtering                           │ │
│  │ • Diversity metrics (alpha, beta, rarefaction)                   │ │
│  │ • Taxonomy classification with BLAST                             │ │
│  │ • Novelty detection                                              │ │
│  │ • Contrastive model training (50 epochs)                         │ │
│  │ • Dynamic scaling with auto-configuration                        │ │
│  │ • Complete visualization suite                                   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  Advanced Options (Optional):                                          │
│  ┌─ 📊 Analysis Settings ────────────────────────────────────────────┐│
│  │ Max Sequences: [0 = all]  Quality Threshold: [20]                ││
│  │ ☑ Quality Analysis  ☑ Diversity  ☑ Taxonomy  ☑ Novelty           ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                        │
│  ┌─ 🧬 Model Settings (Collapsible) ─────────────────────────────────┐│
│  │ Mode: [Train New Model ▼]                                        ││
│  │ Architecture: Contrastive Learning  |  Epochs: 50  |  Batch: 32  ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                        │
│  ┌─ ⚡ Dynamic Scaling (Collapsible) ─────────────────────────────────┐│
│  │ ☑ Auto-configure (Recommended)                                   ││
│  │ Estimated: 50-200 clusters  |  Memory: 2-4 GB                    ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                        │
│  [💾 Save as Custom Preset]                                           │
│                                                                        │
│  [← Back]                              [Start Execution →]            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Step 3: Execute & Monitor (Live Progress)

```
┌────────────────────────────────────────────────────────────────────────┐
│  Step 3: Execute & Monitor                                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Pipeline: Full eDNA Analysis + Model Training                         │
│  Dataset: marine_sample_2025.fasta (45,120 sequences)                 │
│  Started: 2025-11-27 10:30:15  |  Elapsed: 11m 23s  |  ETA: 5m 12s    │
│                                                                        │
│  Overall Progress: ████████████████░░░░░░░ 67%                        │
│                                                                        │
│  ┌─ Pipeline Stages ─────────────────────────────────────────────────┐│
│  │                                                                    ││
│  │  ✓ 1. Preprocessing                         [100%]  (2m 15s)      ││
│  │     └─ 45,120 sequences loaded and validated                      ││
│  │                                                                    ││
│  │  → 2. Embedding Generation                  [67%]   (5m 8s)       ││
│  │     ┌─ Current Status ─────────────────────────────────────────┐  ││
│  │     │ Model: DNABERT-2 (3B parameters) - Loaded ✓              │  ││
│  │     │ Device: CUDA (NVIDIA RTX 3090)                            │  ││
│  │     │ Processed: 30,230 / 45,120 sequences                      │  ││
│  │     │ Batch: 945 / 1,410 (batch_size=32)                        │  ││
│  │     │ Speed: 120 seq/sec                                        │  ││
│  │     │ GPU Memory: 6.2 / 8.0 GB  ██████░░                        │  ││
│  │     │                                                            │  ││
│  │     │ [Live Log - Last 5 lines]                                 │  ││
│  │     │ > Processing batch 945/1410...                            │  ││
│  │     │ > Embedding shape: (32, 768)                              │  ││
│  │     │ > Saving checkpoint at batch 900...                       │  ││
│  │     │ > Checkpoint saved: /checkpoints/batch_900.pt             │  ││
│  │     │ > Resuming embedding generation...                        │  ││
│  │     │                             [View Full Logs]  [Download]  │  ││
│  │     └──────────────────────────────────────────────────────────┘  ││
│  │                                                                    ││
│  │  ⏳ 3. Model Training                      [0%]    (Waiting...)    ││
│  │     └─ Queued: Contrastive learning (50 epochs)                   ││
│  │                                                                    ││
│  │  ⏳ 4. Clustering & Analysis                [0%]    (Waiting...)    ││
│  │                                                                    ││
│  │  ⏳ 5. Results Generation                   [0%]    (Waiting...)    ││
│  │                                                                    ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                        │
│  ┌─ Dynamic Scaling Status ──────────────────────────────────────────┐│
│  │  Current Config: 127 clusters (auto-scaled from 50)               ││
│  │  Memory Usage: 2.3 GB / 8.0 GB  ██████░░░░░░░░░░░░░░░░            ││
│  │  Buffer: Hybrid (exemplars: 1,270, recent: 380, uncertainty: 127) ││
│  │  Adaptations: 3 scaling events                                    ││
│  │                                                                    ││
│  │  [Timeline] [Memory Graph] [Buffer Evolution] [Config History]    ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                        │
│  [⏸️ Pause] [⏹️ Stop] [📋 Export Progress Report]                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Step 4: Results Dashboard

```
┌────────────────────────────────────────────────────────────────────────┐
│  Step 4: Results Dashboard                                            │
├────────────────────────────────────────────────────────────────────────┤
│  Analysis: marine_sample_2025                                          │
│  Completed: 2025-11-27 10:47:23  |  Duration: 17m 8s                  │
│                                                                        │
│  ┌─ Quick Summary (Always Visible) ──────────────────────────────────┐│
│  │                                                                    ││
│  │  📊 45,120 sequences  |  127 clusters  |  23 novel taxa           ││
│  │  🧬 Mean: 324bp  |  GC: 47.3%  |  Quality: 8.5/10                 ││
│  │  ⚡ Scaling: 3 adaptations  |  Peak: 187 clusters                 ││
│  │  🤖 Model: 50 epochs  |  Val Accuracy: 94.2%                      ││
│  │                                                                    ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                        │
│  [📊 Overview] [🧬 Diversity] [🔬 Taxonomy] [📈 Quality] [🤖 Model] [⚡] │
│  ════════════════════════════════════════════════════════════════      │
│                                                                        │
│  📊 Overview Tab                                                       │
│                                                                        │
│  Executive Summary:                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Analyzed 45,120 high-quality sequences from marine sample.       │ │
│  │ Identified 127 distinct clusters with 94.2% classification       │ │
│  │ confidence. Detected 23 potentially novel taxa requiring further │ │
│  │ investigation. Overall diversity index: 3.47 (Shannon).          │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  Key Metrics:                                                          │
│  ┌─────────────┬──────────────┬──────────────┬──────────────┐        │
│  │ Sequences   │ Clusters     │ Novel Taxa   │ Avg Quality  │        │
│  │ 45,120      │ 127          │ 23           │ 8.5/10       │        │
│  │             │              │              │              │        │
│  └─────────────┴──────────────┴──────────────┴──────────────┘        │
│                                                                        │
│  Top Organisms:                                                        │
│  ┌──────────────────────────────────────────┬─────────┬──────────┐   │
│  │ Organism                                 │ Count   │ %        │   │
│  ├──────────────────────────────────────────┼─────────┼──────────┤   │
│  │ Prochlorococcus marinus                  │ 8,234   │ 18.3%    │   │
│  │ Synechococcus sp.                        │ 6,891   │ 15.3%    │   │
│  │ Pelagibacter ubique                      │ 5,432   │ 12.0%    │   │
│  │ Candidatus Actinomarina minuta           │ 3,221   │ 7.1%     │   │
│  │ [Novel Clade A]                          │ 2,109   │ 4.7%     │   │
│  │ ...                                      │ ...     │ ...      │   │
│  └──────────────────────────────────────────┴─────────┴──────────┘   │
│                                                                        │
│  Key Visualizations:                                                   │
│  ┌────────────────────┐ ┌────────────────────┐ ┌──────────────────┐  │
│  │ [Diversity Chart]  │ │ [Taxonomy Tree]    │ │ [Quality Dist]   │  │
│  │                    │ │                    │ │                  │  │
│  │   [Graph]          │ │   [Interactive]    │ │   [Histogram]    │  │
│  │                    │ │                    │ │                  │  │
│  └────────────────────┘ └────────────────────┘ └──────────────────┘  │
│                                                                        │
│  [🔄 New Analysis] [📥 Export Full Report] [🔗 Share Results]         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Task Manager Panel (Bottom - Expanded View)

```
┌────────────────────────────────────────────────────────────────────────┐
│ 📋 Active Tasks                        [Collapse ▲] [Clear Completed]  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ ┌─ Task 1: Marine Analysis ─────────────────────────────────────────┐ │
│ │ 🟢 Running - 67% complete                                         │ │
│ │ ████████████████░░░░░░░░                                          │ │
│ │                                                                   │ │
│ │ Stage: Embedding Generation (batch 945/1410)                      │ │
│ │ Started: 10:30:15  |  Elapsed: 11m 23s  |  ETA: 5m 12s            │ │
│ │ Resources: GPU 6.2/8GB  |  CPU: 45%  |  Memory: 2.3/8GB           │ │
│ │                                                                   │ │
│ │ [📊 View Details] [⏸️ Pause] [⏹️ Stop] [📋 Logs] [📥 Export]       │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│ ┌─ Task 2: Deep Sea Training ───────────────────────────────────────┐ │
│ │ 🟡 Paused - 45% complete                                          │ │
│ │ ███████████░░░░░░░░░░░░░                                          │ │
│ │                                                                   │ │
│ │ Stage: Model Training (epoch 23/50)                               │ │
│ │ Paused at: 10:35:42  |  Runtime: 5m 27s                           │ │
│ │                                                                   │ │
│ │ [📊 View Details] [▶️ Resume] [⏹️ Stop] [📋 Logs]                  │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│ ┌─ Task 3: Soil Dataset ─────────────────────────────────────────────┐│
│ │ ✅ Completed - 100%                                               ││
│ │ █████████████████████████                                         ││
│ │                                                                   ││
│ │ Finished: 10:15:23  |  Total time: 14m 37s                        ││
│ │ Results: 78,901 sequences, 203 clusters, 41 novel taxa            ││
│ │                                                                   ││
│ │ [📊 View Results] [🔄 Re-run] [📥 Export] [🗑️ Remove]              ││
│ └───────────────────────────────────────────────────────────────────┘│
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Mobile Responsive Design (Conceptual)

```
┌──────────────────────┐
│ 🧬 Workflow Hub      │
│ [≡ Menu] [User]      │
├──────────────────────┤
│ Current: Marine (67%)│
│ █████████████░░░░░░░ │
├──────────────────────┤
│                      │
│ Steps:               │
│ [✓ Dataset]          │
│ [→ Configure]        │
│ [○ Execute]          │
│ [○ Results]          │
│                      │
│ ┌──────────────────┐ │
│ │ [Step Content]   │ │
│ │                  │ │
│ │ Responsive       │ │
│ │ Stacked Layout   │ │
│ │                  │ │
│ └──────────────────┘ │
│                      │
│ [Tasks ▼]            │
│                      │
└──────────────────────┘
```

---

## Color Scheme & Design Tokens

```
Primary Colors:
- Primary Blue: #1f77b4 (actions, links)
- Success Green: #2ca02c (completed, success)
- Warning Yellow: #ff7f0e (paused, warnings)
- Error Red: #d62728 (errors, critical)
- Running Green: #00ff00 (active tasks)

Neutral Colors:
- Background: #ffffff (light mode) / #1e1e1e (dark mode)
- Surface: #f8f9fa / #2d2d2d
- Border: #dee2e6 / #444444
- Text Primary: #212529 / #ffffff
- Text Secondary: #6c757d / #a0a0a0

Semantic Colors:
- Info: #17a2b8
- Dataset: #8c564b
- Model: #9467bd
- Analysis: #7f7f7f
- Scaling: #bcbd22

Typography:
- Heading: 'Inter', sans-serif (18-24px, bold)
- Body: 'Inter', sans-serif (14-16px, regular)
- Code: 'Fira Code', monospace (13px)
- Metrics: 'Inter', sans-serif (16-20px, semibold)

Spacing:
- xs: 4px
- sm: 8px
- md: 16px
- lg: 24px
- xl: 32px
- xxl: 48px

Border Radius:
- Small: 4px (buttons, inputs)
- Medium: 8px (cards, panels)
- Large: 12px (modals, major sections)
```

---

## Icons & Visual Language

```
Status Indicators:
• 🟢 Running/Active
• 🟡 Paused/Warning
• ✅ Completed/Success
• ❌ Failed/Error
• ⏳ Queued/Waiting
• ⚪ Not Started

Action Icons:
• ▶️ Start/Resume
• ⏸️ Pause
• ⏹️ Stop
• 🔄 Refresh/Reload
• 📊 View/Details
• 📋 Logs
• 📥 Export/Download
• 🔗 Share
• 🗑️ Delete
• ⚙️ Settings

Category Icons:
• 📂 Dataset
• 🧬 Model
• 📈 Analysis
• ⚡ Scaling
• 🔬 Taxonomy
• 📊 Results
```

This completes the visual mockup guide!
