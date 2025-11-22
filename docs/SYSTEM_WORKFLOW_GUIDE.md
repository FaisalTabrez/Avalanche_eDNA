# Avalanche eDNA System: Comprehensive Workflow Guide

This document provides a detailed overview of every feature, underlying process, and a step-by-step guide to using the Avalanche eDNA biodiversity assessment system.

---

## 1. System Overview

Avalanche is an end-to-end system for analyzing environmental DNA (eDNA) sequences. It combines a high-performance analysis pipeline with a modern interactive dashboard and a centralized data management system.

### Core Components
1.  **Analysis Pipeline (`scripts/run_pipeline.py`)**: A modular 6-step pipeline that transforms raw sequences into biological insights.
2.  **Data Management System (`src/ui/data_manager.py`)**: A centralized engine that organizes runs, manages embeddings, and provides unified access across the UI.
3.  **Interactive Dashboard (Streamlit)**: A web-based interface for exploring results, visualizing clusters, and managing data.
4.  **Embedding Engine**: A sophisticated system for generating, compressing, indexing, and searching sequence embeddings across datasets.

---

## 2. Core Workflow Processes

The analysis pipeline consists of 6 sequential steps. Each step can be configured in `config/config.yaml`.

### Step 1: Preprocessing
*   **Goal**: Clean raw data for analysis.
*   **Process**:
    *   **Quality Filtering**: Removes sequences with low quality scores (default Q20).
    *   **Length Filtering**: Discards sequences too short (<50bp) or too long (>500bp).
    *   **Dereplication**: Collapses identical sequences into unique variants to save computation.
*   **Output**: Cleaned FASTA file.

### Step 2: Embedding Generation
*   **Goal**: Convert biological sequences into numerical vectors (embeddings).
*   **Process**:
    *   **Model**: Uses a Nucleotide Transformer (or configured model) to read DNA sequences.
    *   **Vectorization**: Maps each sequence to a high-dimensional vector (e.g., 256 dimensions) capturing semantic biological meaning.
    *   **Storage**: Saves embeddings as `.npy` (NumPy) files for efficiency.
*   **Output**: `embeddings.npy` (Shape: [N_sequences, Dimensions]).

### Step 3: Clustering
*   **Goal**: Group similar sequences into Operational Taxonomic Units (OTUs) or Amplicon Sequence Variants (ASVs).
*   **Process**:
    *   **Algorithm**: Uses HDBSCAN (density-based) or K-Means on the generated embeddings.
    *   **Dimensionality Reduction**: Optionally applies UMAP/PCA to reduce dimensions before clustering.
*   **Output**: Cluster assignments and visualization coordinates.

### Step 4: Taxonomy Assignment
*   **Goal**: Identify the species/genus of each sequence.
*   **Process**:
    *   **Hybrid Approach**: Combines multiple methods for accuracy.
    *   **KNN/FAISS**: Searches the reference database for nearest neighbors in embedding space.
    *   **BLAST**: (Optional) Performs traditional alignment against reference databases (nt/SILVA).
    *   **Consensus**: Merges predictions to assign the most probable taxonomy.
*   **Output**: `taxonomy_predictions.csv` with lineage (Kingdom -> Species) and confidence scores.

### Step 5: Novelty Detection
*   **Goal**: Identify sequences that don't match known references (potential new species).
*   **Process**:
    *   **Distance Thresholding**: Calculates distance to nearest reference neighbors.
    *   **Outlier Detection**: Flags sequences significantly far from known clusters.
*   **Output**: List of potential novel taxa candidates.

### Step 6: Visualization & Reporting
*   **Goal**: Present results in human-readable formats.
*   **Process**:
    *   **Plots**: Generates static PNGs (Taxonomic distribution, Alpha/Beta diversity).
    *   **Interactive**: Creates HTML plots (3D cluster views).
    *   **Report**: Compiles a text summary (`analysis_report.txt`) and JSON metadata.
*   **Output**: `visualizations/` folder and summary reports.

---

## 3. Data Management System

Avalanche uses a centralized system to manage the lifecycle of data, ensuring scalability for multi-dataset workflows.

### Directory Structure
```
AvalancheData/
├── runs/                       # All analysis runs
│   └── Dataset_Name/
│       └── YYYY-MM-DD_HH-MM-SS/
│           ├── embeddings.npy  # Sequence vectors
│           ├── pipeline_results.json
│           └── taxonomy/       # Predictions
├── datasets/                   # Raw input datasets
└── results/                    # Exported results
```

### Embedding Management Features
*   **Consolidation**: Merges embeddings from multiple runs into a single "Master Reference" (`scripts/consolidate_embeddings.py`).
*   **Compression**: Converts large `.npy` files to compressed `.npz` format (~50% savings) (`scripts/compress_embeddings.py`).
*   **Search**: Finds similar sequences across all historical runs using FAISS indexing (`scripts/search_reference.py`).
*   **Versioning**: Tracks which model version generated which embeddings (`scripts/manage_embedding_versions.py`).

---

## 4. User Interface Guide

The Streamlit dashboard provides four main views:

### 1. Home Page
*   **Overview**: System status and quick stats.
*   **Recent Runs**: One-click access to the 6 most recent analyses.
*   **Quick Actions**: Launch new analyses or jump to results.

### 2. Runs Browser (`Runs` page)
*   **Run Selector**: A powerful filterable list of all historical runs.
*   **Metadata View**: See run configuration, date, and status.
*   **File Explorer**: Browse and download specific output files (CSV, PNG, JSON) for any selected run.

### 3. Results Viewer (`Results` page)
*   **Interactive Analysis**: Deep dive into a specific run.
*   **Tabs**:
    *   *Summary*: High-level metrics (Total reads, Unique ASVs).
    *   *Taxonomy*: Bar charts and sunburst plots of species distribution.
    *   *Clustering*: 2D/3D scatter plots of sequence clusters.
    *   *Novelty*: List of potential new species with confidence scores.

### 4. Taxonomy Explorer (`Taxonomy` page)
*   **Detailed Predictions**: View the full taxonomy table.
*   **Filtering**: Filter by confidence score or specific taxa (e.g., show only "Chordata").
*   **Export**: Download filtered tables for external use.

---

## 5. Step-by-Step Usage Guide

### Phase 1: Installation & Setup
1.  **Clone Repository**:
    ```bash
    git clone https://github.com/FaisalTabrez/Avalanche_eDNA.git
    cd Avalanche_eDNA
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure**:
    Edit `config/config.yaml` to set your preferences (threads, GPU usage, thresholds).

### Phase 2: Running an Analysis
1.  **Prepare Data**: Place your FASTA/FASTQ files in `data/raw/`.
2.  **Run Pipeline**:
    ```bash
    # Basic run
    python scripts/run_pipeline.py --input data/raw/mysample.fasta --output AvalancheData/runs/MyProject/Run1

    # Run with custom model training
    python scripts/run_pipeline.py --input data/raw/mysample.fasta --output ... --train-model
    ```

### Phase 3: Using the Dashboard
1.  **Launch Server**:
    ```bash
    # Windows
    scripts\windows\start_report_system.bat
    
    # Linux/Mac
    streamlit run src/visualization/dashboard.py
    ```
2.  **Navigate**: Open `http://localhost:8501` (or port 8504).
3.  **Select Run**: Go to "Runs" or use the "Recent Runs" on Home to pick your analysis.
4.  **Explore**: Use the "Results" page to visualize taxonomy and clusters.

### Phase 4: Managing Embeddings (Multi-Dataset Workflow)
1.  **Consolidate**: After running multiple datasets, build a master index:
    ```bash
    python scripts/consolidate_embeddings.py --incremental
    ```
2.  **Compress**: Save space by compressing old runs:
    ```bash
    python scripts/compress_embeddings.py --older-than 30 --execute --delete-original
    ```
3.  **Search**: Find a sequence in your historical data:
    ```bash
    python scripts/search_reference.py --run MyProject/Run1 --seq-idx 5 --top-k 20
    ```

### Phase 5: Advanced - Model Versioning
1.  **Register Version**:
    ```bash
    python scripts/manage_embedding_versions.py register --version v1.0 --model "nt-transformer-500m"
    ```
2.  **Tag Runs**:
    ```bash
    python scripts/manage_embedding_versions.py auto-tag --version v1.0
    ```
3.  **Compare**:
    ```bash
    python scripts/manage_embedding_versions.py compare v1.0 v2.0
    ```

---

## 6. Key Scripts Reference

| Script | Purpose |
| :--- | :--- |
| `scripts/run_pipeline.py` | **Main Entry Point**. Runs the full analysis pipeline. |
| `src/visualization/dashboard.py` | **UI Entry Point**. Launches the Streamlit web app. |
| `scripts/consolidate_embeddings.py` | Builds/updates the master reference index. |
| `scripts/compress_embeddings.py` | Compresses storage (NPY -> NPZ). |
| `scripts/search_reference.py` | Searches for similar sequences across datasets. |
| `scripts/manage_embedding_versions.py` | Manages model versions and run tagging. |
| `scripts/build_blast_db.py` | (Optional) Builds local BLAST databases. |
