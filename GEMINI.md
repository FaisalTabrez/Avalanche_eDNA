# 🧬 Deep-Sea eDNA Biodiversity Assessment System

## Project Overview

This project is a comprehensive, Python-based system for analyzing deep-sea environmental DNA (eDNA) to assess biodiversity. It combines a powerful bioinformatics pipeline with machine learning techniques to process, classify, and visualize eDNA sequence data. The system is designed to handle large datasets, discover novel taxa, and provide an interactive user experience through a Streamlit web application.

**Core Technologies:**
- **Language:** Python
- **Backend/Pipeline:** The core logic is a multi-step pipeline that includes:
    - **Preprocessing:** Quality filtering, adapter trimming, and chimera removal using tools like `cutadapt`.
    - **Sequence Embedding:** Generates vector representations of DNA sequences using a pre-trained Nucleotide Transformer model from Hugging Face (`zhihan1996/DNABERT-2-117M`).
    - **Clustering:** Uses `UMAP` for dimensionality reduction and `HDBSCAN` to group sequences into Operational Taxonomic Units (OTUs).
    - **Taxonomic Assignment:** A hybrid approach using `FAISS` for fast k-NN similarity search against a reference database, followed by a Lowest Common Ancestor (LCA) algorithm. A `BLAST`-based fallback is used for sequences that cannot be confidently classified by the primary method.
    - **Novelty Detection:** Identifies sequences that are significantly different from known reference databases, flagging them as potential new taxa.
- **Frontend:** An interactive web dashboard built with **Streamlit** (`streamlit_app.py`) allows users to upload data, configure and run analyses, and explore results through plots and tables.
- **Data Handling:** `pandas` and `numpy` for data manipulation, with `dask` for parallel processing.
- **Bioinformatics Tools:** `biopython`, `pysam`, and local installations of `BLAST`.

## Building and Running

### 1. Installation

The project uses Python and its dependencies are managed via `pip`.

```bash
# It is recommended to use a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
```
*Note: The system may also require external tools like NCBI BLAST to be installed and configured in `config/config.yaml`.*

### 2. Running the Analysis Pipeline (CLI)

The main analysis pipeline can be executed from the command line. This is suitable for batch processing or running on a server.

```bash
# Run the full pipeline on a sample FASTA file
python scripts/run_pipeline.py --input path/to/your/sequences.fasta --output path/to/results_directory
```

### 3. Launching the Interactive Dashboard

The Streamlit application provides a user-friendly graphical interface for all analysis tasks.

```bash
# Launch the web dashboard
streamlit run streamlit_app.py
```
By default, the application will be available at `http://localhost:8501`.

### 4. Running Tests

The project includes a test suite using `pytest`.

```bash
# Run all tests
pytest tests/
```

## Development Conventions

- **Configuration:** All pipeline settings, file paths, and model parameters are managed centrally in `config/config.yaml`. This allows for easy modification of the system's behavior without changing the source code.
- **Modularity:** The codebase is organized into distinct modules within the `src/` directory, separating concerns like `preprocessing`, `clustering`, `taxonomy`, and `visualization`.
- **Entry Points:** The project has two main entry points: `scripts/run_pipeline.py` for command-line execution and `streamlit_app.py` for the web interface.
- **Data Storage:** The `config.yaml` defines specific directories (`AvalancheData/datasets` and `AvalancheData/runs`) for storing uploaded datasets and analysis results, keeping generated data separate from the source code.
- **SRA Integration:** The system is capable of pulling data directly from the NCBI Sequence Read Archive (SRA), with functionality defined in `scripts/download_sra_data.py` and `src/preprocessing/sra_processor.py`.
