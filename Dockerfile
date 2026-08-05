# =============================================================================
# Avalanche eDNA - Production Dockerfile (Multi-Stage, CPU-only)
# =============================================================================
# Stage 1: python-builder  — installs all Python/ML deps (CPU PyTorch)
# Stage 2: blast-builder   — downloads NCBI BLAST+ Linux binaries
# Stage 3: sra-builder     — downloads SRA Toolkit Linux binaries
# Stage 4: runtime         — slim Ubuntu 22.04, copies from all build stages
# =============================================================================

# ---- Stage 1: Python dependency builder -------------------------------------
FROM python:3.11-slim-bookworm AS python-builder

LABEL maintainer="Avalanche eDNA Team"
LABEL description="eDNA Biodiversity Pipeline - Python build stage (CPU)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libhdf5-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements files first for optimal layer caching
COPY requirements/requirements_core.txt requirements_core.txt
COPY requirements.txt requirements.txt
COPY setup.py setup.py

# Install CPU-only PyTorch first (largest dep — cached as its own layer)
# PyTorch 2.4+ is required for NumPy 2.x compatibility and Transformers >= 4.38
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.4.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Install all remaining Python packages (no CUDA, no serving deps)
RUN pip install --no-cache-dir \
    transformers>=4.38.0 \
    accelerate>=0.27.0 \
    biopython>=1.83 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    umap-learn>=0.5.6 \
    hdbscan>=0.8.33 \
    faiss-cpu>=1.7.4 \
    lancedb>=0.6.0 \
    pyarrow>=14.0.0 \
    dask>=2024.1.0 \
    pyyaml>=6.0.1 \
    click>=8.1.7 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    joblib>=1.3.0 \
    mlflow>=2.10.0 \
    psycopg2-binary>=2.9.9 \
    sqlalchemy>=2.0.0 \
    python-dotenv>=1.0.0 \
    cutadapt>=4.6

# ---- Stage 2: BLAST binary installer ----------------------------------------
FROM ubuntu:22.04 AS blast-builder

ENV BLAST_VERSION=2.16.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and extract NCBI BLAST+ Linux x64 binaries
RUN wget -q \
    "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/${BLAST_VERSION}/ncbi-blast-${BLAST_VERSION}+-x64-linux.tar.gz" \
    -O /tmp/blast.tar.gz && \
    tar -xzf /tmp/blast.tar.gz -C /opt/ && \
    mv /opt/ncbi-blast-${BLAST_VERSION}+ /opt/blast && \
    rm /tmp/blast.tar.gz

# ---- Stage 3: SRA Toolkit installer -----------------------------------------
FROM ubuntu:22.04 AS sra-builder

ENV SRA_VERSION=3.0.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q \
    "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/${SRA_VERSION}/sratoolkit.${SRA_VERSION}-ubuntu64.tar.gz" \
    -O /tmp/sra.tar.gz && \
    tar -xzf /tmp/sra.tar.gz -C /opt/ && \
    mv /opt/sratoolkit.${SRA_VERSION}-ubuntu64 /opt/sratoolkit && \
    rm /tmp/sra.tar.gz

# ---- Stage 4: Runtime image (slim, CPU-only) --------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="Avalanche eDNA Team"
LABEL org.opencontainers.image.source="https://github.com/FaisalTabrez/Avalanche_eDNA"
LABEL org.opencontainers.image.description="Avalanche eDNA Biodiversity Assessment Pipeline (CPU)"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EDNA_ENV=production \
    # Put bioinformatics tool binaries on PATH
    PATH="/opt/blast/bin:/opt/sratoolkit/bin:/usr/local/bin:${PATH}" \
    # HuggingFace model cache directory
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Runtime system libraries (no CUDA, no libgomp for GPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libhdf5-103-1 \
    libssl3 \
    zlib1g \
    libbz2-1.0 \
    liblzma5 \
    libffi8 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built Python site-packages from builder stage
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy BLAST+ binaries
COPY --from=blast-builder /opt/blast /opt/blast

# Copy SRA Toolkit binaries
COPY --from=sra-builder /opt/sratoolkit /opt/sratoolkit

# Create non-root user for security
RUN groupadd --gid 1001 edna && \
    useradd --uid 1001 --gid edna --shell /bin/bash --create-home edna

WORKDIR /app

# Copy source code (least-changing layers first for best cache reuse)
COPY --chown=edna:edna config/    ./config/
COPY --chown=edna:edna src/       ./src/
COPY --chown=edna:edna scripts/   ./scripts/
COPY --chown=edna:edna tests/     ./tests/
COPY --chown=edna:edna setup.py   ./setup.py

# Install the src package in editable mode (removes need for sys.path hacks)
RUN pip install --no-cache-dir -e . --no-deps

# Create all persistent volume mount points and fix ownership
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/reference \
    /app/data/output \
    /app/data/taxdump \
    /app/models/trained \
    /app/checkpoints \
    /app/model_registry \
    /app/dataset_memory \
    /app/logs \
    /app/analysis_outputs/datasets \
    /app/analysis_outputs/runs \
    /app/reference/pr2 \
    /app/reference/silva \
    /app/reference/combined/18S \
    /app/reference/indices/18S \
    /app/.cache/huggingface && \
    chown -R edna:edna /app

# Declare volumes for large, persistent data
VOLUME ["/app/data", "/app/models", "/app/reference", "/app/analysis_outputs", "/app/.cache"]

USER edna

# Health check: verify Python + core scientific imports work
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD python3 -c "import torch, transformers, Bio; print('ok')" || exit 1

# Default entrypoint: run the data processing pipeline.
# Override at runtime:
#   docker run ... avalanche-edna python3 scripts/train_model.py --help
CMD ["python3", "scripts/run_pipeline.py", \
     "--input",  "/app/data/raw", \
     "--output", "/app/data/output"]
