# =============================================================================
# Avalanche eDNA - Production Dockerfile (Multi-Stage Build)
# =============================================================================
# Stage 1: Builder — installs Python deps in isolated env
# Stage 2: BLAST Builder — compiles/copies NCBI BLAST binaries
# Stage 3: Runtime — minimal image with only what's needed at inference time

# ---- Stage 1: Python dependency builder -----------------------------------------------
FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04 AS python-builder

LABEL maintainer="Avalanche eDNA Team"
LABEL description="eDNA Biodiversity Pipeline - Python build stage"

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
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /build

# Copy requirements files first for better layer caching
COPY requirements/requirements_core.txt requirements_core.txt
COPY requirements.txt requirements.txt

# Install PyTorch with CUDA 12.2 support first (largest dependency, best cached separately)
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
        torch==2.2.0+cu122 \
        torchvision==0.17.0+cu122 \
        torchaudio==2.2.0+cu122 \
        --index-url https://download.pytorch.org/whl/cu122

# Install all remaining Python packages (pipeline + training only; no serving deps)
RUN pip3 install --no-cache-dir \
    transformers>=4.38.0 \
    accelerate>=0.27.0 \
    biopython>=1.83 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    umap-learn>=0.5.6 \
    hdbscan>=0.8.33 \
    faiss-gpu>=1.7.4 \
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
    pysam>=0.22.0 \
    cutadapt>=4.6

# ---- Stage 2: BLAST binary installer ------------------------------------------------
FROM ubuntu:22.04 AS blast-builder

ENV BLAST_VERSION=2.15.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and extract BLAST+ binaries for Linux
RUN wget -q "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/${BLAST_VERSION}/ncbi-blast-${BLAST_VERSION}+-x64-linux.tar.gz" \
    -O /tmp/blast.tar.gz && \
    tar -xzf /tmp/blast.tar.gz -C /opt/ && \
    mv /opt/ncbi-blast-${BLAST_VERSION}+ /opt/blast && \
    rm /tmp/blast.tar.gz

# ---- Stage 3: SRA Toolkit installer -------------------------------------------------
FROM ubuntu:22.04 AS sra-builder

ENV SRA_VERSION=3.0.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/${SRA_VERSION}/sratoolkit.${SRA_VERSION}-ubuntu64.tar.gz" \
    -O /tmp/sra.tar.gz && \
    tar -xzf /tmp/sra.tar.gz -C /opt/ && \
    mv /opt/sratoolkit.${SRA_VERSION}-ubuntu64 /opt/sratoolkit && \
    rm /tmp/sra.tar.gz

# ---- Stage 4: Production Runtime Image -----------------------------------------------
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04 AS runtime

LABEL maintainer="Avalanche eDNA Team"
LABEL org.opencontainers.image.source="https://github.com/FaisalTabrez/Avalanche_eDNA"
LABEL org.opencontainers.image.description="Avalanche eDNA Biodiversity Assessment Pipeline"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Application environment
    EDNA_ENV=production \
    # PATH for bioinformatics tools
    PATH="/opt/blast/bin:/opt/sratoolkit/bin:/usr/local/bin:${PATH}" \
    # Hugging Face model cache
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    # CUDA memory management
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Runtime system libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgomp1 \
    libhdf5-103 \
    libssl3 \
    zlib1g \
    libbz2-1.0 \
    liblzma5 \
    libffi8 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Update-alternatives for python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy built Python packages from builder stage
COPY --from=python-builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy BLAST+ binaries
COPY --from=blast-builder /opt/blast /opt/blast

# Copy SRA Toolkit binaries
COPY --from=sra-builder /opt/sratoolkit /opt/sratoolkit

# Create non-root application user for security
RUN groupadd --gid 1001 edna && \
    useradd --uid 1001 --gid edna --shell /bin/bash --create-home edna

WORKDIR /app

# Copy project source code (ordered by change frequency for cache efficiency)
COPY --chown=edna:edna config/ ./config/
COPY --chown=edna:edna src/ ./src/
COPY --chown=edna:edna scripts/ ./scripts/
COPY --chown=edna:edna tests/ ./tests/

# Create persistent volume directories
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/reference \
    /app/data/output \
    /app/models/trained \
    /app/checkpoints \
    /app/model_registry \
    /app/dataset_memory \
    /app/logs \
    /app/analysis_outputs/datasets \
    /app/analysis_outputs/runs \
    /app/reference/pr2 \
    /app/reference/silva \
    /app/reference/combined \
    /app/reference/indices \
    /app/.cache/huggingface && \
    chown -R edna:edna /app

# Declare volumes for large, persistent data
VOLUME ["/app/data", "/app/models", "/app/reference", "/app/analysis_outputs", "/app/.cache"]

USER edna

# Health check: verify Python and core imports are intact
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD python3 -c "import torch, transformers, biopython; print('ok')" || exit 1

# Default: run the end-to-end data processing pipeline.
# Override at runtime, e.g.:
#   docker run ... avalanche-edna python3 scripts/train_model.py --help
CMD ["python3", "scripts/run_pipeline.py", \
     "--input", "/app/data/raw", \
     "--output", "/app/data/output"]
