# Multi-stage Dockerfile for Avalanche eDNA Biodiversity Assessment System
# Optimized for production deployment with security best practices

# Stage 1: Base image with system dependencies
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials for Python packages with C extensions
    gcc \
    g++ \
    make \
    # Bioinformatics tools
    ncbi-blast+ \
    bedtools \
    samtools \
    # Network and utilities
    curl \
    wget \
    git \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r avalanche && useradd -r -g avalanche -m -s /bin/bash avalanche

# Set working directory
WORKDIR /app

# Stage 2: Python dependencies
FROM base AS dependencies

# Copy requirements files
COPY requirements.txt requirements_core.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements_core.txt

# Stage 3: Application
FROM dependencies AS application

# Copy application code
COPY --chown=avalanche:avalanche . .

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/reference \
    /app/data/report_storage \
    /app/analysis_outputs/runs \
    /app/analysis_outputs/datasets \
    /app/analysis_outputs/results \
    /app/logs \
    /app/reference/indices \
    && chown -R avalanche:avalanche /app

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Switch to non-root user
USER avalanche

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# Stage 4: Development image (optional)
FROM application AS development

USER root

# Install development tools
RUN pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

USER avalanche

# Override command for development
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
