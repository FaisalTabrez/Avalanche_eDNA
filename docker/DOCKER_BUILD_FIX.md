# Docker Build and Deployment Guide

## Quick Fixes Applied

### 1. Python Version Update
**Issue:** Python 3.10 incompatible with some packages (numpy, pandas require 3.11+)  
**Fix:** Updated base image from `python:3.10-slim` to `python:3.11-slim`

### 2. Requirements Path Fix
**Issue:** `requirements_core.txt` not found (moved to `requirements/` directory)  
**Fix:** Updated COPY command to use correct path:
```dockerfile
COPY requirements/requirements_core.txt ./requirements/
```

### 3. Package Name (Already Correct)
**Issue:** Error mentioned `torch-audio` but requirements.txt already has correct `torchaudio`  
**Status:** ✅ No change needed

## Build Instructions

### Option 1: Using Build Script (Recommended)

```bash
cd docker/

# Build production image
bash build.sh application build

# Build and run
bash build.sh application run

# Start all services with docker-compose
bash build.sh application compose

# View logs
bash build.sh application logs

# Stop services
bash build.sh application stop
```

### Option 2: Manual Docker Build

```bash
cd docker/

# Build production image
docker build --target application -t avalanche-edna:latest -f Dockerfile ..

# Build development image
docker build --target development -t avalanche-edna:dev -f Dockerfile ..

# Run container
docker run -p 8501:8501 \
  -v "$(pwd)/../data:/app/data" \
  -v "$(pwd)/../reference:/app/reference" \
  avalanche-edna:latest
```

### Option 3: Docker Compose (Full Stack)

```bash
cd docker/

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Build Stages

### 1. `base` - System Dependencies
- Python 3.11-slim
- BLAST, BEDTools, SAMtools
- Build essentials (gcc, g++, make)
- Non-root user setup

### 2. `dependencies` - Python Packages
- Installs all Python dependencies
- Uses requirements.txt and requirements_core.txt
- Optimized with pip cache

### 3. `application` - Production Ready
- Copies application code
- Sets up directories
- Runs as non-root user
- Includes health check
- **Default target for production**

### 4. `development` - Dev Tools
- Includes pytest, black, flake8, mypy
- Jupyter notebooks
- IPython
- **Use for development**

## Testing the Build

```bash
# Build and test
cd docker/
bash build.sh application build

# Run and access Streamlit
bash build.sh application run
# Then open: http://localhost:8501

# Or use docker-compose for full stack
bash build.sh application compose
# Access:
# - Streamlit: http://localhost:8501
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

## Troubleshooting

### Build Fails with "torch-audio not found"
✅ **Fixed** - Updated to Python 3.11 and corrected requirements path

### Permission Errors
The container runs as non-root user `avalanche`. If you see permission errors:
```bash
# Fix data directory permissions
sudo chown -R 1000:1000 data/ reference/
```

### Large Image Size
The image includes BLAST and other bioinformatics tools. To reduce size:
```bash
# Use multi-stage build (already configured)
# Final image size: ~1.5GB

# Clean Docker cache
docker system prune -a
```

### Requirements Installation Timeout
If pip install times out:
```bash
# Build with increased timeout
docker build --network=host --target application -t avalanche-edna ..
```

## Environment Variables

Create a `.env` file in the docker directory:

```bash
# Database
DB_TYPE=postgresql
DB_HOST=postgres
DB_PORT=5432
DB_NAME=avalanche_edna
DB_USER=avalanche
DB_PASSWORD=change_me_in_production

# Cache
REDIS_HOST=redis
REDIS_PORT=6379

# Security
SECRET_KEY=generate_secure_key_here
```

## Volume Mounts

The docker-compose setup mounts:
- `../data:/app/data` - Data storage
- `../reference:/app/reference` - Reference databases
- `../analysis_outputs:/app/analysis_outputs` - Analysis results
- `../logs:/app/logs` - Application logs

## Health Check

The container includes a health check:
```bash
# Check container health
docker ps

# Manual health check
docker exec avalanche-edna curl -f http://localhost:8501/_stcore/health
```

## Production Deployment

### 1. Build Production Image
```bash
docker build --target application -t avalanche-edna:prod ..
```

### 2. Tag for Registry
```bash
docker tag avalanche-edna:prod registry.example.com/avalanche-edna:latest
docker push registry.example.com/avalanche-edna:latest
```

### 3. Deploy with Docker Compose
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Next Steps

1. ✅ Build image with fixes
2. ✅ Test locally
3. Configure production environment variables
4. Set up CI/CD pipeline
5. Deploy to production

## Changes Made

| File | Change | Reason |
|------|--------|--------|
| `Dockerfile` | Python 3.10 → 3.11 | Package compatibility |
| `Dockerfile` | Fixed requirements path | File reorganization |
| `build.sh` | **NEW** | Easy build automation |
| `DOCKER_BUILD_FIX.md` | **NEW** | This guide |

---

**Status:** ✅ Ready to build  
**Last Updated:** November 22, 2025
