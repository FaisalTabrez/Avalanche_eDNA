# Avalanche eDNA - Directory Structure

## 📁 Project Organization

```
Avalanche_eDNA/
├── .github/                      # GitHub configuration
│   ├── workflows/               # CI/CD workflows
│   └── .pre-commit-config.yaml  # Pre-commit hooks
│
├── config/                      # Configuration files
│   ├── config.yaml             # Application config
│   ├── grafana/                # Grafana dashboards
│   ├── nginx/                  # Nginx configs
│   └── prometheus/             # Prometheus configs
│
├── consolidated_data/          # Analysis outputs
│   ├── datasets/               # Dataset storage
│   ├── results/                # Analysis results
│   └── runs/                   # Pipeline runs
│
├── data/                       # Data storage
│   ├── processed/              # Processed data
│   ├── raw/                    # Raw data files
│   ├── reference/              # Reference embeddings
│   ├── report_storage/         # Report files
│   └── sample/                 # Sample data
│
├── docker/                     # Docker configuration
│   ├── Dockerfile              # Main Dockerfile
│   ├── .dockerignore           # Docker ignore patterns
│   ├── docker-compose.yml      # Development compose
│   └── docker-compose.prod.yml # Production compose
│
├── docs/                       # Documentation
│   ├── guides/                 # Integration guides
│   │   ├── BLAST_INTEGRATION_GUIDE.md
│   │   ├── INTEGRATION_GUIDE.md
│   │   └── SRA_INTEGRATION_SUMMARY.md
│   ├── reports/                # Phase & test reports
│   │   ├── INTEGRATION_SUMMARY.md
│   │   ├── PHASE_2.3_SUMMARY.md
│   │   ├── PHASE_2.4_SUMMARY.md
│   │   ├── PHASE_3_SUMMARY.md
│   │   ├── TEST_REPORT.md
│   │   └── TESTING_SUMMARY.md
│   ├── archive/                # Historical documents
│   │   ├── DRYRUN_ONEOFFS_FULL.md
│   │   ├── ISSUES_AND_SOLUTIONS.md
│   │   └── REORG_DRYRUN.md
│   ├── api_reference.md        # API documentation
│   ├── configuration.md        # Configuration guide
│   ├── installation.md         # Installation guide
│   ├── troubleshooting.md      # Troubleshooting guide
│   ├── user_guide.md           # User guide
│   ├── DEPLOYMENT_ROADMAP.md   # Deployment plan
│   └── SECURITY.md             # Security guidelines
│
├── logs/                       # Application logs
│
├── notebooks/                  # Jupyter notebooks
│
├── reference/                  # Reference databases
│   ├── combined/               # Combined references
│   ├── eukref/                 # EukRef database
│   ├── indices/                # BLAST indices
│   ├── mappings/               # Taxonomy mappings
│   ├── pr2/                    # PR2 database
│   └── silva/                  # SILVA database
│
├── requirements/               # Python dependencies
│   ├── requirements_core.txt           # Core dependencies
│   ├── requirements_report_management.txt  # Report management
│   └── requirements_windows.txt        # Windows-specific
│
├── scripts/                    # Utility scripts
│   ├── analyze_dataset.py
│   ├── backup_database.py
│   ├── build_blast_db.py
│   ├── build_reference_index.py
│   ├── download_data.py
│   ├── download_sra_data.py
│   ├── export_report_to_pdf.py
│   ├── init_database.py        # Database optimization
│   ├── launch_dashboard.py
│   ├── migrate_database.py
│   ├── monitor_database.py
│   ├── prepare_references.py
│   ├── run_demo.py
│   ├── run_pipeline.py
│   ├── sra_integration_example.py
│   ├── startup.py              # Application startup
│   └── validate_platform.py    # Platform validation
│
├── src/                        # Source code
│   ├── analysis/               # Analysis modules
│   ├── api/                    # API endpoints
│   ├── clustering/             # Clustering algorithms
│   ├── database/               # Database layer
│   ├── dashboards/             # Dashboard components
│   ├── models/                 # ML models
│   ├── novelty/                # Novelty detection
│   ├── organism_profiling/     # Organism profiling
│   ├── preprocessing/          # Data preprocessing
│   ├── report_management/      # Report management
│   ├── similarity/             # Similarity analysis
│   ├── utils/                  # Utilities
│   │   ├── cache.py           # Redis caching
│   │   ├── rate_limiting.py   # Rate limiting
│   │   ├── fastapi_integration.py  # FastAPI middleware
│   │   └── logger.py          # Logging utilities
│   └── visualization/          # Visualization tools
│
├── tests/                      # Test suite
│   ├── test_enhanced_taxonomy.py
│   ├── test_phase3_optimizations.py
│   ├── test_system.py
│   └── conftest.py
│
├── CHANGELOG.md                # Version history
├── LICENSE                     # License file
├── README.md                   # Project overview
├── requirements.txt            # Main requirements
├── start_optimized.py          # Optimized startup script
└── streamlit_app.py            # Streamlit application
```

## 📂 Key Directories

### `/docker/`
Contains all Docker-related files for containerized deployment:
- `Dockerfile`: Multi-stage build for production
- `docker-compose.yml`: Development environment (Redis, PostgreSQL, Prometheus, Grafana)
- `docker-compose.prod.yml`: Production configuration

### `/docs/`
All project documentation organized by type:
- **guides/**: Integration and setup guides
- **reports/**: Phase completion and test reports
- **archive/**: Historical documents and dry runs

### `/requirements/`
Python dependencies split by purpose:
- Core ML and analysis dependencies
- Report management system
- Windows-specific packages

### `/src/utils/`
Phase 3 performance optimizations:
- Redis caching with connection pooling
- Token bucket & sliding window rate limiting
- FastAPI middleware integration

### `/scripts/`
Utility scripts for operations:
- Database initialization and optimization
- Application startup with health checks
- Platform validation
- Data processing and downloads

## 🔄 Migration Notes

**Moved Files:**
- Docker files: Root → `docker/`
- Integration guides: Root → `docs/guides/`
- Phase reports: Root → `docs/reports/`
- Archive docs: Root → `docs/archive/`
- Requirements variants: Root → `requirements/`
- Pre-commit config: Root → `.github/`

**Unchanged:**
- `README.md` (root level for GitHub)
- `LICENSE` (root level for GitHub)
- `CHANGELOG.md` (root level for visibility)
- `requirements.txt` (root level for pip)
- `streamlit_app.py` (root level for easy launch)
- `start_optimized.py` (root level for easy launch)

## 🚀 Quick Start

### Using Docker (Recommended)
```bash
cd docker/
docker-compose up -d
```

### Using Python
```bash
pip install -r requirements.txt
python start_optimized.py
# or
streamlit run streamlit_app.py
```

### Platform Validation
```bash
python scripts/validate_platform.py
```

---

*Last Updated: November 22, 2025*  
*Organization: chore/reorg-codebase branch*
