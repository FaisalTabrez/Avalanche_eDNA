# eDNA Report Management System - Complete Implementation Guide

## Overview

The eDNA Report Management System is a comprehensive feature addition to the Avalanche project that provides advanced capabilities for storing, cataloguing, cross-analyzing, and managing eDNA biodiversity assessment reports. This system creates unique organism profiles, performs similarity analysis across reports, and provides interactive dashboards for data exploration.

## 🎯 Key Features Implemented

### 1. **Database Schema & Storage**
- **Comprehensive database schema** with 10+ tables for storing reports, organisms, and analysis results
- **Organism profiles** with unique identification and taxonomic lineage tracking
- **Analysis reports** with complete metadata and processing information
- **Cross-analysis similarity matrices** for comparing reports
- **Environmental context** storage for location, depth, temperature data

### 2. **Organism Identification & Profiling**
- **Unique organism ID generation** based on taxonomic info and sequence signatures
- **Sequence signature generation** using k-mer analysis for organism fingerprinting
- **Taxonomic matching** with fuzzy matching capabilities
- **Novelty assessment** for identifying potential new species
- **Detection history tracking** across multiple analyses

### 3. **Report Storage & Cataloguing**
- **Automated report storage** with organized directory structure (year/month/report)
- **Comprehensive metadata extraction** from analysis results
- **File compression** for large datasets
- **Search and filtering** capabilities
- **Export functionality** in multiple formats (JSON, CSV)

### 4. **Cross-Analysis Similarity Engine**
- **Multi-dimensional similarity calculation** including:
  - Organism overlap (Jaccard, Dice coefficients)
  - Abundance correlation (Cosine, Pearson, Spearman)
  - Taxonomic composition similarity
  - Diversity metric differences
  - Environmental context similarity
- **Batch comparison** for multiple reports
- **Similarity trends analysis** over time

### 5. **Interactive Dashboard**
- **Web-based interface** using Streamlit
- **Report browser** with advanced filtering
- **Report comparison** with visual similarity metrics
- **Organism profile exploration**
- **Trend analysis** and visualizations
- **Real-time similarity analysis**

### 6. **REST API**
- **Complete REST API** with FastAPI
- **Report management endpoints** (CRUD operations)
- **Organism profile endpoints**
- **Similarity analysis endpoints**
- **Search and filtering capabilities**
- **File upload and export**

## 🏗️ Architecture

```
src/
├── database/              # Database schema and management
│   ├── schema.py         # Complete database schema definition
│   ├── models.py         # Data models and serialization
│   ├── manager.py        # Database CRUD operations
│   └── queries.py        # Advanced query engine
├── organism_profiling/    # Organism identification system
│   └── __init__.py       # Organism identification and matching
├── report_management/     # Report storage and cataloguing
│   └── catalogue_manager.py  # Report storage and organization
├── similarity/           # Cross-analysis similarity engine
│   └── cross_analysis_engine.py  # Similarity calculations
├── dashboards/           # Interactive web dashboards
│   └── report_management_dashboard.py  # Streamlit dashboard
└── api/                  # REST API endpoints
    └── report_management_api.py  # FastAPI application
```

## 📊 Database Schema

### Core Tables:
1. **organism_profiles** - Unique organism identification and metadata
2. **datasets** - Dataset information and environmental context
3. **analysis_reports** - Complete analysis results and summaries
4. **sequences** - Individual sequence data with organism linkage
5. **taxonomic_assignments** - Detailed taxonomic classification
6. **clustering_results** - Clustering analysis results
7. **novelty_detections** - Novel taxa detection results
8. **similarity_matrices** - Cross-analysis comparison results
9. **report_comparisons** - Detailed organism-level comparisons
10. **analysis_metadata** - Analysis parameters and system info

## 🚀 Usage Guide

### 1. **Setting Up the System**

```python
from src.database.manager import DatabaseManager
from src.report_management.catalogue_manager import ReportCatalogueManager

# Initialize database (creates schema automatically)
db_manager = DatabaseManager()

# Initialize report catalogue
catalogue_manager = ReportCatalogueManager(db_manager=db_manager)
```

### 2. **Storing Analysis Reports**

```python
# Store a new analysis report
report_id, storage_path = catalogue_manager.store_analysis_report(
    dataset_file_path="path/to/sequences.fasta",
    analysis_results=analysis_results_dict,
    report_name="Deep Sea Sample Analysis",
    environmental_context={
        'collection_location': 'Mariana Trench',
        'depth_meters': 8000,
        'temperature_celsius': 2.1,
        'collection_date': datetime(2025, 9, 15)
    }
)

print(f"Report stored with ID: {report_id}")
```

### 3. **Organism Identification**

```python
from src.organism_profiling import OrganismIdentifier

identifier = OrganismIdentifier(db_manager)

# Identify organism from sequences
organism_profile = identifier.identify_organism(
    sequences=sequence_records,
    taxonomic_assignments=taxonomy_results,
    environmental_context=env_context
)

# Store organism profile
db_manager.store_organism_profile(organism_profile)
```

### 4. **Cross-Analysis Comparison**

```python
from src.similarity.cross_analysis_engine import CrossAnalysisEngine

engine = CrossAnalysisEngine(db_manager)

# Compare two reports
similarity_matrix = engine.compare_reports(report_id_1, report_id_2)

print(f"Overall similarity: {similarity_matrix.similarity_score:.3f}")
print(f"Shared organisms: {similarity_matrix.organism_overlap_count}")
```

### 5. **Running the Dashboard**

```python
# Launch interactive dashboard
from src.dashboards.report_management_dashboard import ReportManagementDashboard

dashboard = ReportManagementDashboard()
dashboard.run()
```

### 6. **Using the REST API**

```bash
# Start the API server
python -m src.api.report_management_api

# API endpoints available at http://localhost:8000
# - GET /reports - List all reports
# - GET /reports/{id} - Get specific report
# - POST /reports/upload - Upload new dataset
# - GET /organisms - List organisms
# - POST /similarity/compare - Compare reports
```

## 📈 Advanced Features

### 1. **Similarity Metrics**

The system calculates multiple similarity metrics:

- **Organism Overlap**: Jaccard similarity, Overlap coefficient, Dice coefficient
- **Abundance Similarity**: Cosine similarity, Pearson/Spearman correlation
- **Taxonomic Similarity**: Multi-level taxonomic comparison with weighted scoring
- **Diversity Similarity**: Shannon/Simpson diversity differences
- **Environmental Similarity**: Geographic distance, depth/temperature differences

### 2. **Query Engine**

Advanced querying capabilities:

```python
from src.database.queries import ReportQueryEngine

query_engine = ReportQueryEngine(db_manager)

# Search organisms
organisms = query_engine.search_organisms(
    query="Bacteria",
    kingdom="Bacteria",
    is_novel=True,
    min_confidence=0.8
)

# Get organism timeline
timeline = query_engine.get_organism_timeline(organism_id)

# Analyze novelty trends
trends = query_engine.get_novelty_trends(time_period_days=90)
```

### 3. **Organism Matching**

Cross-report organism matching:

```python
from src.organism_profiling import OrganismMatcher

matcher = OrganismMatcher(db_manager)

# Find similar organisms
similar = matcher.find_similar_organisms(
    target_organism_id="ORG_ABC123",
    similarity_threshold=0.8
)

# Match organisms across reports
matches = matcher.match_organisms_across_reports(report_1, report_2)
```

## 🔧 Configuration

### Database Configuration
- **Default location**: `data/reports.db`
- **Auto-initialization**: Database and tables created automatically
- **Migration support**: Schema version tracking for future updates

### Storage Configuration
- **Default storage**: `data/report_storage/`
- **Organization**: Automatic year/month/report organization
- **Compression**: Large files (>10MB) automatically compressed
- **Cleanup**: Configurable automatic cleanup of old reports

## 🧪 Testing

The system includes comprehensive test coverage:

```python
# Run tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_database.py
python -m pytest tests/test_organism_profiling.py
python -m pytest tests/test_similarity_engine.py
```

## 🔄 Integration with Existing Avalanche System

### Seamless Integration
- **Extends existing analysis pipeline** without breaking changes
- **Leverages existing dataset analyzer** and analysis components
- **Maintains compatibility** with current data formats and workflows
- **Adds value** through enhanced storage, organization, and comparison

### Integration Points
1. **Post-analysis storage**: Automatically store results after analysis
2. **Organism profiling**: Enhance existing taxonomy assignment
3. **Cross-analysis**: Compare new results with historical data
4. **Dashboard integration**: Unified interface for all functionality

## 📚 API Documentation

### Authentication
Currently uses no authentication. In production, implement:
- API key authentication
- JWT tokens for user sessions
- Role-based access control

### Rate Limiting
Recommended for production:
- 100 requests/minute for general endpoints
- 10 requests/minute for upload endpoints
- 5 requests/minute for heavy analysis endpoints

### Error Handling
Standardized error responses:
```json
{
  "error": "Resource not found",
  "detail": "Report with ID 'RPT_123' not found",
  "timestamp": "2025-09-23T10:30:00Z"
}
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from src.database.manager import DatabaseManager; DatabaseManager()"

# Run dashboard
python -m streamlit run src/dashboards/report_management_dashboard.py

# Run API
python -m uvicorn src.api.report_management_api:app --reload
```

### Production Deployment
- **Database**: Use PostgreSQL for production instead of SQLite
- **Storage**: Configure robust file storage (e.g., AWS S3)
- **API**: Deploy with proper WSGI server (e.g., Gunicorn)
- **Dashboard**: Deploy with Streamlit Cloud or container

## 🔮 Future Enhancements

### Planned Features
1. **Machine Learning Integration**: Automated organism classification
2. **Real-time Analysis**: Live data processing and alerts
3. **Advanced Visualizations**: 3D similarity networks, phylogenetic trees
4. **Collaboration Features**: Multi-user access, sharing, comments
5. **External Integrations**: NCBI, GBIF, other biodiversity databases

### Scalability Improvements
1. **Database optimization**: Indexing, partitioning for large datasets
2. **Caching layer**: Redis for frequently accessed data
3. **Background processing**: Celery for long-running analyses
4. **Microservices**: Split into specialized services for better scaling

## 🎉 Summary

This comprehensive eDNA Report Management System successfully addresses your requirements for:

✅ **Organism Unique Identity**: Advanced organism profiling with unique ID generation
✅ **Report Storage & Cataloguing**: Organized storage with comprehensive metadata
✅ **Cross-Analysis Similarity**: Multi-dimensional similarity comparison engine
✅ **Interactive Interface**: Web dashboard for data exploration and management
✅ **API Integration**: REST API for external system integration
✅ **Trend Analysis**: Time-based analysis of similarities and novelty detection

The system provides a robust foundation for managing and analyzing eDNA biodiversity assessment results, enabling researchers to:
- Track organism detection across multiple studies
- Identify patterns and trends in biodiversity data
- Compare results across different environments and time periods
- Discover potential novel taxa through advanced similarity analysis
- Export and share results with the research community

This implementation significantly enhances the Avalanche project's capabilities and provides a scalable platform for future biodiversity research initiatives.