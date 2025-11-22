"""
API endpoints for report management integration.

This module provides REST API endpoints for integrating the report management
system with external applications and automation workflows.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import tempfile
import os
from pathlib import Path

from src.database.manager import DatabaseManager
from src.database.queries import ReportQueryEngine
from src.report_management.catalogue_manager import ReportCatalogueManager
from src.similarity.cross_analysis_engine import CrossAnalysisEngine
from src.organism_profiling import OrganismIdentifier
from src.analysis.dataset_analyzer import DatasetAnalyzer

# Performance optimizations
from src.utils.fastapi_integration import (
    init_fastapi_optimizations,
    fastapi_cached,
    fastapi_rate_limit
)

# Initialize FastAPI app
app = FastAPI(
    title="eDNA Report Management API",
    description="API for managing eDNA analysis reports and organism profiles",
    version="1.0.0"
)

# Initialize performance optimizations
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
init_fastapi_optimizations(
    app,
    enable_cache=True,
    enable_rate_limit=True,
    cache_ttl=600,  # 10 minutes default cache
    rate_limit=100,  # 100 requests per minute default
    rate_window=60,
    redis_url=redis_url
)

# Initialize managers
db_manager = DatabaseManager()
query_engine = ReportQueryEngine(db_manager)
catalogue_manager = ReportCatalogueManager(db_manager=db_manager)
cross_analysis_engine = CrossAnalysisEngine(db_manager)
organism_identifier = OrganismIdentifier(db_manager)


# Pydantic models for API requests/responses
class ReportSummary(BaseModel):
    report_id: str
    report_name: str
    dataset_name: str
    analysis_type: str
    status: str
    created_at: datetime
    shannon_diversity: Optional[float] = None
    novel_candidates_count: Optional[int] = None
    collection_location: Optional[str] = None


class OrganismSummary(BaseModel):
    organism_id: str
    organism_name: Optional[str] = None
    kingdom: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    detection_count: int
    is_novel_candidate: bool
    confidence_score: Optional[float] = None


class SimilarityResult(BaseModel):
    comparison_id: str
    report_id_1: str
    report_id_2: str
    similarity_score: float
    jaccard_similarity: float
    cosine_similarity: float
    organism_overlap_count: int


class AnalysisRequest(BaseModel):
    dataset_name: str
    report_name: Optional[str] = None
    environmental_context: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    search_fields: Optional[List[str]] = None
    limit: int = Field(default=50, ge=1, le=1000)


# Dependency to get database connection
def get_db_manager():
    return db_manager


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "eDNA Report Management API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        stats = db_manager.get_database_statistics()
        return {
            "status": "healthy",
            "database": "connected",
            "total_reports": str(stats.get('total_reports', 0))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Report Management Endpoints

@app.get("/reports", response_model=List[ReportSummary])
@fastapi_cached(ttl=300, key_prefix='list_reports')  # Cache for 5 minutes
@fastapi_rate_limit(limit=50, window=60)  # 50 requests per minute
async def list_reports(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    analysis_type: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None)
):
    """List analysis reports with optional filtering."""
    try:
        filter_kwargs = {'limit': limit, 'offset': offset}
        
        if analysis_type:
            filter_kwargs['analysis_type'] = analysis_type
        
        if start_date and end_date:
            filter_kwargs['date_range'] = (start_date, end_date)
        
        reports = catalogue_manager.list_reports(**filter_kwargs)
        
        return [ReportSummary(**report) for report in reports]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@app.get("/reports/{report_id}", response_model=Dict[str, Any])
@fastapi_cached(ttl=600, key_prefix='get_report')  # Cache for 10 minutes
@fastapi_rate_limit(limit=100, window=60)  # 100 requests per minute
async def get_report(request: Request, report_id: str):
    """Get detailed information about a specific report."""
    try:
        report = catalogue_manager.retrieve_analysis_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {str(e)}")


@app.post("/reports/search", response_model=List[ReportSummary])
@fastapi_rate_limit(limit=30, window=60)  # 30 searches per minute
async def search_reports(request: Request, search_request: SearchRequest):
    """Search reports by text query."""
    try:
        results = catalogue_manager.search_reports(
            query=search_request.query,
            search_fields=search_request.search_fields,
            limit=search_request.limit
        )
        
        return [ReportSummary(**result) for result in results]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/reports/upload", response_model=Dict[str, str])
@fastapi_rate_limit(limit=10, window=3600)  # 10 uploads per hour
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    analysis_request: str = Form(..., description="JSON string of AnalysisRequest")
):
    """Upload and analyze a new dataset."""
    try:
        # Parse analysis request
        request_data = json.loads(analysis_request)
        analysis_req = AnalysisRequest(**request_data)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Analyze dataset
            analyzer = DatasetAnalyzer()
            analysis_results = analyzer.analyze_dataset(
                input_path=temp_file_path,
                output_path=temp_file_path + "_report.txt",
                dataset_name=analysis_req.dataset_name
            )
            
            # Store analysis report
            report_id, storage_path = catalogue_manager.store_analysis_report(
                dataset_file_path=temp_file_path,
                analysis_results=analysis_results,
                report_name=analysis_req.report_name,
                environmental_context=analysis_req.environmental_context
            )
            
            return {
                "report_id": report_id,
                "status": "success",
                "message": "Dataset uploaded and analyzed successfully",
                "storage_path": storage_path
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/reports/{report_id}/export")
async def export_report(report_id: str, format: str = Query("json", regex="^(json|csv)$")):
    """Export a report in the specified format."""
    try:
        export_path = catalogue_manager.export_report(report_id, format)
        
        if not export_path or not os.path.exists(export_path):
            raise HTTPException(status_code=404, detail="Export failed or file not found")
        
        return FileResponse(
            path=export_path,
            filename=f"report_{report_id}.{format}",
            media_type="application/octet-stream"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Organism Profile Endpoints

@app.get("/organisms", response_model=List[OrganismSummary])
@fastapi_cached(ttl=300, key_prefix='list_organisms')  # Cache for 5 minutes
@fastapi_rate_limit(limit=50, window=60)  # 50 requests per minute
async def list_organisms(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    query: Optional[str] = Query(None),
    kingdom: Optional[str] = Query(None),
    is_novel: Optional[bool] = Query(None)
):
    """List organism profiles with optional filtering."""
    try:
        search_kwargs = {'limit': limit}
        
        if query:
            search_kwargs['query'] = query
        if kingdom:
            search_kwargs['kingdom'] = kingdom
        if is_novel is not None:
            search_kwargs['is_novel'] = is_novel
        
        organisms = query_engine.search_organisms(**search_kwargs)
        
        return [
            OrganismSummary(
                organism_id=org.organism_id,
                organism_name=org.organism_name,
                kingdom=org.kingdom,
                genus=org.genus,
                species=org.species,
                detection_count=org.detection_count,
                is_novel_candidate=org.is_novel_candidate,
                confidence_score=org.confidence_score
            )
            for org in organisms
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list organisms: {str(e)}")


@app.get("/organisms/{organism_id}", response_model=Dict[str, Any])
@fastapi_cached(ttl=600, key_prefix='get_organism')  # Cache for 10 minutes
@fastapi_rate_limit(limit=100, window=60)  # 100 requests per minute
async def get_organism(request: Request, organism_id: str):
    """Get detailed information about a specific organism."""
    try:
        organism = db_manager.get_organism_profile(organism_id)
        
        if not organism:
            raise HTTPException(status_code=404, detail="Organism not found")
        
        return organism.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve organism: {str(e)}")


@app.get("/organisms/{organism_id}/timeline", response_model=Dict[str, Any])
async def get_organism_timeline(organism_id: str):
    """Get detection timeline for an organism."""
    try:
        timeline = query_engine.get_organism_timeline(organism_id)
        
        if not timeline:
            raise HTTPException(status_code=404, detail="Organism timeline not found")
        
        return timeline
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")


# Similarity Analysis Endpoints

@app.post("/similarity/compare", response_model=SimilarityResult)
async def compare_reports(report_id_1: str, report_id_2: str):
    """Compare two analysis reports."""
    try:
        similarity_matrix = cross_analysis_engine.compare_reports(report_id_1, report_id_2)
        
        if not similarity_matrix:
            raise HTTPException(status_code=404, detail="Failed to compare reports")
        
        return SimilarityResult(
            comparison_id=similarity_matrix.comparison_id,
            report_id_1=similarity_matrix.report_id_1,
            report_id_2=similarity_matrix.report_id_2,
            similarity_score=similarity_matrix.similarity_score,
            jaccard_similarity=similarity_matrix.jaccard_similarity,
            cosine_similarity=similarity_matrix.cosine_similarity,
            organism_overlap_count=similarity_matrix.organism_overlap_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/similarity/batch-compare", response_model=List[SimilarityResult])
async def batch_compare_reports(report_ids: List[str]):
    """Perform batch comparison of multiple reports."""
    try:
        if len(report_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 reports required for comparison")
        
        similarity_matrices = cross_analysis_engine.batch_compare_reports(report_ids)
        
        return [
            SimilarityResult(
                comparison_id=sm.comparison_id,
                report_id_1=sm.report_id_1,
                report_id_2=sm.report_id_2,
                similarity_score=sm.similarity_score,
                jaccard_similarity=sm.jaccard_similarity,
                cosine_similarity=sm.cosine_similarity,
                organism_overlap_count=sm.organism_overlap_count
            )
            for sm in similarity_matrices
        ]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch comparison failed: {str(e)}")


@app.get("/similarity/trends", response_model=Dict[str, Any])
async def get_similarity_trends(time_period_days: int = Query(90, ge=1, le=365)):
    """Get similarity trends over time."""
    try:
        trends = cross_analysis_engine.get_similarity_trends(time_period_days)
        return trends
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@app.get("/reports/{report_id}/similar", response_model=List[Dict[str, Any]])
async def find_similar_reports(
    report_id: str,
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
    max_results: int = Query(10, ge=1, le=100)
):
    """Find reports similar to the specified report."""
    try:
        similar_reports = cross_analysis_engine.find_similar_reports(
            report_id, similarity_threshold, max_results
        )
        
        return [
            {"report_id": report_id, "similarity_score": score}
            for report_id, score in similar_reports
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar reports: {str(e)}")


# Statistics and Analytics Endpoints

@app.get("/statistics", response_model=Dict[str, Any])
async def get_system_statistics():
    """Get comprehensive system statistics."""
    try:
        stats = db_manager.get_database_statistics()
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.get("/analytics/novelty-trends", response_model=Dict[str, Any])
async def get_novelty_trends(time_period_days: int = Query(90, ge=1, le=365)):
    """Get novelty detection trends."""
    try:
        trends = query_engine.get_novelty_trends(time_period_days)
        return trends
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get novelty trends: {str(e)}")


@app.get("/analytics/taxonomic-diversity", response_model=Dict[str, Any])
async def get_taxonomic_diversity(report_ids: Optional[List[str]] = Query(None)):
    """Get taxonomic diversity analysis."""
    try:
        diversity_analysis = query_engine.get_taxonomic_diversity_analysis(report_ids)
        return diversity_analysis
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get diversity analysis: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "detail": str(exc.detail) if hasattr(exc, 'detail') else str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc.detail) if hasattr(exc, 'detail') else str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)