"""
Data models for eDNA analysis report management system.

This module defines SQLAlchemy-style data models and Pydantic models
for organism profiles, analysis reports, and related entities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import hashlib
import uuid


class AnalysisStatus(Enum):
    """Enumeration of possible analysis statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SequenceType(Enum):
    """Enumeration of sequence types."""
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"
    UNKNOWN = "unknown"


class NoveltyValidationStatus(Enum):
    """Enumeration of novelty validation statuses."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"


class TaskStatus(Enum):
    """Enumeration of Celery task statuses."""
    PENDING = "pending"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REVOKED = "revoked"


@dataclass
class OrganismProfile:
    """
    Data model for organism profiles with unique identification.
    """
    organism_id: str
    organism_name: Optional[str] = None
    taxonomic_lineage: Optional[str] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_name: Optional[str] = None  # 'class' is reserved keyword
    order_name: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    sequence_signature: Optional[str] = None
    first_detected: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    detection_count: int = 1
    confidence_score: Optional[float] = None
    is_novel_candidate: bool = False
    novelty_score: Optional[float] = None
    reference_databases: Optional[List[str]] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = field(default_factory=datetime.now)
    
    @classmethod
    def generate_organism_id(cls, genus: str, species: str, sequence_hash: str) -> str:
        """
        Generate unique organism ID based on taxonomic info and sequence signature.
        
        Args:
            genus: Genus name
            species: Species name
            sequence_hash: Hash of representative sequence
            
        Returns:
            Unique organism ID
        """
        # Create deterministic ID based on taxonomic info and sequence
        combined = f"{genus}_{species}_{sequence_hash[:16]}"
        hash_obj = hashlib.md5(combined.encode())
        return f"ORG_{hash_obj.hexdigest()[:12].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'organism_id': self.organism_id,
            'organism_name': self.organism_name,
            'taxonomic_lineage': self.taxonomic_lineage,
            'kingdom': self.kingdom,
            'phylum': self.phylum,
            'class': self.class_name,
            'order': self.order_name,
            'family': self.family,
            'genus': self.genus,
            'species': self.species,
            'sequence_signature': self.sequence_signature,
            'first_detected': self.first_detected.isoformat() if self.first_detected else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'detection_count': self.detection_count,
            'confidence_score': self.confidence_score,
            'is_novel_candidate': self.is_novel_candidate,
            'novelty_score': self.novelty_score,
            'reference_databases': self.reference_databases,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class DatasetInfo:
    """
    Data model for dataset metadata and environmental context.
    """
    dataset_id: str
    dataset_name: str
    file_path: Optional[str] = None
    file_format: Optional[str] = None
    file_size_mb: Optional[float] = None
    total_sequences: Optional[int] = None
    sequence_type: Optional[SequenceType] = None
    collection_date: Optional[datetime] = None
    collection_location: Optional[str] = None
    depth_meters: Optional[float] = None
    temperature_celsius: Optional[float] = None
    ph_level: Optional[float] = None
    salinity: Optional[float] = None
    environmental_conditions: Optional[Dict[str, Any]] = None
    preprocessing_params: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = field(default_factory=datetime.now)
    
    @classmethod
    def generate_dataset_id(cls, dataset_name: str, file_path: str) -> str:
        """
        Generate unique dataset ID.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to the dataset file
            
        Returns:
            Unique dataset ID
        """
        combined = f"{dataset_name}_{file_path}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(combined.encode())
        return f"DS_{hash_obj.hexdigest()[:12].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'file_path': self.file_path,
            'file_format': self.file_format,
            'file_size_mb': self.file_size_mb,
            'total_sequences': self.total_sequences,
            'sequence_type': self.sequence_type.value if self.sequence_type else None,
            'collection_date': self.collection_date.isoformat() if self.collection_date else None,
            'collection_location': self.collection_location,
            'depth_meters': self.depth_meters,
            'temperature_celsius': self.temperature_celsius,
            'ph_level': self.ph_level,
            'salinity': self.salinity,
            'environmental_conditions': self.environmental_conditions,
            'preprocessing_params': self.preprocessing_params,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class AnalysisReport:
    """
    Data model for complete analysis reports.
    """
    report_id: str
    dataset_id: str
    report_name: Optional[str] = None
    analysis_type: str = "full"
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    processing_time_seconds: Optional[float] = None
    
    # Basic statistics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    mean_length: Optional[float] = None
    median_length: Optional[float] = None
    std_length: Optional[float] = None
    
    # Composition analysis
    sequence_type_detected: Optional[SequenceType] = None
    composition_data: Optional[Dict[str, Any]] = None
    
    # Biodiversity metrics
    shannon_diversity: Optional[float] = None
    simpson_diversity: Optional[float] = None
    evenness: Optional[float] = None
    species_richness: Optional[int] = None
    
    # Clustering results summary
    n_clusters: Optional[int] = None
    silhouette_score: Optional[float] = None
    cluster_coherence: Optional[float] = None
    
    # Taxonomy assignment summary
    sequences_with_taxonomy: Optional[int] = None
    taxonomy_confidence_avg: Optional[float] = None
    
    # Novelty detection summary
    novel_candidates_count: Optional[int] = None
    novel_percentage: Optional[float] = None
    novelty_threshold: Optional[float] = None
    
    # File references
    full_report_path: Optional[str] = None
    results_json_path: Optional[str] = None
    visualizations_dir: Optional[str] = None
    
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = field(default_factory=datetime.now)
    
    @classmethod
    def generate_report_id(cls, dataset_id: str, analysis_type: str) -> str:
        """
        Generate unique report ID.
        
        Args:
            dataset_id: ID of the associated dataset
            analysis_type: Type of analysis performed
            
        Returns:
            Unique report ID
        """
        combined = f"{dataset_id}_{analysis_type}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(combined.encode())
        return f"RPT_{hash_obj.hexdigest()[:12].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'report_id': self.report_id,
            'dataset_id': self.dataset_id,
            'report_name': self.report_name,
            'analysis_type': self.analysis_type,
            'status': self.status.value,
            'processing_time_seconds': self.processing_time_seconds,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'mean_length': self.mean_length,
            'median_length': self.median_length,
            'std_length': self.std_length,
            'sequence_type_detected': self.sequence_type_detected.value if self.sequence_type_detected else None,
            'composition_data': self.composition_data,
            'shannon_diversity': self.shannon_diversity,
            'simpson_diversity': self.simpson_diversity,
            'evenness': self.evenness,
            'species_richness': self.species_richness,
            'n_clusters': self.n_clusters,
            'silhouette_score': self.silhouette_score,
            'cluster_coherence': self.cluster_coherence,
            'sequences_with_taxonomy': self.sequences_with_taxonomy,
            'taxonomy_confidence_avg': self.taxonomy_confidence_avg,
            'novel_candidates_count': self.novel_candidates_count,
            'novel_percentage': self.novel_percentage,
            'novelty_threshold': self.novelty_threshold,
            'full_report_path': self.full_report_path,
            'results_json_path': self.results_json_path,
            'visualizations_dir': self.visualizations_dir,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class SimilarityMatrix:
    """
    Data model for cross-analysis similarity comparisons.
    """
    comparison_id: str
    report_id_1: str
    report_id_2: str
    
    organism_overlap_count: Optional[int] = None
    organism_overlap_percentage: Optional[float] = None
    jaccard_similarity: Optional[float] = None
    cosine_similarity: Optional[float] = None
    
    # Taxonomic composition similarity
    kingdom_similarity: Optional[float] = None
    phylum_similarity: Optional[float] = None
    genus_similarity: Optional[float] = None
    
    # Diversity metric differences
    shannon_diversity_diff: Optional[float] = None
    simpson_diversity_diff: Optional[float] = None
    evenness_diff: Optional[float] = None
    
    # Clustering similarity
    cluster_structure_similarity: Optional[float] = None
    
    # Environmental context similarity
    location_distance_km: Optional[float] = None
    depth_difference_m: Optional[float] = None
    temporal_difference_days: Optional[float] = None
    
    similarity_score: Optional[float] = None  # Overall composite score
    comparison_method: Optional[str] = None
    
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    
    @classmethod
    def generate_comparison_id(cls, report_id_1: str, report_id_2: str) -> str:
        """
        Generate unique comparison ID.
        
        Args:
            report_id_1: First report ID
            report_id_2: Second report ID
            
        Returns:
            Unique comparison ID
        """
        # Sort IDs to ensure consistent comparison IDs regardless of order
        sorted_ids = sorted([report_id_1, report_id_2])
        combined = f"{sorted_ids[0]}_{sorted_ids[1]}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(combined.encode())
        return f"CMP_{hash_obj.hexdigest()[:12].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'comparison_id': self.comparison_id,
            'report_id_1': self.report_id_1,
            'report_id_2': self.report_id_2,
            'organism_overlap_count': self.organism_overlap_count,
            'organism_overlap_percentage': self.organism_overlap_percentage,
            'jaccard_similarity': self.jaccard_similarity,
            'cosine_similarity': self.cosine_similarity,
            'kingdom_similarity': self.kingdom_similarity,
            'phylum_similarity': self.phylum_similarity,
            'genus_similarity': self.genus_similarity,
            'shannon_diversity_diff': self.shannon_diversity_diff,
            'simpson_diversity_diff': self.simpson_diversity_diff,
            'evenness_diff': self.evenness_diff,
            'cluster_structure_similarity': self.cluster_structure_similarity,
            'location_distance_km': self.location_distance_km,
            'depth_difference_m': self.depth_difference_m,
            'temporal_difference_days': self.temporal_difference_days,
            'similarity_score': self.similarity_score,
            'comparison_method': self.comparison_method,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class ReportComparison:
    """
    Data model for detailed organism-level report comparisons.
    """
    comparison_detail_id: str
    comparison_id: str
    organism_id: str
    
    present_in_report_1: bool = False
    present_in_report_2: bool = False
    abundance_report_1: Optional[float] = None
    abundance_report_2: Optional[float] = None
    abundance_ratio: Optional[float] = None
    
    novelty_score_report_1: Optional[float] = None
    novelty_score_report_2: Optional[float] = None
    novelty_status_changed: bool = False
    
    taxonomy_agreement: bool = True
    taxonomy_confidence_1: Optional[float] = None
    taxonomy_confidence_2: Optional[float] = None
    
    notes: Optional[str] = None
    
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    
    @classmethod
    def generate_comparison_detail_id(cls) -> str:
        """Generate unique comparison detail ID."""
        return f"CD_{uuid.uuid4().hex[:12].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'comparison_detail_id': self.comparison_detail_id,
            'comparison_id': self.comparison_id,
            'organism_id': self.organism_id,
            'present_in_report_1': self.present_in_report_1,
            'present_in_report_2': self.present_in_report_2,
            'abundance_report_1': self.abundance_report_1,
            'abundance_report_2': self.abundance_report_2,
            'abundance_ratio': self.abundance_ratio,
            'novelty_score_report_1': self.novelty_score_report_1,
            'novelty_score_report_2': self.novelty_score_report_2,
            'novelty_status_changed': self.novelty_status_changed,
            'taxonomy_agreement': self.taxonomy_agreement,
            'taxonomy_confidence_1': self.taxonomy_confidence_1,
            'taxonomy_confidence_2': self.taxonomy_confidence_2,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


def serialize_to_json(obj: Any) -> str:
    """
    Serialize object to JSON string with datetime handling.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string representation
    """
    def default_serializer(o):
        if isinstance(o, datetime):
            return o.isoformat()
        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        elif isinstance(o, Enum):
            return o.value
        return str(o)
    
    return json.dumps(obj, default=default_serializer, indent=2)


def deserialize_from_json(json_str: str, target_class: type) -> Any:
    """
    Deserialize JSON string to target class instance.
    
    Args:
        json_str: JSON string to deserialize
        target_class: Target class for deserialization
        
    Returns:
        Instance of target class
    """
    data = json.loads(json_str)
    
    # Handle datetime fields
    datetime_fields = ['created_at', 'updated_at', 'first_detected', 'last_updated', 'collection_date']
    for field in datetime_fields:
        if field in data and data[field]:
            data[field] = datetime.fromisoformat(data[field])
    
    # Handle enum fields
    if 'status' in data and hasattr(target_class, '__annotations__'):
        if target_class.__annotations__.get('status') == AnalysisStatus:
            data['status'] = AnalysisStatus(data['status'])
    
    if 'sequence_type' in data and data['sequence_type']:
        data['sequence_type'] = SequenceType(data['sequence_type'])
    
    return target_class(**data)

@dataclass
class AnalysisRun:
    """Model for tracking analysis runs"""
    id: Optional[int] = None
    name: str = ""
    dataset_path: Optional[str] = None
    analysis_type: str = "taxonomic"
    status: str = "pending"
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None
    celery_task_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingRun:
    """Model for tracking model training runs"""
    id: Optional[int] = None
    name: str = ""
    model_type: str = "transformer"
    training_data_path: Optional[str] = None
    status: str = "pending"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    celery_task_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class DownloadJob:
    """Model for tracking download jobs"""
    id: Optional[int] = None
    accession: str = ""
    source: str = "SRA"
    status: str = "pending"
    output_path: Optional[str] = None
    file_size: int = 0
    metadata: Optional[Dict[str, Any]] = None
    celery_task_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
