"""
Multi-Source Taxonomy Integration and Custom Taxonomy Support

This module provides functionality to:
- Integrate taxonomic data from multiple sources (SILVA, PR2, UNITE, GTDB)
- Reconcile conflicting taxonomic assignments
- Support custom taxonomy hierarchies
- Handle user-defined taxonomic overrides
- Provide taxonomy source management
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sqlite3
from abc import ABC, abstractmethod

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Import enhanced lineage components
from .enhanced_lineage import (
    EnhancedLineage, TaxonomicRank, EvidenceType, LineageEvidence, 
    TaxonomicName, ExternalReference
)

# Setup logging
logger = logging.getLogger(__name__)

class TaxonomySource(ABC):
    """Abstract base class for taxonomy sources"""
    
    def __init__(self, name: str, description: str, priority: int = 50):
        self.name = name
        self.description = description
        self.priority = priority  # Higher = more trusted (0-100)
        self._loaded = False
    
    @abstractmethod
    def load_data(self) -> bool:
        """Load taxonomy data from source"""
        pass
    
    @abstractmethod
    def get_lineage(self, identifier: str) -> Optional[EnhancedLineage]:
        """Get lineage for a given identifier"""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[EnhancedLineage]:
        """Search for taxa matching query"""
        pass
    
    def is_loaded(self) -> bool:
        """Check if source data is loaded"""
        return self._loaded

@dataclass
class TaxonomyMapping:
    """Mapping between different taxonomy sources"""
    source_name: str
    source_id: str
    target_name: str
    target_id: str
    confidence: float = 1.0
    mapping_type: str = "exact"  # exact, approximate, manual

class SILVATaxonomySource(TaxonomySource):
    """SILVA ribosomal RNA database taxonomy source"""
    
    def __init__(self, silva_file: Optional[str] = None):
        super().__init__(
            name="SILVA",
            description="SILVA ribosomal RNA gene database",
            priority=90
        )
        self.silva_file = Path(silva_file) if silva_file else None
        self._taxonomy_data: Dict[str, EnhancedLineage] = {}
    
    def load_data(self) -> bool:
        """Load SILVA taxonomy data"""
        if not self.silva_file or not self.silva_file.exists():
            logger.warning(f"SILVA taxonomy file not found: {self.silva_file}")
            return False
        
        try:
            # Load SILVA taxonomy (assuming tab-separated format)
            df = pd.read_csv(self.silva_file, sep='\t', header=0)
            
            for _, row in df.iterrows():
                lineage = self._parse_silva_record(row)
                if lineage and lineage.scientific_name:
                    self._taxonomy_data[lineage.scientific_name.lower()] = lineage
            
            self._loaded = True
            logger.info(f"Loaded {len(self._taxonomy_data)} SILVA taxonomy records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading SILVA data: {e}")
            return False
    
    def _parse_silva_record(self, row: pd.Series) -> Optional[EnhancedLineage]:
        """Parse a SILVA taxonomy record"""
        try:
            # Assuming SILVA format with taxonomy string
            silva_id = str(row.get('accession', ''))
            taxonomy_string = str(row.get('taxonomy', ''))
            organism = str(row.get('organism', ''))
            
            if not taxonomy_string:
                return None
            
            # Parse SILVA taxonomy string (e.g., "Bacteria;Proteobacteria;...")
            ranks = taxonomy_string.split(';')
            
            lineage = EnhancedLineage(
                scientific_name=organism or None,
                names=TaxonomicName(scientific_name=organism or ""),
                last_updated=datetime.now()
            )
            
            # Map SILVA ranks to our extended ranks
            silva_rank_mapping = [
                TaxonomicRank.SUPERKINGDOM,  # Domain
                TaxonomicRank.PHYLUM,
                TaxonomicRank.CLASS,
                TaxonomicRank.ORDER,
                TaxonomicRank.FAMILY,
                TaxonomicRank.GENUS,
                TaxonomicRank.SPECIES
            ]
            
            for i, rank_name in enumerate(ranks):
                rank_name = rank_name.strip()
                if rank_name and i < len(silva_rank_mapping):
                    lineage.lineage[silva_rank_mapping[i]] = rank_name
            
            # Add evidence
            evidence = LineageEvidence(
                source=EvidenceType.SILVA,
                confidence=0.9,
                method_details="SILVA database lookup",
                timestamp=datetime.now(),
                metadata={'silva_id': silva_id}
            )
            lineage.add_evidence(evidence)
            
            # Add external reference
            if silva_id:
                silva_ref = ExternalReference(
                    database='SILVA',
                    identifier=silva_id
                )
                lineage.external_refs.append(silva_ref)
            
            return lineage
            
        except Exception as e:
            logger.debug(f"Error parsing SILVA record: {e}")
            return None
    
    def get_lineage(self, identifier: str) -> Optional[EnhancedLineage]:
        """Get lineage by organism name or SILVA ID"""
        if not self._loaded:
            self.load_data()
        
        # Try direct lookup by name
        result = self._taxonomy_data.get(identifier.lower())
        if result:
            return result
        
        # Try partial match
        for name, lineage in self._taxonomy_data.items():
            if identifier.lower() in name:
                return lineage
        
        return None
    
    def search(self, query: str, limit: int = 10) -> List[EnhancedLineage]:
        """Search SILVA taxonomy"""
        if not self._loaded:
            self.load_data()
        
        results = []
        query_lower = query.lower()
        
        for name, lineage in self._taxonomy_data.items():
            if query_lower in name and len(results) < limit:
                results.append(lineage)
        
        return results

class PR2TaxonomySource(TaxonomySource):
    """Protist Ribosomal Reference (PR2) database taxonomy source"""
    
    def __init__(self, pr2_file: Optional[str] = None):
        super().__init__(
            name="PR2",
            description="Protist Ribosomal Reference database",
            priority=85
        )
        self.pr2_file = Path(pr2_file) if pr2_file else None
        self._taxonomy_data: Dict[str, EnhancedLineage] = {}
    
    def load_data(self) -> bool:
        """Load PR2 taxonomy data"""
        if not self.pr2_file or not self.pr2_file.exists():
            logger.warning(f"PR2 taxonomy file not found: {self.pr2_file}")
            return False
        
        try:
            df = pd.read_csv(self.pr2_file, sep='\t', header=0)
            
            for _, row in df.iterrows():
                lineage = self._parse_pr2_record(row)
                if lineage and lineage.scientific_name:
                    self._taxonomy_data[lineage.scientific_name.lower()] = lineage
            
            self._loaded = True
            logger.info(f"Loaded {len(self._taxonomy_data)} PR2 taxonomy records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PR2 data: {e}")
            return False
    
    def _parse_pr2_record(self, row: pd.Series) -> Optional[EnhancedLineage]:
        """Parse a PR2 taxonomy record"""
        try:
            pr2_id = str(row.get('pr2_accession', ''))
            species = str(row.get('species', ''))
            genus = str(row.get('genus', ''))
            family = str(row.get('family', ''))
            order = str(row.get('order', ''))
            class_name = str(row.get('class', ''))
            phylum = str(row.get('phylum', ''))
            kingdom = str(row.get('kingdom', ''))
            
            lineage = EnhancedLineage(
                scientific_name=species or None,
                names=TaxonomicName(scientific_name=species or ""),
                last_updated=datetime.now()
            )
            
            # Map PR2 ranks
            if kingdom and kingdom != 'nan':
                lineage.lineage[TaxonomicRank.KINGDOM] = kingdom
            if phylum and phylum != 'nan':
                lineage.lineage[TaxonomicRank.PHYLUM] = phylum
            if class_name and class_name != 'nan':
                lineage.lineage[TaxonomicRank.CLASS] = class_name
            if order and order != 'nan':
                lineage.lineage[TaxonomicRank.ORDER] = order
            if family and family != 'nan':
                lineage.lineage[TaxonomicRank.FAMILY] = family
            if genus and genus != 'nan':
                lineage.lineage[TaxonomicRank.GENUS] = genus
            if species and species != 'nan':
                lineage.lineage[TaxonomicRank.SPECIES] = species
            
            # Add evidence
            evidence = LineageEvidence(
                source=EvidenceType.PR2,
                confidence=0.9,
                method_details="PR2 database lookup",
                timestamp=datetime.now(),
                metadata={'pr2_id': pr2_id}
            )
            lineage.add_evidence(evidence)
            
            # Add external reference
            if pr2_id:
                pr2_ref = ExternalReference(
                    database='PR2',
                    identifier=pr2_id
                )
                lineage.external_refs.append(pr2_ref)
            
            return lineage
            
        except Exception as e:
            logger.debug(f"Error parsing PR2 record: {e}")
            return None
    
    def get_lineage(self, identifier: str) -> Optional[EnhancedLineage]:
        """Get lineage by organism name or PR2 ID"""
        if not self._loaded:
            self.load_data()
        
        return self._taxonomy_data.get(identifier.lower())
    
    def search(self, query: str, limit: int = 10) -> List[EnhancedLineage]:
        """Search PR2 taxonomy"""
        if not self._loaded:
            self.load_data()
        
        results = []
        query_lower = query.lower()
        
        for name, lineage in self._taxonomy_data.items():
            if query_lower in name and len(results) < limit:
                results.append(lineage)
        
        return results

class CustomTaxonomySource(TaxonomySource):
    """User-defined custom taxonomy source"""
    
    def __init__(self, custom_file: Optional[str] = None, name: str = "Custom"):
        super().__init__(
            name=name,
            description="User-defined custom taxonomy",
            priority=100  # Highest priority for user overrides
        )
        self.custom_file = Path(custom_file) if custom_file else None
        self._taxonomy_data: Dict[str, EnhancedLineage] = {}
        self._overrides: Dict[str, EnhancedLineage] = {}
    
    def load_data(self) -> bool:
        """Load custom taxonomy data"""
        if self.custom_file and self.custom_file.exists():
            return self._load_from_file()
        else:
            self._loaded = True
            return True
    
    def _load_from_file(self) -> bool:
        """Load custom taxonomy from JSON or CSV file"""
        try:
            if self.custom_file.suffix.lower() == '.json':
                return self._load_from_json()
            elif self.custom_file.suffix.lower() == '.csv':
                return self._load_from_csv()
            else:
                logger.warning(f"Unsupported custom taxonomy file format: {self.custom_file.suffix}")
                return False
        except Exception as e:
            logger.error(f"Error loading custom taxonomy: {e}")
            return False
    
    def _load_from_json(self) -> bool:
        """Load from JSON format"""
        with open(self.custom_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data:
            lineage = self._parse_custom_record(record)
            if lineage and lineage.scientific_name:
                self._taxonomy_data[lineage.scientific_name.lower()] = lineage
        
        self._loaded = True
        logger.info(f"Loaded {len(self._taxonomy_data)} custom taxonomy records from JSON")
        return True
    
    def _load_from_csv(self) -> bool:
        """Load from CSV format"""
        df = pd.read_csv(self.custom_file)
        
        for _, row in df.iterrows():
            lineage = self._parse_custom_record(row.to_dict())
            if lineage and lineage.scientific_name:
                self._taxonomy_data[lineage.scientific_name.lower()] = lineage
        
        self._loaded = True
        logger.info(f"Loaded {len(self._taxonomy_data)} custom taxonomy records from CSV")
        return True
    
    def _parse_custom_record(self, record: Dict[str, Any]) -> Optional[EnhancedLineage]:
        """Parse a custom taxonomy record"""
        try:
            scientific_name = record.get('scientific_name', '')
            if not scientific_name:
                return None
            
            lineage = EnhancedLineage(
                scientific_name=scientific_name,
                names=TaxonomicName(
                    scientific_name=scientific_name,
                    common_names=record.get('common_names', []),
                    synonyms=record.get('synonyms', []),
                    authority=record.get('authority')
                ),
                last_updated=datetime.now()
            )
            
            # Map taxonomic ranks
            rank_mapping = {
                'kingdom': TaxonomicRank.KINGDOM,
                'phylum': TaxonomicRank.PHYLUM,
                'class': TaxonomicRank.CLASS,
                'order': TaxonomicRank.ORDER,
                'family': TaxonomicRank.FAMILY,
                'genus': TaxonomicRank.GENUS,
                'species': TaxonomicRank.SPECIES,
                'subspecies': TaxonomicRank.SUBSPECIES,
                'subgenus': TaxonomicRank.SUBGENUS,
                'subfamily': TaxonomicRank.SUBFAMILY,
                'tribe': TaxonomicRank.TRIBE,
                'subtribe': TaxonomicRank.SUBTRIBE
            }
            
            for rank_name, rank_enum in rank_mapping.items():
                if rank_name in record and record[rank_name]:
                    lineage.lineage[rank_enum] = str(record[rank_name])
            
            # Add evidence
            evidence = LineageEvidence(
                source=EvidenceType.CUSTOM_DATABASE,
                confidence=record.get('confidence', 1.0),
                method_details="Custom taxonomy database",
                timestamp=datetime.now(),
                metadata=record.get('metadata', {})
            )
            lineage.add_evidence(evidence)
            
            # Add external references
            if 'external_refs' in record:
                for ref_data in record['external_refs']:
                    ref = ExternalReference(
                        database=ref_data.get('database', ''),
                        identifier=ref_data.get('identifier', ''),
                        url=ref_data.get('url')
                    )
                    lineage.external_refs.append(ref)
            
            return lineage
            
        except Exception as e:
            logger.debug(f"Error parsing custom record: {e}")
            return None
    
    def add_override(self, scientific_name: str, lineage: EnhancedLineage) -> None:
        """Add a taxonomic override"""
        self._overrides[scientific_name.lower()] = lineage
        logger.info(f"Added taxonomic override for {scientific_name}")
    
    def remove_override(self, scientific_name: str) -> bool:
        """Remove a taxonomic override"""
        key = scientific_name.lower()
        if key in self._overrides:
            del self._overrides[key]
            logger.info(f"Removed taxonomic override for {scientific_name}")
            return True
        return False
    
    def get_lineage(self, identifier: str) -> Optional[EnhancedLineage]:
        """Get lineage with overrides taking precedence"""
        if not self._loaded:
            self.load_data()
        
        key = identifier.lower()
        
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]
        
        # Check regular data
        return self._taxonomy_data.get(key)
    
    def search(self, query: str, limit: int = 10) -> List[EnhancedLineage]:
        """Search custom taxonomy"""
        if not self._loaded:
            self.load_data()
        
        results = []
        query_lower = query.lower()
        
        # Search overrides first
        for name, lineage in self._overrides.items():
            if query_lower in name and len(results) < limit:
                results.append(lineage)
        
        # Search regular data
        for name, lineage in self._taxonomy_data.items():
            if query_lower in name and len(results) < limit:
                results.append(lineage)
        
        return results
    
    def export_overrides(self, output_path: str) -> None:
        """Export current overrides to file"""
        output_path = Path(output_path)
        
        overrides_data = []
        for name, lineage in self._overrides.items():
            overrides_data.append(lineage.to_dict())
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(overrides_data, f, indent=2, ensure_ascii=False)
        else:
            # Export as CSV
            if overrides_data:
                df = pd.json_normalize(overrides_data)
                df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(overrides_data)} taxonomic overrides to {output_path}")

class MultiSourceTaxonomyResolver:
    """Resolver that integrates multiple taxonomy sources"""
    
    def __init__(self):
        self.sources: List[TaxonomySource] = []
        self.mappings: List[TaxonomyMapping] = []
        self._cache: Dict[str, EnhancedLineage] = {}
    
    def add_source(self, source: TaxonomySource) -> None:
        """Add a taxonomy source"""
        self.sources.append(source)
        self.sources.sort(key=lambda x: x.priority, reverse=True)  # Sort by priority
        logger.info(f"Added taxonomy source: {source.name} (priority: {source.priority})")
    
    def remove_source(self, source_name: str) -> bool:
        """Remove a taxonomy source"""
        for i, source in enumerate(self.sources):
            if source.name == source_name:
                del self.sources[i]
                logger.info(f"Removed taxonomy source: {source_name}")
                return True
        return False
    
    def add_mapping(self, mapping: TaxonomyMapping) -> None:
        """Add a taxonomy mapping between sources"""
        self.mappings.append(mapping)
        logger.info(f"Added mapping: {mapping.source_name}:{mapping.source_id} -> {mapping.target_name}:{mapping.target_id}")
    
    def resolve_lineage(self, identifier: str, prefer_source: Optional[str] = None) -> Optional[EnhancedLineage]:
        """Resolve lineage from multiple sources with conflict resolution"""
        cache_key = f"{identifier}|{prefer_source or 'auto'}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        candidates = []
        
        # Collect candidates from all sources
        source_order = self.sources.copy()
        
        # If preferred source specified, try it first
        if prefer_source:
            preferred_sources = [s for s in self.sources if s.name == prefer_source]
            other_sources = [s for s in self.sources if s.name != prefer_source]
            source_order = preferred_sources + other_sources
        
        for source in source_order:
            try:
                lineage = source.get_lineage(identifier)
                if lineage:
                    candidates.append((source, lineage))
            except Exception as e:
                logger.debug(f"Error querying {source.name}: {e}")
        
        if not candidates:
            return None
        
        # If only one candidate, return it
        if len(candidates) == 1:
            result = candidates[0][1]
            self._cache[cache_key] = result
            return result
        
        # Resolve conflicts between multiple sources
        resolved_lineage = self._resolve_conflicts(candidates)
        self._cache[cache_key] = resolved_lineage
        
        return resolved_lineage
    
    def _resolve_conflicts(self, candidates: List[Tuple[TaxonomySource, EnhancedLineage]]) -> EnhancedLineage:
        """Resolve conflicts between multiple lineage candidates"""
        if len(candidates) == 1:
            return candidates[0][1]
        
        # Start with the highest priority source as base
        base_source, base_lineage = candidates[0]
        merged_lineage = EnhancedLineage(
            taxid=base_lineage.taxid,
            scientific_name=base_lineage.scientific_name,
            names=TaxonomicName(
                scientific_name=base_lineage.names.scientific_name,
                common_names=base_lineage.names.common_names.copy(),
                synonyms=base_lineage.names.synonyms.copy(),
                authority=base_lineage.names.authority
            ),
            lineage=base_lineage.lineage.copy(),
            metadata=base_lineage.metadata.copy(),
            last_updated=datetime.now()
        )
        
        # Merge evidence from all sources
        all_evidence = []
        all_external_refs = []
        
        for source, lineage in candidates:
            all_evidence.extend(lineage.evidence)
            all_external_refs.extend(lineage.external_refs)
            
            # Merge common names and synonyms
            for name in lineage.names.common_names:
                if name not in merged_lineage.names.common_names:
                    merged_lineage.names.common_names.append(name)
            
            for synonym in lineage.names.synonyms:
                if synonym not in merged_lineage.names.synonyms:
                    merged_lineage.names.synonyms.append(synonym)
            
            # Conflict resolution for lineage ranks
            for rank, name in lineage.lineage.items():
                if name and (rank not in merged_lineage.lineage or not merged_lineage.lineage[rank]):
                    merged_lineage.lineage[rank] = name
                elif (rank in merged_lineage.lineage and merged_lineage.lineage[rank] 
                      and name and merged_lineage.lineage[rank] != name):
                    # Handle conflicts: prefer higher priority source
                    if source.priority > base_source.priority:
                        merged_lineage.lineage[rank] = name
                        # Add conflict metadata
                        merged_lineage.metadata[f'conflict_{rank.rank_name}'] = {
                            'sources': [base_source.name, source.name],
                            'values': [merged_lineage.lineage[rank], name],
                            'resolution': f'Preferred {source.name} (higher priority)'
                        }
        
        merged_lineage.evidence = all_evidence
        merged_lineage.external_refs = all_external_refs
        merged_lineage._update_confidence()
        
        # Add reconciliation evidence
        reconciliation_evidence = LineageEvidence(
            source=EvidenceType.MANUAL_CURATION,  # Closest match for multi-source reconciliation
            confidence=merged_lineage.overall_confidence,
            method_details=f"Multi-source reconciliation from {len(candidates)} sources",
            timestamp=datetime.now(),
            metadata={
                'sources': [s.name for s, _ in candidates],
                'method': 'priority_based_conflict_resolution'
            }
        )
        merged_lineage.add_evidence(reconciliation_evidence)
        
        return merged_lineage
    
    def search_all_sources(self, query: str, limit: int = 10) -> List[Tuple[str, EnhancedLineage]]:
        """Search across all taxonomy sources"""
        all_results = []
        
        for source in self.sources:
            try:
                results = source.search(query, limit)
                for result in results:
                    all_results.append((source.name, result))
            except Exception as e:
                logger.debug(f"Error searching {source.name}: {e}")
        
        return all_results[:limit]
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about all sources"""
        stats = {}
        
        for source in self.sources:
            source_stats = {
                'name': source.name,
                'description': source.description,
                'priority': source.priority,
                'loaded': source.is_loaded()
            }
            
            # Try to get source-specific stats if available
            if hasattr(source, 'get_statistics'):
                try:
                    source_stats.update(source.get_statistics())
                except Exception as e:
                    logger.debug(f"Error getting stats for {source.name}: {e}")
            
            stats[source.name] = source_stats
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the resolution cache"""
        self._cache.clear()
        logger.info("Cleared taxonomy resolution cache")

# Factory functions for common configurations

def create_standard_resolver(config_dict: Optional[Dict[str, Any]] = None) -> MultiSourceTaxonomyResolver:
    """Create a standard multi-source resolver with common databases"""
    resolver = MultiSourceTaxonomyResolver()
    
    if not config_dict:
        config_dict = config.get('taxonomy', {}).get('multi_source', {})
    
    # Add SILVA if configured
    silva_file = config_dict.get('silva_file')
    if silva_file:
        silva_source = SILVATaxonomySource(silva_file)
        resolver.add_source(silva_source)
    
    # Add PR2 if configured
    pr2_file = config_dict.get('pr2_file')
    if pr2_file:
        pr2_source = PR2TaxonomySource(pr2_file)
        resolver.add_source(pr2_source)
    
    # Add custom taxonomy if configured
    custom_file = config_dict.get('custom_file')
    if custom_file:
        custom_source = CustomTaxonomySource(custom_file)
        resolver.add_source(custom_source)
    
    return resolver

def create_custom_only_resolver(custom_file: str) -> MultiSourceTaxonomyResolver:
    """Create a resolver with only custom taxonomy"""
    resolver = MultiSourceTaxonomyResolver()
    custom_source = CustomTaxonomySource(custom_file)
    resolver.add_source(custom_source)
    return resolver