"""
Enhanced Taxonomic Lineage Resolution System

This module provides comprehensive taxonomic lineage enrichment with support for:
- Extended taxonomic ranks beyond the basic 7
- Common names and synonyms
- Confidence scoring and evidence tracking
- Multi-source taxonomy integration
- External database linking
- Custom taxonomy support
- Enhanced visualization and export capabilities
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
import logging
from functools import lru_cache
import requests
from urllib.parse import quote
import re
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """Types of evidence for taxonomic assignments"""
    NCBI_TAXDUMP = "ncbi_taxdump"
    SILVA = "silva"
    PR2 = "pr2"
    UNITE = "unite"
    GTDB = "gtdb"
    BLAST_HIT = "blast_hit"
    ML_PREDICTION = "ml_prediction"
    KNN_CONSENSUS = "knn_consensus"
    MANUAL_CURATION = "manual_curation"
    CUSTOM_DATABASE = "custom_database"

class TaxonomicRank(Enum):
    """Extended taxonomic ranks including intermediate levels"""
    # Primary ranks
    SUPERKINGDOM = ("superkingdom", 1)
    KINGDOM = ("kingdom", 2)
    SUBKINGDOM = ("subkingdom", 3)
    SUPERPHYLUM = ("superphylum", 4)
    PHYLUM = ("phylum", 5)
    SUBPHYLUM = ("subphylum", 6)
    SUPERCLASS = ("superclass", 7)
    CLASS = ("class", 8)
    SUBCLASS = ("subclass", 9)
    INFRACLASS = ("infraclass", 10)
    SUPERORDER = ("superorder", 11)
    ORDER = ("order", 12)
    SUBORDER = ("suborder", 13)
    INFRAORDER = ("infraorder", 14)
    PARVORDER = ("parvorder", 15)
    SUPERFAMILY = ("superfamily", 16)
    FAMILY = ("family", 17)
    SUBFAMILY = ("subfamily", 18)
    TRIBE = ("tribe", 19)
    SUBTRIBE = ("subtribe", 20)
    GENUS = ("genus", 21)
    SUBGENUS = ("subgenus", 22)
    SPECIES_GROUP = ("species group", 23)
    SPECIES_SUBGROUP = ("species subgroup", 24)
    SPECIES = ("species", 25)
    SUBSPECIES = ("subspecies", 26)
    VARIETAS = ("varietas", 27)  # variety
    FORMA = ("forma", 28)  # form
    
    def __init__(self, rank_name: str, priority: int):
        self.rank_name = rank_name
        self.priority = priority
    
    @classmethod
    def from_string(cls, rank_str: str) -> Optional['TaxonomicRank']:
        """Get TaxonomicRank from string representation"""
        rank_str = rank_str.lower().strip()
        for rank in cls:
            if rank.rank_name.lower() == rank_str:
                return rank
        return None
    
    @classmethod
    def basic_ranks(cls) -> List['TaxonomicRank']:
        """Get the basic 7 taxonomic ranks for compatibility"""
        return [cls.KINGDOM, cls.PHYLUM, cls.CLASS, cls.ORDER, 
                cls.FAMILY, cls.GENUS, cls.SPECIES]

@dataclass
class ExternalReference:
    """External database reference for taxonomic entities"""
    database: str  # e.g., 'NCBI', 'ITIS', 'GBIF', 'WoRMS'
    identifier: str  # Database-specific ID
    url: Optional[str] = None  # Direct URL if available
    
    def generate_url(self) -> str:
        """Generate URL for external database lookup"""
        if self.url:
            return self.url
            
        # URL templates for major taxonomic databases
        url_templates = {
            'NCBI': 'https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={id}',
            'ITIS': 'https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value={id}',
            'GBIF': 'https://www.gbif.org/species/{id}',
            'WoRMS': 'https://www.marinespecies.org/aphia.php?p=taxdetails&id={id}',
            'EOL': 'https://eol.org/pages/{id}',
            'BOLD': 'https://www.boldsystems.org/index.php/Taxbrowser_Taxonpage?taxid={id}',
            'SILVA': 'https://www.arb-silva.de/browser/ssu/silva/{id}',
            'PR2': 'https://pr2-database.org/browse/{id}',
            'UNITE': 'https://unite.ut.ee/bl_forw.php?id={id}',
            'GTDB': 'https://gtdb.ecogenomic.org/taxon_overview?id={id}'
        }
        
        template = url_templates.get(self.database.upper())
        if template:
            return template.format(id=quote(str(self.identifier)))
        return f"https://www.google.com/search?q={quote(f'{self.database} {self.identifier}')}"

@dataclass
class LineageEvidence:
    """Evidence supporting a taxonomic assignment"""
    source: EvidenceType
    confidence: float  # 0.0 to 1.0
    method_details: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaxonomicName:
    """Represents different names for a taxonomic entity"""
    scientific_name: str
    common_names: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    authority: Optional[str] = None  # Taxonomic authority (e.g., "Linnaeus, 1758")
    name_status: str = "valid"  # valid, synonym, invalid, etc.

@dataclass
class EnhancedLineage:
    """Comprehensive taxonomic lineage with enriched information"""
    taxid: Optional[int] = None
    scientific_name: Optional[str] = None
    
    # Extended rank assignments
    lineage: Dict[TaxonomicRank, Optional[str]] = field(default_factory=dict)
    
    # Naming information
    names: TaxonomicName = field(default_factory=lambda: TaxonomicName(""))
    
    # Evidence and confidence
    evidence: List[LineageEvidence] = field(default_factory=list)
    overall_confidence: float = 0.0
    
    # External references
    external_refs: List[ExternalReference] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    
    def get_basic_lineage(self) -> Dict[str, Optional[str]]:
        """Get basic 7-rank lineage for compatibility"""
        basic_mapping = {
            'kingdom': TaxonomicRank.KINGDOM,
            'phylum': TaxonomicRank.PHYLUM,
            'class': TaxonomicRank.CLASS,
            'order': TaxonomicRank.ORDER,
            'family': TaxonomicRank.FAMILY,
            'genus': TaxonomicRank.GENUS,
            'species': TaxonomicRank.SPECIES
        }
        
        result = {}
        for basic_name, rank_enum in basic_mapping.items():
            result[basic_name] = self.lineage.get(rank_enum)
        
        return result
    
    def add_evidence(self, evidence: LineageEvidence) -> None:
        """Add evidence and update overall confidence"""
        self.evidence.append(evidence)
        self._update_confidence()
    
    def _update_confidence(self) -> None:
        """Update overall confidence based on evidence"""
        if not self.evidence:
            self.overall_confidence = 0.0
            return
        
        # Weight evidence by source reliability
        source_weights = {
            EvidenceType.NCBI_TAXDUMP: 1.0,
            EvidenceType.SILVA: 0.9,
            EvidenceType.PR2: 0.9,
            EvidenceType.UNITE: 0.9,
            EvidenceType.GTDB: 0.8,
            EvidenceType.BLAST_HIT: 0.7,
            EvidenceType.ML_PREDICTION: 0.6,
            EvidenceType.KNN_CONSENSUS: 0.8,
            EvidenceType.MANUAL_CURATION: 0.95,
            EvidenceType.CUSTOM_DATABASE: 0.5
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for evidence in self.evidence:
            weight = source_weights.get(evidence.source, 0.5)
            weighted_sum += evidence.confidence * weight
            total_weight += weight
        
        self.overall_confidence = weighted_sum / max(total_weight, 1e-12)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        
        # Convert enums to strings
        result['lineage'] = {rank.rank_name: name for rank, name in self.lineage.items()}
        result['evidence'] = [
            {
                'source': ev.source.value,
                'confidence': ev.confidence,
                'method_details': ev.method_details,
                'timestamp': ev.timestamp.isoformat() if ev.timestamp else None,
                'metadata': ev.metadata
            }
            for ev in self.evidence
        ]
        result['last_updated'] = self.last_updated.isoformat() if self.last_updated else None
        
        return result

class EnhancedTaxdumpResolver:
    """Enhanced version of TaxdumpResolver with extended capabilities"""
    
    def __init__(self, 
                 taxdump_dir: Optional[str],
                 cache_db_path: Optional[str] = None,
                 enable_caching: bool = True):
        self.taxdump_dir = Path(taxdump_dir) if taxdump_dir else None
        self.enable_caching = enable_caching
        
        # SQLite cache for performance
        self.cache_db_path = cache_db_path or (self.taxdump_dir / "lineage_cache.db" if self.taxdump_dir else None)
        
        # Data structures
        self._name_to_taxid: Optional[Dict[str, int]] = None
        self._taxid_to_name: Optional[Dict[int, str]] = None
        self._common_names: Optional[Dict[int, List[str]]] = None  # taxid -> [common names]
        self._synonyms: Optional[Dict[int, List[str]]] = None  # taxid -> [synonyms]
        self._nodes: Optional[Dict[int, Tuple[int, str]]] = None
        self._merged_old2new: Optional[Dict[int, int]] = None
        
        # Initialize cache database
        if self.enable_caching and self.cache_db_path:
            self._init_cache_db()
    
    def _init_cache_db(self) -> None:
        """Initialize SQLite cache database"""
        if not self.cache_db_path:
            return
        
        try:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables for caching
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS lineage_cache (
                        taxid INTEGER PRIMARY KEY,
                        lineage_json TEXT NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS name_cache (
                        name_key TEXT PRIMARY KEY,
                        taxid INTEGER NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info(f"Cache database initialized at {self.cache_db_path}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize cache database: {e}")
            self.enable_caching = False
    
    def available(self) -> bool:
        """Check if taxdump files are available"""
        return bool(self.taxdump_dir) \
            and (self.taxdump_dir / 'names.dmp').exists() \
            and (self.taxdump_dir / 'nodes.dmp').exists()
    
    def _load(self) -> None:
        """Load taxdump data with enhanced parsing"""
        if (self._name_to_taxid is not None and self._nodes is not None 
            and self._taxid_to_name is not None):
            return
        
        logger.info("Loading enhanced taxdump data...")
        
        name_to_taxid: Dict[str, int] = {}
        taxid_to_name: Dict[int, str] = {}
        common_names: Dict[int, List[str]] = {}
        synonyms: Dict[int, List[str]] = {}
        nodes: Dict[int, Tuple[int, str]] = {}
        merged: Dict[int, int] = {}
        
        names_path = self.taxdump_dir / 'names.dmp'
        nodes_path = self.taxdump_dir / 'nodes.dmp'
        merged_path = self.taxdump_dir / 'merged.dmp'
        
        # Parse names.dmp with enhanced name type support
        with open(names_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 4:
                    continue
                
                taxid_str, name_txt, _unique, name_class = parts[:4]
                try:
                    taxid = int(taxid_str)
                except ValueError:
                    continue
                
                name_txt = name_txt.strip()
                name_class = name_class.strip()
                
                if name_class == 'scientific name':
                    key = name_txt.lower()
                    if key not in name_to_taxid:
                        name_to_taxid[key] = taxid
                    if taxid not in taxid_to_name:
                        taxid_to_name[taxid] = name_txt
                
                elif name_class in ['common name', 'genbank common name']:
                    if taxid not in common_names:
                        common_names[taxid] = []
                    if name_txt not in common_names[taxid]:
                        common_names[taxid].append(name_txt)
                
                elif name_class in ['synonym', 'equivalent name', 'anamorph']:
                    if taxid not in synonyms:
                        synonyms[taxid] = []
                    if name_txt not in synonyms[taxid]:
                        synonyms[taxid].append(name_txt)
        
        # Parse nodes.dmp
        with open(nodes_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 3:
                    continue
                try:
                    taxid = int(parts[0])
                    parent = int(parts[1])
                except ValueError:
                    continue
                rank = parts[2].strip()
                nodes[taxid] = (parent, rank)
        
        # Parse merged.dmp
        if merged_path.exists():
            with open(merged_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 2:
                        continue
                    try:
                        old = int(parts[0])
                        new = int(parts[1])
                    except ValueError:
                        continue
                    merged[old] = new
        
        # Store data
        self._name_to_taxid = name_to_taxid
        self._taxid_to_name = taxid_to_name
        self._common_names = common_names
        self._synonyms = synonyms
        self._nodes = nodes
        self._merged_old2new = merged if merged else None
        
        logger.info(f"Enhanced taxdump loaded: {len(name_to_taxid)} names, "
                   f"{len(nodes)} nodes, {len(common_names)} with common names, "
                   f"{len(synonyms)} with synonyms")
    
    @lru_cache(maxsize=100000)
    def enhanced_lineage_by_name(self, scientific_name: Optional[str]) -> EnhancedLineage:
        """Get enhanced lineage by scientific name"""
        if not scientific_name:
            return EnhancedLineage()
        
        self._load()
        assert self._name_to_taxid is not None
        
        taxid = self._name_to_taxid.get(scientific_name.lower())
        if taxid is not None:
            return self.enhanced_lineage_by_taxid(taxid)
        
        # Create basic lineage with provided name
        lineage = EnhancedLineage(
            scientific_name=scientific_name,
            names=TaxonomicName(scientific_name=scientific_name)
        )
        lineage.lineage[TaxonomicRank.SPECIES] = scientific_name
        
        return lineage
    
    @lru_cache(maxsize=100000)
    def enhanced_lineage_by_taxid(self, taxid: Optional[int]) -> EnhancedLineage:
        """Get enhanced lineage by taxonomic ID"""
        if taxid is None or not self.available():
            return EnhancedLineage()
        
        # Check cache first
        cached_lineage = self._get_cached_lineage(taxid)
        if cached_lineage:
            return cached_lineage
        
        self._load()
        assert (self._nodes is not None and self._taxid_to_name is not None 
                and self._common_names is not None and self._synonyms is not None)
        
        # Remap merged taxids
        original_taxid = taxid
        if self._merged_old2new and taxid in self._merged_old2new:
            taxid = self._merged_old2new[taxid]
        
        # Create enhanced lineage
        lineage = EnhancedLineage(
            taxid=taxid,
            last_updated=datetime.now()
        )
        
        # Get scientific name
        scientific_name = self._taxid_to_name.get(taxid)
        if scientific_name:
            lineage.scientific_name = scientific_name
            lineage.names.scientific_name = scientific_name
        
        # Get common names and synonyms
        if taxid in self._common_names:
            lineage.names.common_names = self._common_names[taxid].copy()
        
        if taxid in self._synonyms:
            lineage.names.synonyms = self._synonyms[taxid].copy()
        
        # Traverse lineage
        seen = set()
        steps = 0
        current_taxid = taxid
        
        while current_taxid in self._nodes and current_taxid not in seen and steps < 100:
            seen.add(current_taxid)
            parent, rank_str = self._nodes[current_taxid]
            
            # Map rank string to enum
            rank_enum = TaxonomicRank.from_string(rank_str)
            if rank_enum and current_taxid in self._taxid_to_name:
                lineage.lineage[rank_enum] = self._taxid_to_name[current_taxid]
            
            if parent == current_taxid:
                break
            current_taxid = parent
            steps += 1
        
        # Add evidence
        evidence = LineageEvidence(
            source=EvidenceType.NCBI_TAXDUMP,
            confidence=1.0,
            method_details="NCBI Taxdump traversal",
            timestamp=datetime.now(),
            metadata={'original_taxid': original_taxid}
        )
        lineage.add_evidence(evidence)
        
        # Add external references
        if taxid:
            ncbi_ref = ExternalReference(database='NCBI', identifier=str(taxid))
            lineage.external_refs.append(ncbi_ref)
        
        # Cache the result
        self._cache_lineage(taxid, lineage)
        
        return lineage
    
    def _get_cached_lineage(self, taxid: int) -> Optional[EnhancedLineage]:
        """Retrieve lineage from cache"""
        if not self.enable_caching or not self.cache_db_path:
            return None
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT lineage_json FROM lineage_cache WHERE taxid = ?",
                    (taxid,)
                )
                result = cursor.fetchone()
                if result:
                    lineage_dict = json.loads(result[0])
                    return self._dict_to_lineage(lineage_dict)
        except Exception as e:
            logger.debug(f"Cache retrieval failed for taxid {taxid}: {e}")
        
        return None
    
    def _cache_lineage(self, taxid: int, lineage: EnhancedLineage) -> None:
        """Store lineage in cache"""
        if not self.enable_caching or not self.cache_db_path:
            return
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                lineage_json = json.dumps(lineage.to_dict())
                cursor.execute(
                    "INSERT OR REPLACE INTO lineage_cache (taxid, lineage_json) VALUES (?, ?)",
                    (taxid, lineage_json)
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Cache storage failed for taxid {taxid}: {e}")
    
    def _dict_to_lineage(self, lineage_dict: Dict[str, Any]) -> EnhancedLineage:
        """Convert dictionary back to EnhancedLineage object"""
        lineage = EnhancedLineage()
        
        # Basic fields
        lineage.taxid = lineage_dict.get('taxid')
        lineage.scientific_name = lineage_dict.get('scientific_name')
        lineage.overall_confidence = lineage_dict.get('overall_confidence', 0.0)
        lineage.metadata = lineage_dict.get('metadata', {})
        
        # Parse dates
        if lineage_dict.get('last_updated'):
            lineage.last_updated = datetime.fromisoformat(lineage_dict['last_updated'])
        
        # Reconstruct lineage mapping
        if 'lineage' in lineage_dict:
            for rank_name, name in lineage_dict['lineage'].items():
                if name is not None:
                    rank_enum = TaxonomicRank.from_string(rank_name)
                    if rank_enum:
                        lineage.lineage[rank_enum] = name
        
        # Reconstruct names
        if 'names' in lineage_dict:
            names_data = lineage_dict['names']
            lineage.names = TaxonomicName(
                scientific_name=names_data.get('scientific_name', ''),
                common_names=names_data.get('common_names', []),
                synonyms=names_data.get('synonyms', []),
                authority=names_data.get('authority'),
                name_status=names_data.get('name_status', 'valid')
            )
        
        # Reconstruct evidence
        if 'evidence' in lineage_dict:
            for ev_data in lineage_dict['evidence']:
                evidence = LineageEvidence(
                    source=EvidenceType(ev_data['source']),
                    confidence=ev_data['confidence'],
                    method_details=ev_data.get('method_details'),
                    timestamp=datetime.fromisoformat(ev_data['timestamp']) if ev_data.get('timestamp') else None,
                    metadata=ev_data.get('metadata', {})
                )
                lineage.evidence.append(evidence)
        
        # Reconstruct external references
        if 'external_refs' in lineage_dict:
            for ref_data in lineage_dict['external_refs']:
                ref = ExternalReference(
                    database=ref_data['database'],
                    identifier=ref_data['identifier'],
                    url=ref_data.get('url')
                )
                lineage.external_refs.append(ref)
        
        return lineage
    
    def search_by_common_name(self, common_name: str) -> List[EnhancedLineage]:
        """Search for taxa by common name"""
        self._load()
        assert self._common_names is not None
        
        results = []
        common_name_lower = common_name.lower()
        
        for taxid, names in self._common_names.items():
            for name in names:
                if common_name_lower in name.lower():
                    lineage = self.enhanced_lineage_by_taxid(taxid)
                    results.append(lineage)
                    break  # Avoid duplicates
        
        return results
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded taxonomic data"""
        self._load()
        
        if not all([self._name_to_taxid, self._nodes, self._common_names, self._synonyms]):
            return {}
        
        # Count entries by rank
        rank_counts = {}
        for taxid, (parent, rank) in self._nodes.items():
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        return {
            'total_names': len(self._name_to_taxid),
            'total_nodes': len(self._nodes),
            'taxa_with_common_names': len(self._common_names),
            'taxa_with_synonyms': len(self._synonyms),
            'rank_distribution': rank_counts,
            'merged_taxa': len(self._merged_old2new) if self._merged_old2new else 0
        }

# Backward compatibility function
def create_enhanced_resolver(taxdump_dir: Optional[str] = None) -> EnhancedTaxdumpResolver:
    """Create an enhanced taxdump resolver with default settings"""
    if taxdump_dir is None:
        taxdump_dir = config.get('taxonomy', {}).get('taxdump_dir')
    
    return EnhancedTaxdumpResolver(taxdump_dir)