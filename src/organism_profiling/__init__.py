"""
Organism identification and unique ID generation system.

This module provides sophisticated organism identification based on sequence similarity,
taxonomic information, and environmental context to create unique organism profiles.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio import Align
from scipy.spatial.distance import cosine
import pickle
import json

# Import project modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.models import OrganismProfile, SequenceType
from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class SequenceSignatureGenerator:
    """
    Generates unique signatures for sequences to enable organism identification.
    """
    
    def __init__(self):
        """Initialize signature generator."""
        self.kmer_size = 6
        self.signature_length = 64
    
    def generate_sequence_signature(self, sequence: str) -> str:
        """
        Generate a unique signature for a sequence.
        
        Args:
            sequence: DNA/RNA/protein sequence string
            
        Returns:
            Unique signature hash
        """
        # Normalize sequence
        sequence = sequence.upper().strip()
        
        # Generate k-mers
        kmers = self._extract_kmers(sequence, self.kmer_size)
        
        # Create k-mer frequency profile
        kmer_counts = Counter(kmers)
        
        # Create signature from most frequent k-mers
        top_kmers = dict(kmer_counts.most_common(self.signature_length))
        
        # Create deterministic signature
        signature_data = json.dumps(top_kmers, sort_keys=True)
        signature_hash = hashlib.sha256(signature_data.encode()).hexdigest()
        
        return signature_hash[:32]  # First 32 characters
    
    def _extract_kmers(self, sequence: str, k: int) -> List[str]:
        """Extract k-mers from sequence."""
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            # Skip k-mers with ambiguous nucleotides/amino acids
            if 'N' not in kmer and 'X' not in kmer:
                kmers.append(kmer)
        return kmers
    
    def calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """
        Calculate similarity between two sequence signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Similarity score between 0 and 1
        """
        # For hash-based signatures, we can only do exact matching
        # In a real implementation, you might use LSH or other methods
        return 1.0 if sig1 == sig2 else 0.0


class TaxonomicMatcher:
    """
    Matches organisms based on taxonomic information with fuzzy matching capabilities.
    """
    
    def __init__(self):
        """Initialize taxonomic matcher."""
        self.taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        self.level_weights = {
            'kingdom': 0.1,
            'phylum': 0.15,
            'class': 0.15,
            'order': 0.15,
            'family': 0.15,
            'genus': 0.2,
            'species': 0.3
        }
    
    def calculate_taxonomic_similarity(self, taxonomy1: Dict[str, str], 
                                     taxonomy2: Dict[str, str]) -> float:
        """
        Calculate similarity between two taxonomic assignments.
        
        Args:
            taxonomy1: First taxonomic assignment
            taxonomy2: Second taxonomic assignment
            
        Returns:
            Similarity score between 0 and 1
        """
        similarity = 0.0
        
        for level in self.taxonomic_levels:
            tax1 = taxonomy1.get(level, '').lower().strip()
            tax2 = taxonomy2.get(level, '').lower().strip()
            
            if tax1 and tax2:
                if tax1 == tax2:
                    similarity += self.level_weights[level]
                elif self._fuzzy_match(tax1, tax2):
                    similarity += self.level_weights[level] * 0.8  # Partial credit for fuzzy match
        
        return similarity
    
    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """
        Perform fuzzy matching for taxonomic names.
        
        Args:
            name1: First taxonomic name
            name2: Second taxonomic name
            
        Returns:
            True if names are similar enough
        """
        # Simple fuzzy matching - in practice, you might use more sophisticated methods
        if not name1 or not name2:
            return False
        
        # Check if one is a substring of the other
        if name1 in name2 or name2 in name1:
            return True
        
        # Check for common prefixes (e.g., bacterial names)
        if len(name1) > 3 and len(name2) > 3:
            if name1[:3] == name2[:3]:
                return True
        
        return False
    
    def extract_taxonomic_info(self, sequence_record: SeqRecord) -> Dict[str, str]:
        """
        Extract taxonomic information from sequence record.
        
        Args:
            sequence_record: Bio.SeqRecord object
            
        Returns:
            Dictionary with taxonomic information
        """
        taxonomy = {}
        
        # Extract from annotations
        if hasattr(sequence_record, 'annotations'):
            organism = sequence_record.annotations.get('organism', '')
            if organism:
                taxonomy['organism'] = organism
        
        # Extract from description
        if hasattr(sequence_record, 'description'):
            desc = sequence_record.description.lower()
            
            # Simple extraction patterns (in practice, use more sophisticated parsing)
            if 'bacteria' in desc:
                taxonomy['kingdom'] = 'Bacteria'
            elif 'archaea' in desc:
                taxonomy['kingdom'] = 'Archaea'
            elif 'eukaryota' in desc or 'eukaryotic' in desc:
                taxonomy['kingdom'] = 'Eukaryota'
        
        # Extract from ID if available
        if hasattr(sequence_record, 'id'):
            seq_id = sequence_record.id
            # Extract taxonomic info from ID patterns (database-specific)
            
        return taxonomy


class OrganismIdentifier:
    """
    Main organism identification system that combines multiple identification methods.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize organism identifier.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.signature_generator = SequenceSignatureGenerator()
        self.taxonomic_matcher = TaxonomicMatcher()
        
        # Thresholds for organism matching
        self.sequence_similarity_threshold = 0.95
        self.taxonomic_similarity_threshold = 0.8
        self.combined_similarity_threshold = 0.85
        
        # Cache for performance
        self._organism_cache = {}
        self._signature_cache = {}
    
    def identify_organism(self, sequences: List[SeqRecord], 
                         taxonomic_assignments: Optional[List[Dict[str, Any]]] = None,
                         environmental_context: Optional[Dict[str, Any]] = None) -> OrganismProfile:
        """
        Identify organism from sequence data and create unique profile.
        
        Args:
            sequences: List of sequence records for this organism
            taxonomic_assignments: Optional taxonomic assignment results
            environmental_context: Optional environmental metadata
            
        Returns:
            OrganismProfile instance with unique ID
        """
        logger.info(f"Identifying organism from {len(sequences)} sequences")
        
        # Generate sequence signatures
        signatures = []
        for seq in sequences:
            seq_str = str(seq.seq).upper()
            signature = self.signature_generator.generate_sequence_signature(seq_str)
            signatures.append(signature)
        
        # Create consensus signature
        consensus_signature = self._create_consensus_signature(signatures)
        
        # Extract taxonomic information
        taxonomy = self._extract_consensus_taxonomy(sequences, taxonomic_assignments)
        
        # Check if organism already exists
        existing_organism = self._find_matching_organism(consensus_signature, taxonomy)
        
        if existing_organism:
            logger.info(f"Found matching organism: {existing_organism.organism_id}")
            # Update detection count and context
            existing_organism.detection_count += 1
            existing_organism.last_updated = datetime.now()
            return existing_organism
        
        # Create new organism profile
        organism_id = self._generate_organism_id(taxonomy, consensus_signature)
        
        # Determine if this is a potential novel organism
        is_novel, novelty_score = self._assess_novelty(consensus_signature, taxonomy)
        
        organism_profile = OrganismProfile(
            organism_id=organism_id,
            organism_name=self._generate_organism_name(taxonomy),
            taxonomic_lineage=self._build_taxonomic_lineage(taxonomy),
            kingdom=taxonomy.get('kingdom'),
            phylum=taxonomy.get('phylum'),
            class_name=taxonomy.get('class'),
            order_name=taxonomy.get('order'),
            family=taxonomy.get('family'),
            genus=taxonomy.get('genus'),
            species=taxonomy.get('species'),
            sequence_signature=consensus_signature,
            first_detected=datetime.now(),
            last_updated=datetime.now(),
            detection_count=1,
            confidence_score=self._calculate_confidence_score(sequences, taxonomy),
            is_novel_candidate=is_novel,
            novelty_score=novelty_score,
            reference_databases=[],  # To be populated by analysis pipeline
            notes=self._generate_notes(sequences, environmental_context)
        )
        
        logger.info(f"Created new organism profile: {organism_id}")
        return organism_profile
    
    def batch_identify_organisms(self, sequence_groups: List[List[SeqRecord]], 
                                taxonomic_data: Optional[List[List[Dict[str, Any]]]] = None) -> List[OrganismProfile]:
        """
        Batch identify multiple organisms for efficiency.
        
        Args:
            sequence_groups: List of sequence groups, each representing one organism
            taxonomic_data: Optional taxonomic data for each group
            
        Returns:
            List of OrganismProfile instances
        """
        logger.info(f"Batch identifying {len(sequence_groups)} organism groups")
        
        organisms = []
        for i, sequences in enumerate(sequence_groups):
            taxonomy_assignments = taxonomic_data[i] if taxonomic_data else None
            organism = self.identify_organism(sequences, taxonomy_assignments)
            organisms.append(organism)
        
        return organisms
    
    def _create_consensus_signature(self, signatures: List[str]) -> str:
        """
        Create consensus signature from multiple sequence signatures.
        
        Args:
            signatures: List of individual sequence signatures
            
        Returns:
            Consensus signature
        """
        if len(signatures) == 1:
            return signatures[0]
        
        # For now, use the most common signature
        # In practice, you might use more sophisticated consensus methods
        signature_counts = Counter(signatures)
        most_common_signature = signature_counts.most_common(1)[0][0]
        
        return most_common_signature
    
    def _extract_consensus_taxonomy(self, sequences: List[SeqRecord], 
                                  taxonomic_assignments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        """
        Extract consensus taxonomic information from sequences and assignments.
        
        Args:
            sequences: List of sequence records
            taxonomic_assignments: Optional external taxonomic assignments
            
        Returns:
            Consensus taxonomic information
        """
        # Extract taxonomy from sequence records
        sequence_taxonomies = []
        for seq in sequences:
            taxonomy = self.taxonomic_matcher.extract_taxonomic_info(seq)
            if taxonomy:
                sequence_taxonomies.append(taxonomy)
        
        # Combine with external assignments
        all_taxonomies = sequence_taxonomies[:]
        if taxonomic_assignments:
            all_taxonomies.extend(taxonomic_assignments)
        
        # Create consensus taxonomy
        consensus = {}
        for level in self.taxonomic_matcher.taxonomic_levels:
            level_values = [tax.get(level) for tax in all_taxonomies if tax.get(level)]
            if level_values:
                # Use most common value for this level
                level_counts = Counter(level_values)
                consensus[level] = level_counts.most_common(1)[0][0]
        
        return consensus
    
    def _find_matching_organism(self, signature: str, taxonomy: Dict[str, str]) -> Optional[OrganismProfile]:
        """
        Find existing organism that matches the given signature and taxonomy.
        
        Args:
            signature: Sequence signature
            taxonomy: Taxonomic information
            
        Returns:
            Matching OrganismProfile or None
        """
        # Check cache first
        cache_key = f"{signature}_{hash(frozenset(taxonomy.items()))}"
        if cache_key in self._organism_cache:
            return self._organism_cache[cache_key]
        
        # Search database for similar organisms
        try:
            with self.db_manager.get_connection() as conn:
                # First, try exact signature match
                cursor = conn.execute("""
                    SELECT organism_id FROM organism_profiles 
                    WHERE sequence_signature = ?
                """, (signature,))
                
                exact_matches = [row[0] for row in cursor.fetchall()]
                
                for organism_id in exact_matches:
                    organism = self.db_manager.get_organism_profile(organism_id)
                    if organism:
                        # Check taxonomic compatibility
                        organism_taxonomy = {
                            'kingdom': organism.kingdom,
                            'phylum': organism.phylum,
                            'class': organism.class_name,
                            'order': organism.order_name,
                            'family': organism.family,
                            'genus': organism.genus,
                            'species': organism.species
                        }
                        
                        tax_similarity = self.taxonomic_matcher.calculate_taxonomic_similarity(
                            taxonomy, organism_taxonomy
                        )
                        
                        if tax_similarity >= self.taxonomic_similarity_threshold:
                            self._organism_cache[cache_key] = organism
                            return organism
                
                # If no exact match, search for taxonomically similar organisms
                # This is a simplified version - in practice, you'd use more sophisticated matching
                if taxonomy.get('genus') and taxonomy.get('species'):
                    cursor = conn.execute("""
                        SELECT organism_id FROM organism_profiles 
                        WHERE genus = ? AND species = ?
                    """, (taxonomy['genus'], taxonomy['species']))
                    
                    for row in cursor.fetchall():
                        organism = self.db_manager.get_organism_profile(row[0])
                        if organism:
                            # Calculate overall similarity
                            sig_similarity = self.signature_generator.calculate_signature_similarity(
                                signature, organism.sequence_signature
                            )
                            
                            organism_taxonomy = {
                                'kingdom': organism.kingdom,
                                'phylum': organism.phylum,
                                'class': organism.class_name,
                                'order': organism.order_name,
                                'family': organism.family,
                                'genus': organism.genus,
                                'species': organism.species
                            }
                            
                            tax_similarity = self.taxonomic_matcher.calculate_taxonomic_similarity(
                                taxonomy, organism_taxonomy
                            )
                            
                            combined_similarity = (sig_similarity + tax_similarity) / 2
                            
                            if combined_similarity >= self.combined_similarity_threshold:
                                self._organism_cache[cache_key] = organism
                                return organism
                
        except Exception as e:
            logger.error(f"Error finding matching organism: {str(e)}")
        
        return None
    
    def _generate_organism_id(self, taxonomy: Dict[str, str], signature: str) -> str:
        """
        Generate unique organism ID based on taxonomy and signature.
        
        Args:
            taxonomy: Taxonomic information
            signature: Sequence signature
            
        Returns:
            Unique organism ID
        """
        # Create deterministic ID based on available taxonomic info
        genus = taxonomy.get('genus', 'Unknown')
        species = taxonomy.get('species', 'Unknown')
        
        # Include signature for uniqueness
        combined = f"{genus}_{species}_{signature[:8]}"
        hash_obj = hashlib.md5(combined.encode())
        
        return f"ORG_{hash_obj.hexdigest()[:12].upper()}"
    
    def _generate_organism_name(self, taxonomy: Dict[str, str]) -> str:
        """
        Generate organism name from taxonomic information.
        
        Args:
            taxonomy: Taxonomic information
            
        Returns:
            Generated organism name
        """
        genus = taxonomy.get('genus', '')
        species = taxonomy.get('species', '')
        
        if genus and species:
            return f"{genus} {species}"
        elif genus:
            return f"{genus} sp."
        elif taxonomy.get('family'):
            return f"{taxonomy['family']} family member"
        elif taxonomy.get('order'):
            return f"{taxonomy['order']} order member"
        else:
            return "Unclassified organism"
    
    def _build_taxonomic_lineage(self, taxonomy: Dict[str, str]) -> str:
        """
        Build taxonomic lineage string.
        
        Args:
            taxonomy: Taxonomic information
            
        Returns:
            Taxonomic lineage string
        """
        lineage_parts = []
        
        for level in self.taxonomic_matcher.taxonomic_levels:
            value = taxonomy.get(level)
            if value:
                lineage_parts.append(f"{level[0].upper()}__{value}")
        
        return "; ".join(lineage_parts) if lineage_parts else "Unclassified"
    
    def _assess_novelty(self, signature: str, taxonomy: Dict[str, str]) -> Tuple[bool, float]:
        """
        Assess if organism is potentially novel.
        
        Args:
            signature: Sequence signature
            taxonomy: Taxonomic information
            
        Returns:
            Tuple of (is_novel, novelty_score)
        """
        # Simple novelty assessment - in practice, this would be more sophisticated
        
        # Check if genus/species is known in database
        genus = taxonomy.get('genus')
        species = taxonomy.get('species')
        
        if not genus or not species:
            return True, 0.8  # High novelty if taxonomy is incomplete
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM organism_profiles 
                    WHERE genus = ? AND species = ?
                """, (genus, species))
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    return True, 0.9  # Likely novel if no known organisms with same genus/species
                elif count < 3:
                    return True, 0.7  # Possibly novel if only a few known
                else:
                    return False, 0.3  # Likely known
                    
        except Exception as e:
            logger.error(f"Error assessing novelty: {str(e)}")
            return False, 0.5
    
    def _calculate_confidence_score(self, sequences: List[SeqRecord], 
                                  taxonomy: Dict[str, str]) -> float:
        """
        Calculate confidence score for organism identification.
        
        Args:
            sequences: List of sequence records
            taxonomy: Taxonomic information
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0
        
        # Base confidence on number of sequences
        seq_count_score = min(len(sequences) / 10, 1.0)  # Max score at 10+ sequences
        confidence += seq_count_score * 0.3
        
        # Base confidence on taxonomic completeness
        taxonomic_levels_filled = sum(1 for level in self.taxonomic_matcher.taxonomic_levels 
                                    if taxonomy.get(level))
        taxonomic_completeness = taxonomic_levels_filled / len(self.taxonomic_matcher.taxonomic_levels)
        confidence += taxonomic_completeness * 0.4
        
        # Base confidence on sequence quality (length as proxy)
        if sequences:
            avg_length = np.mean([len(seq.seq) for seq in sequences])
            length_score = min(avg_length / 500, 1.0)  # Max score at 500+ bp
            confidence += length_score * 0.3
        
        return min(confidence, 1.0)
    
    def _generate_notes(self, sequences: List[SeqRecord], 
                       environmental_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate notes for organism profile.
        
        Args:
            sequences: List of sequence records
            environmental_context: Optional environmental metadata
            
        Returns:
            Notes string
        """
        notes = []
        
        notes.append(f"Identified from {len(sequences)} sequences")
        
        if sequences:
            lengths = [len(seq.seq) for seq in sequences]
            notes.append(f"Sequence lengths: {min(lengths)}-{max(lengths)} bp")
        
        if environmental_context:
            if environmental_context.get('collection_location'):
                notes.append(f"Location: {environmental_context['collection_location']}")
            if environmental_context.get('depth_meters'):
                notes.append(f"Depth: {environmental_context['depth_meters']} meters")
        
        notes.append(f"Identified on: {datetime.now().strftime('%Y-%m-%d')}")
        
        return "; ".join(notes)


class OrganismMatcher:
    """
    Utility class for matching and comparing organisms across different analyses.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize organism matcher.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.signature_generator = SequenceSignatureGenerator()
        self.taxonomic_matcher = TaxonomicMatcher()
    
    def find_similar_organisms(self, target_organism_id: str, 
                             similarity_threshold: float = 0.8,
                             max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Find organisms similar to the target organism.
        
        Args:
            target_organism_id: ID of target organism
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of (organism_id, similarity_score) tuples
        """
        target_organism = self.db_manager.get_organism_profile(target_organism_id)
        if not target_organism:
            return []
        
        similar_organisms = []
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT organism_id, sequence_signature, kingdom, phylum, 
                           class, order_name, family, genus, species
                    FROM organism_profiles 
                    WHERE organism_id != ?
                """, (target_organism_id,))
                
                target_taxonomy = {
                    'kingdom': target_organism.kingdom,
                    'phylum': target_organism.phylum,
                    'class': target_organism.class_name,
                    'order': target_organism.order_name,
                    'family': target_organism.family,
                    'genus': target_organism.genus,
                    'species': target_organism.species
                }
                
                for row in cursor.fetchall():
                    organism_id, signature, kingdom, phylum, class_name, order_name, family, genus, species = row
                    
                    # Calculate signature similarity
                    sig_similarity = self.signature_generator.calculate_signature_similarity(
                        target_organism.sequence_signature, signature
                    )
                    
                    # Calculate taxonomic similarity
                    other_taxonomy = {
                        'kingdom': kingdom,
                        'phylum': phylum,
                        'class': class_name,
                        'order': order_name,
                        'family': family,
                        'genus': genus,
                        'species': species
                    }
                    
                    tax_similarity = self.taxonomic_matcher.calculate_taxonomic_similarity(
                        target_taxonomy, other_taxonomy
                    )
                    
                    # Combined similarity
                    combined_similarity = (sig_similarity + tax_similarity) / 2
                    
                    if combined_similarity >= similarity_threshold:
                        similar_organisms.append((organism_id, combined_similarity))
                
                # Sort by similarity and limit results
                similar_organisms.sort(key=lambda x: x[1], reverse=True)
                return similar_organisms[:max_results]
                
        except Exception as e:
            logger.error(f"Error finding similar organisms: {str(e)}")
            return []
    
    def match_organisms_across_reports(self, report_id_1: str, report_id_2: str) -> Dict[str, Any]:
        """
        Match organisms between two analysis reports.
        
        Args:
            report_id_1: First report ID
            report_id_2: Second report ID
            
        Returns:
            Dictionary with matching results
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Get organisms from both reports
                cursor1 = conn.execute("""
                    SELECT DISTINCT organism_id FROM sequences WHERE report_id = ?
                """, (report_id_1,))
                organisms_1 = set(row[0] for row in cursor1.fetchall())
                
                cursor2 = conn.execute("""
                    SELECT DISTINCT organism_id FROM sequences WHERE report_id = ?
                """, (report_id_2,))
                organisms_2 = set(row[0] for row in cursor2.fetchall())
                
                # Find exact matches
                exact_matches = organisms_1.intersection(organisms_2)
                
                # Find similar organisms between the reports
                similar_matches = []
                unique_to_1 = organisms_1 - organisms_2
                unique_to_2 = organisms_2 - organisms_1
                
                for org_1 in unique_to_1:
                    for org_2 in unique_to_2:
                        similarity = self._calculate_organism_similarity(org_1, org_2)
                        if similarity >= 0.8:  # High similarity threshold for cross-report matching
                            similar_matches.append((org_1, org_2, similarity))
                
                return {
                    'report_id_1': report_id_1,
                    'report_id_2': report_id_2,
                    'organisms_report_1': len(organisms_1),
                    'organisms_report_2': len(organisms_2),
                    'exact_matches': len(exact_matches),
                    'exact_match_ids': list(exact_matches),
                    'similar_matches': similar_matches,
                    'unique_to_report_1': len(unique_to_1),
                    'unique_to_report_2': len(unique_to_2),
                    'jaccard_similarity': len(exact_matches) / len(organisms_1.union(organisms_2)) if organisms_1.union(organisms_2) else 0
                }
                
        except Exception as e:
            logger.error(f"Error matching organisms across reports: {str(e)}")
            return {}
    
    def _calculate_organism_similarity(self, organism_id_1: str, organism_id_2: str) -> float:
        """
        Calculate similarity between two organisms.
        
        Args:
            organism_id_1: First organism ID
            organism_id_2: Second organism ID
            
        Returns:
            Similarity score between 0 and 1
        """
        org_1 = self.db_manager.get_organism_profile(organism_id_1)
        org_2 = self.db_manager.get_organism_profile(organism_id_2)
        
        if not org_1 or not org_2:
            return 0.0
        
        # Signature similarity
        sig_similarity = self.signature_generator.calculate_signature_similarity(
            org_1.sequence_signature, org_2.sequence_signature
        )
        
        # Taxonomic similarity
        taxonomy_1 = {
            'kingdom': org_1.kingdom,
            'phylum': org_1.phylum,
            'class': org_1.class_name,
            'order': org_1.order_name,
            'family': org_1.family,
            'genus': org_1.genus,
            'species': org_1.species
        }
        
        taxonomy_2 = {
            'kingdom': org_2.kingdom,
            'phylum': org_2.phylum,
            'class': org_2.class_name,
            'order': org_2.order_name,
            'family': org_2.family,
            'genus': org_2.genus,
            'species': org_2.species
        }
        
        tax_similarity = self.taxonomic_matcher.calculate_taxonomic_similarity(
            taxonomy_1, taxonomy_2
        )
        
        # Combined similarity
        return (sig_similarity + tax_similarity) / 2