"""
Advanced Taxonomic Analysis Module

This module provides enhanced taxonomic assignment and analysis capabilities
for eDNA sequences, including phylogenetic analysis, confidence scoring,
and taxonomic novelty detection.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

# Placeholder for actual taxonomic databases and tools
# In a real implementation, these would integrate with:
# - BLAST+ for sequence alignment
# - Kraken2/Bracken for k-mer based classification
# - QIIME2 for phylogenetic analysis
# - Custom ML models for novel species detection

logger = logging.getLogger(__name__)


class AdvancedTaxonomicAnalyzer:
    """Enhanced taxonomic analysis with confidence scoring and phylogenetic insights."""
    
    def __init__(self, reference_db_path: Optional[str] = None):
        """
        Initialize advanced taxonomic analyzer.
        
        Args:
            reference_db_path: Path to reference database
        """
        self.reference_db_path = reference_db_path
        self.taxonomic_ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
    def analyze_taxonomic_composition(self, sequences: List, 
                                    confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Perform comprehensive taxonomic analysis with confidence scoring.
        
        Args:
            sequences: List of sequence records
            confidence_threshold: Minimum confidence for reliable assignments
            
        Returns:
            Comprehensive taxonomic analysis results
        """
        results = {
            'total_sequences': len(sequences),
            'assigned_sequences': 0,
            'high_confidence_assignments': 0,
            'taxonomic_distribution': {},
            'confidence_statistics': {},
            'novel_candidates': [],
            'phylogenetic_insights': {},
            'alpha_diversity': {},
            'beta_diversity_prep': {}
        }
        
        # Simulate taxonomic assignments with confidence scores
        assignments = self._perform_taxonomic_assignment(sequences)
        
        # Process assignments
        for assignment in assignments:
            if assignment['confidence'] >= confidence_threshold:
                results['high_confidence_assignments'] += 1
            
            if assignment['taxonomy'] != 'Unknown':
                results['assigned_sequences'] += 1
        
        # Calculate taxonomic distribution at each rank
        results['taxonomic_distribution'] = self._calculate_rank_distributions(assignments)
        
        # Calculate confidence statistics
        results['confidence_statistics'] = self._calculate_confidence_stats(assignments)
        
        # Identify potential novel taxa
        results['novel_candidates'] = self._identify_novel_candidates(assignments, confidence_threshold)
        
        # Calculate alpha diversity metrics
        results['alpha_diversity'] = self._calculate_alpha_diversity(assignments)
        
        # Prepare data for beta diversity analysis
        results['beta_diversity_prep'] = self._prepare_beta_diversity_data(assignments)
        
        # Phylogenetic insights
        results['phylogenetic_insights'] = self._analyze_phylogenetic_patterns(assignments)
        
        return results
    
    def _perform_taxonomic_assignment(self, sequences: List) -> List[Dict[str, Any]]:
        """Simulate taxonomic assignment with confidence scores."""
        assignments = []
        
        # Simulate different taxonomic groups with varying confidence
        taxa_templates = [
            {'kingdom': 'Bacteria', 'phylum': 'Proteobacteria', 'class': 'Gammaproteobacteria'},
            {'kingdom': 'Bacteria', 'phylum': 'Firmicutes', 'class': 'Bacilli'},
            {'kingdom': 'Archaea', 'phylum': 'Euryarchaeota', 'class': 'Methanobacteria'},
            {'kingdom': 'Eukaryota', 'phylum': 'Stramenopiles', 'class': 'Diatoms'},
            {'kingdom': 'Eukaryota', 'phylum': 'Alveolata', 'class': 'Dinoflagellates'},
        ]
        
        for i, seq in enumerate(sequences):
            # Simulate assignment probability
            assignment_prob = np.random.random()
            
            if assignment_prob > 0.15:  # 85% get some assignment
                template = taxa_templates[np.random.randint(len(taxa_templates))]
                confidence = np.random.beta(7, 2)  # Skewed toward higher confidence
                
                # Build full taxonomy string
                taxonomy_parts = []
                for rank in self.taxonomic_ranks:
                    if rank in template:
                        taxonomy_parts.append(f"{rank}__{template[rank]}")
                    elif confidence > 0.6:  # Add more specific ranks for high confidence
                        taxonomy_parts.append(f"{rank}__Unknown_{rank}_{i % 100}")
                    else:
                        break
                
                taxonomy = ';'.join(taxonomy_parts)
                
                assignments.append({
                    'sequence_id': getattr(seq, 'id', f'seq_{i}'),
                    'taxonomy': taxonomy,
                    'confidence': confidence,
                    'method': 'BLAST+ML_ensemble',
                    'e_value': np.random.exponential(1e-20),
                    'identity': confidence * 100,
                    'coverage': np.random.uniform(0.7, 1.0)
                })
            else:
                # Unassigned
                assignments.append({
                    'sequence_id': getattr(seq, 'id', f'seq_{i}'),
                    'taxonomy': 'Unknown',
                    'confidence': 0.0,
                    'method': 'unassigned',
                    'e_value': None,
                    'identity': 0,
                    'coverage': 0
                })
        
        return assignments
    
    def _calculate_rank_distributions(self, assignments: List[Dict]) -> Dict[str, Dict]:
        """Calculate taxonomic distribution at each rank."""
        rank_distributions = {}
        
        for rank in self.taxonomic_ranks:
            rank_counts = Counter()
            total_assigned = 0
            
            for assignment in assignments:
                if assignment['taxonomy'] != 'Unknown':
                    # Parse taxonomy string
                    taxonomy_parts = assignment['taxonomy'].split(';')
                    rank_assignment = None
                    
                    for part in taxonomy_parts:
                        if part.startswith(f"{rank}__"):
                            rank_assignment = part.replace(f"{rank}__", "")
                            break
                    
                    if rank_assignment:
                        rank_counts[rank_assignment] += 1
                        total_assigned += 1
            
            # Calculate percentages and statistics
            rank_distributions[rank] = {
                'counts': dict(rank_counts.most_common(20)),  # Top 20
                'total_assigned': total_assigned,
                'unique_taxa': len(rank_counts),
                'top_taxon': rank_counts.most_common(1)[0] if rank_counts else ('Unknown', 0),
                'assignment_rate': total_assigned / len(assignments) if assignments else 0
            }
        
        return rank_distributions
    
    def _calculate_confidence_stats(self, assignments: List[Dict]) -> Dict[str, float]:
        """Calculate confidence score statistics."""
        confidences = [a['confidence'] for a in assignments if a['confidence'] > 0]
        
        if not confidences:
            return {'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'median': float(np.median(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'q25': float(np.percentile(confidences, 25)),
            'q75': float(np.percentile(confidences, 75)),
            'high_confidence_rate': float(sum(1 for c in confidences if c > 0.8) / len(confidences))
        }
    
    def _identify_novel_candidates(self, assignments: List[Dict], threshold: float) -> List[Dict]:
        """Identify potential novel taxa based on confidence and taxonomy patterns."""
        novel_candidates = []
        
        for assignment in assignments:
            # Criteria for novel candidates:
            # 1. Low confidence in existing databases
            # 2. Unique taxonomic patterns
            # 3. High sequence quality but poor matches
            
            if (assignment['confidence'] < threshold and 
                assignment['confidence'] > 0.3 and  # Not completely unassigned
                assignment.get('identity', 0) is not None and
                assignment.get('identity', 0) < 90):  # Low identity to known sequences
                
                novel_candidates.append({
                    'sequence_id': assignment['sequence_id'],
                    'best_match_taxonomy': assignment['taxonomy'],
                    'confidence': assignment['confidence'],
                    'identity': assignment.get('identity', 0),
                    'novelty_score': 1 - assignment['confidence'],
                    'potential_rank': self._estimate_novel_rank(assignment)
                })
        
        # Sort by novelty score
        novel_candidates.sort(key=lambda x: x['novelty_score'], reverse=True)
        
        return novel_candidates[:50]  # Top 50 candidates
    
    def _estimate_novel_rank(self, assignment: Dict) -> str:
        """Estimate the taxonomic rank at which novelty occurs."""
        identity = assignment.get('identity', 0)
        
        # Handle None values
        if identity is None:
            identity = 0
        
        if identity < 70:
            return 'genus_or_higher'
        elif identity < 85:
            return 'species'
        elif identity < 95:
            return 'subspecies_or_strain'
        else:
            return 'sequence_variant'
    
    def _calculate_alpha_diversity(self, assignments: List[Dict]) -> Dict[str, float]:
        """Calculate alpha diversity metrics at different taxonomic ranks."""
        diversity_metrics = {}
        
        for rank in ['genus', 'species', 'family']:
            taxa = []
            for assignment in assignments:
                if assignment['taxonomy'] != 'Unknown':
                    # Extract taxon at specific rank
                    taxonomy_parts = assignment['taxonomy'].split(';')
                    for part in taxonomy_parts:
                        if part.startswith(f"{rank}__"):
                            taxa.append(part.replace(f"{rank}__", ""))
                            break
            
            if taxa:
                taxon_counts = Counter(taxa)
                total = sum(taxon_counts.values())
                proportions = [count/total for count in taxon_counts.values()]
                
                # Calculate Shannon diversity
                shannon = -sum(p * np.log(p) for p in proportions if p > 0)
                
                # Calculate Simpson diversity
                simpson = 1 - sum(p**2 for p in proportions)
                
                # Calculate Chao1 richness estimator
                singletons = sum(1 for count in taxon_counts.values() if count == 1)
                doubletons = sum(1 for count in taxon_counts.values() if count == 2)
                chao1 = len(taxon_counts) + (singletons**2) / (2 * doubletons) if doubletons > 0 else len(taxon_counts)
                
                diversity_metrics[f"{rank}_shannon"] = shannon
                diversity_metrics[f"{rank}_simpson"] = simpson
                diversity_metrics[f"{rank}_richness"] = len(taxon_counts)
                diversity_metrics[f"{rank}_chao1"] = chao1
                diversity_metrics[f"{rank}_evenness"] = shannon / np.log(len(taxon_counts)) if len(taxon_counts) > 1 else 0
        
        return diversity_metrics
    
    def _prepare_beta_diversity_data(self, assignments: List[Dict]) -> Dict[str, Any]:
        """Prepare data structures for beta diversity analysis."""
        # Create abundance matrix for beta diversity calculations
        genus_counts = Counter()
        species_counts = Counter()
        
        for assignment in assignments:
            if assignment['taxonomy'] != 'Unknown':
                taxonomy_parts = assignment['taxonomy'].split(';')
                
                for part in taxonomy_parts:
                    if part.startswith('genus__'):
                        genus_counts[part.replace('genus__', '')] += 1
                    elif part.startswith('species__'):
                        species_counts[part.replace('species__', '')] += 1
        
        return {
            'genus_abundance_vector': dict(genus_counts),
            'species_abundance_vector': dict(species_counts),
            'total_genera': len(genus_counts),
            'total_species': len(species_counts),
            'rare_taxa_threshold': 5,  # Taxa with < 5 sequences
            'rare_genera': [genus for genus, count in genus_counts.items() if count < 5],
            'dominant_genera': dict(genus_counts.most_common(10))
        }
    
    def _analyze_phylogenetic_patterns(self, assignments: List[Dict]) -> Dict[str, Any]:
        """Analyze phylogenetic patterns in the data."""
        # Simulate phylogenetic analysis results
        kingdom_counts = Counter()
        phylum_counts = Counter()
        
        for assignment in assignments:
            if assignment['taxonomy'] != 'Unknown':
                taxonomy_parts = assignment['taxonomy'].split(';')
                
                for part in taxonomy_parts:
                    if part.startswith('kingdom__'):
                        kingdom_counts[part.replace('kingdom__', '')] += 1
                    elif part.startswith('phylum__'):
                        phylum_counts[part.replace('phylum__', '')] += 1
        
        # Calculate phylogenetic diversity metrics
        total_sequences = len(assignments)
        
        return {
            'kingdom_diversity': {
                'count': len(kingdom_counts),
                'distribution': dict(kingdom_counts),
                'evenness': self._calculate_evenness(kingdom_counts),
                'dominance': max(kingdom_counts.values()) / total_sequences if kingdom_counts else 0
            },
            'phylum_diversity': {
                'count': len(phylum_counts),
                'distribution': dict(phylum_counts.most_common(15)),
                'evenness': self._calculate_evenness(phylum_counts),
                'rare_phyla': [phylum for phylum, count in phylum_counts.items() if count < 10]
            },
            'phylogenetic_breadth': len(kingdom_counts) * len(phylum_counts),
            'cross_domain_contamination': self._detect_contamination(kingdom_counts)
        }
    
    def _calculate_evenness(self, counts: Counter) -> float:
        """Calculate evenness (Pielou's J) for a distribution."""
        if len(counts) <= 1:
            return 0.0
        
        total = sum(counts.values())
        proportions = [count/total for count in counts.values()]
        shannon = -sum(p * np.log(p) for p in proportions if p > 0)
        
        return shannon / np.log(len(counts))
    
    def _detect_contamination(self, kingdom_counts: Counter) -> Dict[str, Any]:
        """Detect potential cross-domain contamination."""
        total = sum(kingdom_counts.values())
        
        contamination_indicators = {
            'multiple_domains': len(kingdom_counts) > 1,
            'unexpected_archaea': kingdom_counts.get('Archaea', 0) > total * 0.05,  # >5% archaea might be unexpected
            'low_eukaryote_ratio': kingdom_counts.get('Eukaryota', 0) < total * 0.1,  # <10% eukaryotes might be low for marine samples
            'contamination_score': 1 - max(kingdom_counts.values()) / total if total > 0 else 0
        }
        
        return contamination_indicators