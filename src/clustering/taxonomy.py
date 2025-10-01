"""
Taxonomic assignment using traditional methods (BLAST) and machine learning
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Blast import NCBIXML
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
from functools import lru_cache

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxdumpResolver:
    """Resolve scientific names to full lineage using NCBI taxdump (names.dmp, nodes.dmp).

    Builds:
      - _name_to_taxid: canonical scientific name (lowercased) -> taxid
      - _taxid_to_name: taxid -> primary scientific name
      - _nodes: taxid -> (parent_taxid, rank)
      - optional _merged_old2new: remapped taxids from merged.dmp
    """
    def __init__(self, taxdump_dir: Optional[str]):
        self.taxdump_dir = Path(taxdump_dir) if taxdump_dir else None
        self._name_to_taxid: Optional[Dict[str, int]] = None
        self._taxid_to_name: Optional[Dict[int, str]] = None
        self._nodes: Optional[Dict[int, Tuple[int, str]]] = None  # taxid -> (parent_taxid, rank)
        self._merged_old2new: Optional[Dict[int, int]] = None

    def available(self) -> bool:
        return bool(self.taxdump_dir) \
            and (self.taxdump_dir / 'names.dmp').exists() \
            and (self.taxdump_dir / 'nodes.dmp').exists()

    def _load(self) -> None:
        if self._name_to_taxid is not None and self._nodes is not None and self._taxid_to_name is not None:
            return
        name_to_taxid: Dict[str, int] = {}
        taxid_to_name: Dict[int, str] = {}
        nodes: Dict[int, Tuple[int, str]] = {}
        merged: Dict[int, int] = {}
        names_path = self.taxdump_dir / 'names.dmp'  # type: ignore
        nodes_path = self.taxdump_dir / 'nodes.dmp'  # type: ignore
        merged_path = self.taxdump_dir / 'merged.dmp'  # type: ignore
        # Parse names.dmp: tax_id | name_txt | unique name | name class |
        with open(names_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 4:
                    continue
                taxid_str, name_txt, _unique, name_class = parts[:4]
                if name_class == 'scientific name':
                    try:
                        taxid = int(taxid_str)
                    except ValueError:
                        continue
                    sci_name = name_txt.strip()
                    key = sci_name.lower()
                    if key not in name_to_taxid:
                        name_to_taxid[key] = taxid
                    # Prefer first scientific name as primary label
                    if taxid not in taxid_to_name:
                        taxid_to_name[taxid] = sci_name
        # Parse nodes.dmp: tax_id | parent_tax_id | rank | ...
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
                rank = parts[2]
                nodes[taxid] = (parent, rank)
        # Parse merged.dmp to remap deprecated IDs (optional)
        if merged_path.exists():
            with open(merged_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 2:
                        continue
                    try:
                        old = int(parts[0]); new = int(parts[1])
                    except ValueError:
                        continue
                    merged[old] = new
        self._name_to_taxid = name_to_taxid
        self._taxid_to_name = taxid_to_name
        self._nodes = nodes
        self._merged_old2new = merged if merged else None
        logger.info("Taxdump loaded: %d names, %d nodes", len(name_to_taxid), len(nodes))

    @lru_cache(maxsize=100000)
    def lineage_by_name(self, scientific_name: Optional[str]) -> Dict[str, Optional[str]]:
        if not scientific_name:
            return {r: None for r in ['kingdom','phylum','class','order','family','genus','species']}
        self._load()
        assert self._name_to_taxid is not None
        taxid = self._name_to_taxid.get(scientific_name.lower())
        out = self.lineage_by_taxid(taxid) if taxid is not None else {r: None for r in ['kingdom','phylum','class','order','family','genus','species']}
        if out.get('species') is None:
            out['species'] = scientific_name
        return out

    @lru_cache(maxsize=100000)
    def lineage_by_taxid(self, taxid: Optional[int]) -> Dict[str, Optional[str]]:
        rank_targets = ['kingdom','phylum','class','order','family','genus','species']
        out = {r: None for r in rank_targets}
        if taxid is None or not self.available():
            return out
        self._load()
        assert self._nodes is not None and self._taxid_to_name is not None
        # Remap merged
        if self._merged_old2new and taxid in self._merged_old2new:
            taxid = self._merged_old2new[taxid]
        seen = set()
        steps = 0
        while taxid in self._nodes and taxid not in seen and steps < 100:
            seen.add(taxid)
            parent, rank = self._nodes[taxid]
            rank = rank.lower()
            if rank in out and out[rank] is None:
                out[rank] = self._taxid_to_name.get(taxid)
            if parent == taxid:
                break
            taxid = parent
            steps += 1
        return out

class BlastTaxonomyAssigner:
    """BLAST-based taxonomic assignment"""
    
    def __init__(self,
                 blast_db: str,
                 evalue: float = 1e-5,
                 max_targets: int = 10,
                 identity_threshold: float = 97.0):
        """
        Initialize BLAST taxonomy assigner
        
        Args:
            blast_db: Path to BLAST database
            evalue: E-value threshold
            max_targets: Maximum number of target sequences
            identity_threshold: Minimum identity threshold for assignment
        """
        self.blast_db = blast_db
        self.evalue = evalue
        self.max_targets = max_targets
        self.identity_threshold = identity_threshold
        
        # Check if BLAST is available
        self._check_blast_availability()
        
        logger.info(f"BLAST taxonomy assigner initialized with database: {blast_db}")
    
    def _check_blast_availability(self) -> None:
        """Check if BLAST tools are available"""
        try:
            result = subprocess.run(['blastn', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("BLAST tools not found")
        except FileNotFoundError:
            raise RuntimeError("BLAST tools not installed or not in PATH")
    
    def assign_taxonomy(self, 
                       sequences: List[str],
                       sequence_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Assign taxonomy to sequences using BLAST
        
        Args:
            sequences: List of DNA sequences
            sequence_ids: Optional list of sequence IDs
            
        Returns:
            List of taxonomy assignment results
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        logger.info(f"Assigning taxonomy to {len(sequences)} sequences using BLAST")
        
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_fasta:
            for seq_id, sequence in zip(sequence_ids, sequences):
                temp_fasta.write(f">{seq_id}\n{sequence}\n")
            temp_fasta_path = temp_fasta.name
        
        try:
            # Run BLAST
            blast_results = self._run_blast(temp_fasta_path)
            
            # Parse results
            taxonomy_assignments = self._parse_blast_results(blast_results, sequence_ids)
            
            return taxonomy_assignments
        
        finally:
            # Clean up temporary file
            os.unlink(temp_fasta_path)
        

    def _apply_cluster_consensus(self, results: List[Dict[str, Any]], cluster_labels: np.ndarray) -> List[Dict[str, Any]]:
        # Build consensus per cluster at the most specific rank with agreement >= threshold
        df = pd.DataFrame(results)
        df['cluster'] = cluster_labels
        out = results.copy()
        for cl, grp in df.groupby('cluster'):
            if cl == -1:
                continue
            # Prefer species consensus, else genus, else family
            for rank in ['species', 'genus', 'family']:
                labels = grp.loc[grp['assigned_rank'] == rank, 'assigned_label']
                if labels.empty:
                    continue
                value_counts = labels.value_counts()
                top_label, count = (value_counts.index[0], int(value_counts.iloc[0]))
                agreement = count / max(len(grp), 1)
                if agreement >= self.cluster_consensus_threshold:
                    # Propagate to low-confidence or unknowns
                    idxs = grp.index.tolist()
                    for i in idxs:
                        r = out[i]
                        if (r['assigned_rank'] is None) or (r['assigned_rank'] == rank and r['confidence'] < agreement):
                            r.update({
                                'assigned_rank': rank,
                                'assigned_label': top_label,
                                'confidence': max(r.get('confidence', 0.0), float(agreement)),
                                'source': r.get('source', 'knn') + '+consensus'
                            })
                    break  # Use the most specific rank satisfied
        return out
    
    def _run_blast(self, query_file: str) -> str:
        """Run BLAST search"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_output:
            output_file = temp_output.name
        
        try:
            cmd = [
                'blastn',
                '-query', query_file,
                '-db', self.blast_db,
                '-evalue', str(self.evalue),
                '-max_target_seqs', str(self.max_targets),
                '-outfmt', '5',  # XML format
                '-out', output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"BLAST failed: {result.stderr}")
            
            with open(output_file, 'r') as f:
                blast_output = f.read()
            
            return blast_output
        
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def _parse_blast_results(self, 
                           blast_output: str, 
                           sequence_ids: List[str]) -> List[Dict[str, Any]]:
        """Parse BLAST XML results and extract tentative taxids if present"""
        results = []
        from io import StringIO
        blast_records = NCBIXML.parse(StringIO(blast_output))
        for record in blast_records:
            seq_id = record.query
            result_dict = {
                'sequence_id': seq_id,
                'best_hit': None,
                'identity': 0.0,
                'evalue': float('inf'),
                'coverage': 0.0,
                'taxonomy': 'Unknown',
                'taxid': None,
                'all_hits': []
            }
            if record.alignments:
                for alignment in record.alignments:
                    hit_def = alignment.hit_def
                    for hsp in alignment.hsps:
                        identity = (hsp.identities / hsp.align_length) * 100
                        coverage = (hsp.align_length / record.query_length) * 100
                        taxid = self._extract_taxid_from_hit(hit_def)
                        hit_info = {
                            'hit_def': hit_def,
                            'identity': identity,
                            'evalue': hsp.expect,
                            'coverage': coverage,
                            'alignment_length': hsp.align_length,
                            'taxonomy': self._extract_taxonomy_from_hit(hit_def),
                            'taxid': taxid
                        }
                        result_dict['all_hits'].append(hit_info)
                        if (identity >= self.identity_threshold and identity > result_dict['identity']):
                            result_dict.update({
                                'best_hit': hit_def,
                                'identity': identity,
                                'evalue': hsp.expect,
                                'coverage': coverage,
                                'taxonomy': hit_info['taxonomy'],
                                'taxid': taxid
                            })
            results.append(result_dict)
        return results
    
    def _extract_taxonomy_from_hit(self, hit_def: str) -> str:
        """Extract taxonomy information from BLAST hit definition (best-effort)"""
        patterns = [
            r'\b([A-Z][a-z]+\s+[a-z][a-z\-]+)\b',  # Genus species
            r'\[([^\]]+)\]',  # Text in brackets (organism)
        ]
        for pattern in patterns:
            match = re.search(pattern, hit_def)
            if match:
                return match.group(1)
        return "Unknown"

    def _extract_taxid_from_hit(self, hit_def: str) -> Optional[int]:
        """Try to extract NCBI taxid from hit definition string"""
        for pat in [r'taxid\|(\d+)', r'TaxID=(\d+)', r'taxid=(\d+)', r'taxon:(\d+)']:
            m = re.search(pat, hit_def)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return None
        return None

class MLTaxonomyClassifier:
    """Machine learning-based taxonomic classifier"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize ML taxonomy classifier
        
        Args:
            model_type: Type of ML model ('random_forest', 'svm', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        
        self._initialize_model()
        
        logger.info(f"ML taxonomy classifier initialized with model: {model_type}")
    
    def _initialize_model(self) -> None:
        """Initialize the ML model"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, 
             embeddings: np.ndarray,
             taxonomic_labels: List[str],
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the classifier
        
        Args:
            embeddings: Sequence embeddings [n_sequences, embedding_dim]
            taxonomic_labels: List of taxonomic labels
            validation_split: Fraction of data for validation
            
        Returns:
            Training results
        """
        logger.info(f"Training {self.model_type} classifier on {len(embeddings)} sequences")
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(taxonomic_labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, encoded_labels, 
            test_size=validation_split, 
            random_state=42,
            stratify=encoded_labels
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # Classification report
        val_report = classification_report(
            y_val, val_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        self.is_trained = True
        
        results = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'classification_report': val_report,
            'n_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
        
        logger.info(f"Training complete. Validation accuracy: {val_accuracy:.4f}")
        
        return results
    
    def predict(self, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict taxonomy for embeddings
        
        Args:
            embeddings: Sequence embeddings
            
        Returns:
            List of prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions and probabilities
        predictions = self.model.predict(embeddings)
        probabilities = self.model.predict_proba(embeddings)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            taxonomy = self.label_encoder.inverse_transform([pred])[0]
            confidence = np.max(probs)
            
            # Get top 3 predictions
            top_indices = np.argsort(probs)[-3:][::-1]
            top_predictions = [
                {
                    'taxonomy': self.label_encoder.inverse_transform([idx])[0],
                    'probability': probs[idx]
                }
                for idx in top_indices
            ]
            
            results.append({
                'predicted_taxonomy': taxonomy,
                'confidence': confidence,
                'top_predictions': top_predictions
            })
        
        return results
    
    def save_model(self, save_path: Path) -> None:
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path) -> None:
        """Load a trained model"""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {load_path}")

# Optional FAISS import for KNN search
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

SUPPORTED_RANKS = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

class TaxonomyIndex:
    """FAISS-based index over reference embeddings with taxonomy labels."""
    def __init__(self, ref_embeddings: np.ndarray, labels_df: pd.DataFrame, normalize: bool = True, index_type: str = "flat_ip"):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is not available; install faiss-cpu to use KNN taxonomy")
        self.labels_df = labels_df.reset_index(drop=True)
        xb = ref_embeddings.astype(np.float32)
        self.normalize = normalize
        if self.normalize:
            xb = _l2_normalize(xb)
        d = xb.shape[1]
        if index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(d)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        self.index.add(xb)

    def search(self, Xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        Xq = Xq.astype(np.float32)
        if self.normalize:
            Xq = _l2_normalize(Xq)
        sims, idx = self.index.search(Xq, k)
        return sims, idx

class KNNLCATaxonomyAssigner:
    """KNN-based Lowest Common Ancestor (LCA) taxonomy assigner."""
    def __init__(self,
                 taxonomy_index: TaxonomyIndex,
                 ranks: Optional[List[str]] = None,
                 min_agreement: Optional[Dict[str, float]] = None,
                 min_similarity: float = 0.6,
                 distance_margin: float = 0.07,
                 k: int = 50):
        self.index = taxonomy_index
        self.ranks = ranks or SUPPORTED_RANKS
        self.min_agreement = min_agreement or {"species": 0.8, "genus": 0.7, "family": 0.6}
        self.min_similarity = min_similarity
        self.distance_margin = distance_margin
        self.k = k

    def _neighbor_labels(self, idx_row: np.ndarray) -> Dict[str, List[Optional[str]]]:
        rows = self.index.labels_df.iloc[idx_row]
        labels_by_rank: Dict[str, List[Optional[str]]] = {r: [] for r in self.ranks}
        for _, row in rows.iterrows():
            for r in self.ranks:
                if r in row.index and pd.notna(row[r]):
                    labels_by_rank[r].append(str(row[r]))
                else:
                    labels_by_rank[r].append(None)
        return labels_by_rank

    def _lca_assign(self, neighbor_labels: Dict[str, List[Optional[str]]], neighbor_sims: np.ndarray) -> Dict[str, Any]:
        sims = neighbor_sims.tolist()
        for rank in self.ranks:
            labels = neighbor_labels.get(rank, [])
            # Filter out missing
            pairs = [(lab, sim) for lab, sim in zip(labels, sims) if lab is not None and sim >= self.min_similarity]
            if not pairs:
                continue
            votes: Dict[str, float] = {}
            for lab, w in pairs:
                votes[lab] = votes.get(lab, 0.0) + float(w)
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            top_lab, top_w = sorted_votes[0]
            second_w = sorted_votes[1][1] if len(sorted_votes) > 1 else 0.0
            total_w = sum(votes.values())
            agreement = top_w / max(total_w, 1e-12)
            if agreement >= self.min_agreement.get(rank, 0.6) and (top_w - second_w) >= self.distance_margin:
                return {"rank": rank, "label": top_lab, "confidence": float(agreement), "top_weight": float(top_w), "second_weight": float(second_w)}
        return {"rank": None, "label": None, "confidence": 0.0, "top_weight": 0.0, "second_weight": 0.0}

    def assign(self, embeddings: np.ndarray, sequence_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        sims, idxs = self.index.search(embeddings, self.k)
        results: List[Dict[str, Any]] = []
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(embeddings))]
        for i in range(len(embeddings)):
            neighbor_labels = self._neighbor_labels(idxs[i])
            lca = self._lca_assign(neighbor_labels, sims[i])
            lca.update({
                "sequence_id": sequence_ids[i],
                "knn_mean_similarity": float(np.mean(sims[i] if sims.shape[1] > 0 else [0.0])),
                "knn_top_similarity": float(np.max(sims[i] if sims.shape[1] > 0 else [0.0]))
            })
            results.append(lca)
        return results

class HybridTaxonomyAssigner:
    """Hybrid taxonomy assignment combining KNN-LCA and BLAST/ML fallbacks."""
    
    def __init__(self,
                 blast_assigner: Optional[BlastTaxonomyAssigner] = None,
                 ml_classifier: Optional[MLTaxonomyClassifier] = None,
                 confidence_threshold: float = 0.8,
                 knn_assigner: Optional[KNNLCATaxonomyAssigner] = None,
                 cluster_consensus_threshold: float = 0.7):
        self.blast_assigner = blast_assigner
        self.ml_classifier = ml_classifier
        self.knn_assigner = knn_assigner
        self.confidence_threshold = confidence_threshold
        self.cluster_consensus_threshold = cluster_consensus_threshold
        logger.info("Hybrid taxonomy assigner initialized (KNN + BLAST/ML)")

    def assign_taxonomy(self,
                       sequences: List[str],
                       embeddings: Optional[np.ndarray] = None,
                       sequence_ids: Optional[List[str]] = None,
                       cluster_labels: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        results: List[Dict[str, Any]] = []

        # KNN-LCA primary
        knn_results = []
        if self.knn_assigner is not None and embeddings is not None:
            try:
                knn_results = self.knn_assigner.assign(embeddings, sequence_ids=sequence_ids)
            except Exception as e:
                logger.warning(f"KNN-LCA assignment failed: {e}")
        
        if knn_results:
            for r in knn_results:
                results.append({
                    'sequence_id': r['sequence_id'],
                    'assigned_rank': r.get('rank'),
                    'assigned_label': r.get('label'),
                    'confidence': r.get('confidence', 0.0),
                    'knn_rank': r.get('rank'),
                    'knn_label': r.get('label'),
                    'knn_confidence': r.get('confidence', 0.0),
                    'knn_top_similarity': r.get('knn_top_similarity', 0.0),
                    'knn_mean_similarity': r.get('knn_mean_similarity', 0.0),
                    'source': 'knn'
                })
        else:
            results = [{
                'sequence_id': sid,
                'assigned_rank': None,
                'assigned_label': None,
                'confidence': 0.0,
                'knn_top_similarity': 0.0,
                'knn_mean_similarity': 0.0,
                'source': 'none'
            } for sid in sequence_ids]

        # Cluster consensus
        if cluster_labels is not None:
            try:
                results = self._apply_cluster_consensus(results, cluster_labels)
            except Exception as e:
                logger.warning(f"Cluster consensus step failed: {e}")

        # BLAST fallback
        if self.blast_assigner is not None:
            try:
                to_blast = []
                seq_map = {sid: seq for sid, seq in zip(sequence_ids, sequences)}
                for r in results:
                    if (r['assigned_rank'] is None) or (r['assigned_rank'] == 'species' and r['confidence'] < self.confidence_threshold):
                        to_blast.append(r['sequence_id'])
                if to_blast:
                    logger.info(f"Running BLAST fallback for {len(to_blast)} sequences")
                    blast_sequences = [seq_map[sid] for sid in to_blast]
                    blast_assignments = self.blast_assigner.assign_taxonomy(blast_sequences, sequence_ids=to_blast)
                    blast_by_id = {a['sequence_id']: a for a in blast_assignments}
                    for r in results:
                        b = blast_by_id.get(r['sequence_id'])
                        if not b:
                            continue
                        if b.get('taxonomy') and b.get('identity', 0.0) >= self.blast_assigner.identity_threshold:
                            r.update({
                                'assigned_rank': 'species',
                                'assigned_label': b.get('taxonomy'),
                                'confidence': max(r.get('confidence', 0.0), min(0.99, b.get('identity', 0.0) / 100.0)),
                                'blast_identity': b.get('identity'),
                                'blast_taxid': b.get('taxid'),
                                'blast_label': b.get('taxonomy'),
                                'source': r.get('source', 'knn') + '+blast'
                            })
            except Exception as e:
                logger.warning(f"BLAST fallback failed or unavailable: {e}")

        return results

    def _apply_cluster_consensus(self, results: List[Dict[str, Any]], cluster_labels: np.ndarray) -> List[Dict[str, Any]]:
        df = pd.DataFrame(results)
        df['cluster'] = cluster_labels
        out = results.copy()
        for cl, grp in df.groupby('cluster'):
            if cl == -1:
                continue
            for rank in ['species', 'genus', 'family']:
                labels = grp.loc[grp['assigned_rank'] == rank, 'assigned_label']
                if labels.empty:
                    continue
                vc = labels.value_counts()
                top_label, count = (vc.index[0], int(vc.iloc[0]))
                agreement = count / max(len(grp), 1)
                if agreement >= self.cluster_consensus_threshold:
                    for idx in grp.index.tolist():
                        r = out[idx]
                        if (r['assigned_rank'] is None) or (r['assigned_rank'] == rank and r['confidence'] < agreement):
                            r.update({
                                'assigned_rank': rank,
                                'assigned_label': top_label,
                                'confidence': max(r.get('confidence', 0.0), float(agreement)),
                                'source': r.get('source', 'knn') + '+consensus'
                            })
                    break
        return out
