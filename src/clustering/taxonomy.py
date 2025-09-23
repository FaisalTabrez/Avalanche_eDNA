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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """Parse BLAST XML results"""
        results = []
        
        # Parse XML
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
                'all_hits': []
            }
            
            if record.alignments:
                # Process all hits
                for alignment in record.alignments:
                    hit_def = alignment.hit_def
                    
                    for hsp in alignment.hsps:
                        identity = (hsp.identities / hsp.align_length) * 100
                        coverage = (hsp.align_length / record.query_length) * 100
                        
                        hit_info = {
                            'hit_def': hit_def,
                            'identity': identity,
                            'evalue': hsp.expect,
                            'coverage': coverage,
                            'alignment_length': hsp.align_length,
                            'taxonomy': self._extract_taxonomy_from_hit(hit_def)
                        }
                        
                        result_dict['all_hits'].append(hit_info)
                        
                        # Update best hit if this is better
                        if (identity >= self.identity_threshold and 
                            identity > result_dict['identity']):
                            result_dict.update({
                                'best_hit': hit_def,
                                'identity': identity,
                                'evalue': hsp.expect,
                                'coverage': coverage,
                                'taxonomy': hit_info['taxonomy']
                            })
            
            results.append(result_dict)
        
        return results
    
    def _extract_taxonomy_from_hit(self, hit_def: str) -> str:
        """Extract taxonomy information from BLAST hit definition"""
        # This is a simplified taxonomy extraction
        # In practice, you'd want more sophisticated parsing
        
        # Try to extract organism name from description
        # Common patterns in BLAST hit definitions
        patterns = [
            r'(\w+\s+\w+)',  # Genus species
            r'\[([^\]]+)\]',  # Text in brackets
        ]
        
        for pattern in patterns:
            match = re.search(pattern, hit_def)
            if match:
                return match.group(1)
        
        return "Unknown"

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

class HybridTaxonomyAssigner:
    """Hybrid taxonomy assignment combining BLAST and ML approaches"""
    
    def __init__(self,
                 blast_assigner: BlastTaxonomyAssigner,
                 ml_classifier: Optional[MLTaxonomyClassifier] = None,
                 confidence_threshold: float = 0.8):
        """
        Initialize hybrid taxonomy assigner
        
        Args:
            blast_assigner: BLAST-based assigner
            ml_classifier: Optional ML classifier
            confidence_threshold: Confidence threshold for ML predictions
        """
        self.blast_assigner = blast_assigner
        self.ml_classifier = ml_classifier
        self.confidence_threshold = confidence_threshold
        
        logger.info("Hybrid taxonomy assigner initialized")
    
    def assign_taxonomy(self,
                       sequences: List[str],
                       embeddings: Optional[np.ndarray] = None,
                       sequence_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Assign taxonomy using hybrid approach
        
        Args:
            sequences: List of DNA sequences
            embeddings: Optional sequence embeddings for ML classification
            sequence_ids: Optional sequence IDs
            
        Returns:
            List of taxonomy assignments
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        logger.info(f"Assigning taxonomy to {len(sequences)} sequences using hybrid approach")
        
        # Get BLAST results
        blast_results = self.blast_assigner.assign_taxonomy(sequences, sequence_ids)
        
        # Get ML results if classifier and embeddings are available
        ml_results = None
        if self.ml_classifier and self.ml_classifier.is_trained and embeddings is not None:
            ml_results = self.ml_classifier.predict(embeddings)
        
        # Combine results
        combined_results = []
        for i, blast_result in enumerate(blast_results):
            result = {
                'sequence_id': blast_result['sequence_id'],
                'blast_taxonomy': blast_result['taxonomy'],
                'blast_identity': blast_result['identity'],
                'blast_evalue': blast_result['evalue'],
                'ml_taxonomy': None,
                'ml_confidence': None,
                'final_taxonomy': None,
                'assignment_method': None,
                'confidence_score': 0.0
            }
            
            # Add ML results if available
            if ml_results:
                ml_result = ml_results[i]
                result['ml_taxonomy'] = ml_result['predicted_taxonomy']
                result['ml_confidence'] = ml_result['confidence']
            
            # Determine final assignment
            result.update(self._determine_final_assignment(blast_result, ml_results[i] if ml_results else None))
            
            combined_results.append(result)
        
        return combined_results
    
    def _determine_final_assignment(self,
                                  blast_result: Dict[str, Any],
                                  ml_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine final taxonomy assignment"""
        
        # Priority rules:
        # 1. High confidence BLAST hit (>97% identity)
        # 2. High confidence ML prediction
        # 3. Lower confidence BLAST hit
        # 4. Lower confidence ML prediction
        # 5. Unknown
        
        assignment = {
            'final_taxonomy': 'Unknown',
            'assignment_method': 'none',
            'confidence_score': 0.0
        }
        
        # Check high-confidence BLAST
        if (blast_result['identity'] >= self.blast_assigner.identity_threshold and
            blast_result['taxonomy'] != 'Unknown'):
            assignment.update({
                'final_taxonomy': blast_result['taxonomy'],
                'assignment_method': 'blast_high_confidence',
                'confidence_score': blast_result['identity'] / 100.0
            })
            return assignment
        
        # Check high-confidence ML
        if (ml_result and 
            ml_result['confidence'] >= self.confidence_threshold):
            assignment.update({
                'final_taxonomy': ml_result['predicted_taxonomy'],
                'assignment_method': 'ml_high_confidence',
                'confidence_score': ml_result['confidence']
            })
            return assignment
        
        # Check lower-confidence BLAST
        if (blast_result['identity'] >= 80.0 and  # Lower threshold
            blast_result['taxonomy'] != 'Unknown'):
            assignment.update({
                'final_taxonomy': blast_result['taxonomy'],
                'assignment_method': 'blast_low_confidence',
                'confidence_score': blast_result['identity'] / 100.0
            })
            return assignment
        
        # Check lower-confidence ML
        if ml_result and ml_result['confidence'] >= 0.5:
            assignment.update({
                'final_taxonomy': ml_result['predicted_taxonomy'],
                'assignment_method': 'ml_low_confidence',
                'confidence_score': ml_result['confidence']
            })
            return assignment
        
        return assignment
    
    def generate_assignment_report(self, 
                                 results: List[Dict[str, Any]],
                                 save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate taxonomy assignment report
        
        Args:
            results: List of assignment results
            save_path: Optional path to save report
            
        Returns:
            DataFrame with assignment summary
        """
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Summary statistics
        method_counts = df['assignment_method'].value_counts()
        confidence_stats = df.groupby('assignment_method')['confidence_score'].describe()
        
        # Print summary
        logger.info("Taxonomy Assignment Summary:")
        logger.info(f"Total sequences: {len(results)}")
        
        for method, count in method_counts.items():
            percentage = (count / len(results)) * 100
            logger.info(f"{method}: {count} ({percentage:.1f}%)")
        
        # Save if requested
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Assignment report saved to {save_path}")
        
        return df

def main():
    """Main function for testing taxonomy assignment"""
    # Create mock data for testing
    sequences = [
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAAT"
    ]
    
    sequence_ids = ["seq1", "seq2", "seq3"]
    
    # Create mock embeddings
    embeddings = np.random.randn(3, 256)
    
    # Test ML classifier (if we have training data)
    logger.info("Testing ML taxonomy classifier...")
    
    # Create mock training data
    train_embeddings = np.random.randn(100, 256)
    train_labels = np.random.choice(['Bacteria', 'Archaea', 'Eukaryota'], size=100)
    
    ml_classifier = MLTaxonomyClassifier()
    training_results = ml_classifier.train(train_embeddings, train_labels)
    
    logger.info(f"Training results: {training_results}")
    
    # Test prediction
    predictions = ml_classifier.predict(embeddings)
    
    for i, pred in enumerate(predictions):
        logger.info(f"Sequence {i+1}: {pred['predicted_taxonomy']} "
                   f"(confidence: {pred['confidence']:.3f})")
    
    # Save model
    model_save_path = Path("models") / "taxonomy_classifier.pkl"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    ml_classifier.save_model(model_save_path)
    
    logger.info("Taxonomy assignment testing complete!")

if __name__ == "__main__":
    main()