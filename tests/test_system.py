"""
Test suite for eDNA biodiversity assessment system
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.tokenizer import DNATokenizer, SequenceDataset
from models.embeddings import DNATransformerEmbedder, DNAAutoencoder
from clustering.algorithms import EmbeddingClusterer
from clustering.taxonomy import MLTaxonomyClassifier
from novelty.detection import NoveltyDetector, NoveltyAnalyzer
from preprocessing.pipeline import SequenceQualityFilter, PreprocessingPipeline

class TestDNATokenizer(unittest.TestCase):
    """Test DNA tokenizer functionality"""
    
    def setUp(self):
        self.sequences = [
            "ATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTA",
            "TTAATTAATTAATTAATTAA"
        ]
        self.tokenizer = DNATokenizer(encoding_type="kmer", kmer_size=3)
    
    def test_kmer_generation(self):
        """Test k-mer generation"""
        sequence = "ATCGATCG"
        kmers = self.tokenizer.sequence_to_kmers(sequence)
        expected_kmers = ["ATC", "TCG", "CGA", "GAT", "ATC", "TCG"]
        self.assertEqual(kmers, expected_kmers)
    
    def test_sequence_encoding(self):
        """Test sequence encoding"""
        sequence = "ATCGATCG"
        encoded = self.tokenizer.encode_sequence(sequence, max_length=20)
        
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)
        self.assertEqual(len(encoded['input_ids']), 20)
        self.assertEqual(len(encoded['attention_mask']), 20)
    
    def test_batch_encoding(self):
        """Test batch encoding"""
        encoded = self.tokenizer.encode_sequences(self.sequences, max_length=15)
        
        self.assertEqual(encoded['input_ids'].shape[0], len(self.sequences))
        self.assertEqual(encoded['input_ids'].shape[1], 15)
    
    def test_tokenizer_save_load(self):
        """Test tokenizer save/load functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "tokenizer.pkl"
            
            # Save tokenizer
            self.tokenizer.save(save_path)
            
            # Load tokenizer
            loaded_tokenizer = DNATokenizer.load(save_path)
            
            # Test that loaded tokenizer works the same
            original_encoded = self.tokenizer.encode_sequence(self.sequences[0])
            loaded_encoded = loaded_tokenizer.encode_sequence(self.sequences[0])
            
            np.testing.assert_array_equal(
                original_encoded['input_ids'], 
                loaded_encoded['input_ids']
            )

class TestEmbeddingModels(unittest.TestCase):
    """Test embedding models"""
    
    def setUp(self):
        self.tokenizer = DNATokenizer(encoding_type="kmer", kmer_size=3)
        self.vocab_size = self.tokenizer.vocab_size
        self.sequences = [
            "ATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTA", 
            "TTAATTAATTAATTAATTAA"
        ]
    
    def test_transformer_embedder(self):
        """Test transformer embedder"""
        model = DNATransformerEmbedder(
            vocab_size=self.vocab_size,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        
        # Encode sequences
        encoded = self.tokenizer.encode_sequences(self.sequences, max_length=20)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Get embeddings
        import torch
        with torch.no_grad():
            embeddings = model(torch.tensor(input_ids), torch.tensor(attention_mask))
        
        self.assertEqual(embeddings.shape[0], len(self.sequences))
        self.assertEqual(embeddings.shape[1], 64)
    
    def test_autoencoder(self):
        """Test autoencoder model"""
        model = DNAAutoencoder(
            vocab_size=self.vocab_size,
            embedding_dim=32,
            hidden_dims=[64, 128, 64],
            latent_dim=16
        )
        
        # Encode sequences
        encoded = self.tokenizer.encode_sequences(self.sequences, max_length=20)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Get embeddings and reconstruction
        import torch
        with torch.no_grad():
            latent, reconstructed = model(torch.tensor(input_ids), torch.tensor(attention_mask))
        
        self.assertEqual(latent.shape[0], len(self.sequences))
        self.assertEqual(latent.shape[1], 16)
        self.assertEqual(reconstructed.shape[0], len(self.sequences))

class TestClustering(unittest.TestCase):
    """Test clustering algorithms"""
    
    def setUp(self):
        # Create mock embeddings
        np.random.seed(42)
        self.embeddings = np.random.randn(100, 64)
        self.sequences = [f"SEQUENCE_{i}" for i in range(100)]
    
    def test_hdbscan_clustering(self):
        """Test HDBSCAN clustering"""
        clusterer = EmbeddingClusterer(method="hdbscan", min_cluster_size=5)
        labels = clusterer.fit(self.embeddings)
        
        self.assertEqual(len(labels), len(self.embeddings))
        self.assertIsNotNone(clusterer.cluster_stats)
        self.assertGreaterEqual(clusterer.cluster_stats['n_clusters'], 0)
    
    def test_kmeans_clustering(self):
        """Test K-means clustering"""
        clusterer = EmbeddingClusterer(method="kmeans", n_clusters=5)
        labels = clusterer.fit(self.embeddings)
        
        self.assertEqual(len(labels), len(self.embeddings))
        # K-means should find exactly 5 clusters
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 5)
    
    def test_cluster_representatives(self):
        """Test cluster representative selection"""
        clusterer = EmbeddingClusterer(method="kmeans", n_clusters=3)
        clusterer.fit(self.embeddings)
        
        representatives = clusterer.get_cluster_representatives(self.sequences, n_representatives=2)
        
        # Should have representatives for each cluster (excluding noise)
        self.assertGreater(len(representatives), 0)
        
        # Each cluster should have requested number of representatives
        for cluster_id, reps in representatives.items():
            self.assertLessEqual(len(reps), 2)

class TestTaxonomy(unittest.TestCase):
    """Test taxonomy classification"""
    
    def setUp(self):
        # Create mock training data
        np.random.seed(42)
        self.train_embeddings = np.random.randn(200, 64)
        self.train_labels = np.random.choice(['Bacteria', 'Archaea', 'Eukaryota'], size=200)
        
        # Create test data
        self.test_embeddings = np.random.randn(50, 64)
    
    def test_ml_classifier_training(self):
        """Test ML taxonomy classifier training"""
        classifier = MLTaxonomyClassifier(model_type="random_forest")
        
        results = classifier.train(self.train_embeddings, self.train_labels)
        
        self.assertTrue(classifier.is_trained)
        self.assertIn('train_accuracy', results)
        self.assertIn('val_accuracy', results)
        self.assertGreater(results['val_accuracy'], 0.0)
    
    def test_ml_classifier_prediction(self):
        """Test ML taxonomy classifier prediction"""
        classifier = MLTaxonomyClassifier()
        classifier.train(self.train_embeddings, self.train_labels)
        
        predictions = classifier.predict(self.test_embeddings)
        
        self.assertEqual(len(predictions), len(self.test_embeddings))
        
        # Check prediction structure
        for pred in predictions:
            self.assertIn('predicted_taxonomy', pred)
            self.assertIn('confidence', pred)
            self.assertIn('top_predictions', pred)
            self.assertGreaterEqual(pred['confidence'], 0.0)
            self.assertLessEqual(pred['confidence'], 1.0)
    
    def test_classifier_save_load(self):
        """Test classifier save/load functionality"""
        classifier = MLTaxonomyClassifier()
        classifier.train(self.train_embeddings, self.train_labels)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "classifier.pkl"
            
            # Save classifier
            classifier.save_model(save_path)
            
            # Load classifier
            new_classifier = MLTaxonomyClassifier()
            new_classifier.load_model(save_path)
            
            # Test predictions are the same
            original_pred = classifier.predict(self.test_embeddings[:5])
            loaded_pred = new_classifier.predict(self.test_embeddings[:5])
            
            for orig, loaded in zip(original_pred, loaded_pred):
                self.assertEqual(orig['predicted_taxonomy'], loaded['predicted_taxonomy'])

class TestNoveltyDetection(unittest.TestCase):
    """Test novelty detection"""
    
    def setUp(self):
        np.random.seed(42)
        # Reference data (known taxa)
        self.reference_embeddings = np.random.randn(200, 64)
        
        # Query data (mix of known and novel)
        self.query_embeddings = np.vstack([
            np.random.randn(80, 64),  # Similar to reference
            np.random.randn(20, 64) * 3  # Outliers (novel)
        ])
        
        self.query_sequences = [f"QUERY_{i}" for i in range(100)]
    
    def test_novelty_detector(self):
        """Test basic novelty detector"""
        detector = NoveltyDetector(method="isolation_forest")
        detector.fit(self.reference_embeddings)
        
        predictions = detector.predict(self.query_embeddings)
        scores = detector.decision_function(self.query_embeddings)
        
        self.assertEqual(len(predictions), len(self.query_embeddings))
        self.assertEqual(len(scores), len(self.query_embeddings))
        
        # Should detect some novel points
        novel_count = np.sum(predictions == -1)
        self.assertGreater(novel_count, 0)
    
    def test_novelty_analyzer(self):
        """Test comprehensive novelty analyzer"""
        analyzer = NoveltyAnalyzer()
        
        results = analyzer.analyze_novelty(
            query_embeddings=self.query_embeddings,
            reference_embeddings=self.reference_embeddings,
            query_sequences=self.query_sequences
        )
        
        self.assertIn('total_sequences', results)
        self.assertIn('novel_candidates', results)
        self.assertIn('novel_percentage', results)
        self.assertIn('predictions', results)
        self.assertEqual(results['total_sequences'], len(self.query_sequences))

class TestPreprocessing(unittest.TestCase):
    """Test preprocessing pipeline"""
    
    def setUp(self):
        self.sequences = [
            "ATCGATCGATCGATCGATCGATCGATCGATCG",  # Good sequence
            "ATCGNNNNNGATCGATCGATCGATCGATCGATC",  # Too many Ns
            "ATCG",  # Too short
            "A" * 1000,  # Too long
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"  # Good sequence
        ]
    
    def test_quality_filter(self):
        """Test sequence quality filtering"""
        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq
        
        filter_obj = SequenceQualityFilter(
            min_length=10,
            max_length=500,
            max_n_bases=3
        )
        
        # Create mock sequence records
        records = []
        for i, seq in enumerate(self.sequences):
            record = SeqRecord(Seq(seq), id=f"seq_{i}")
            records.append(record)
        
        # Test filtering
        filtered_records = [record for record in records if filter_obj.filter_sequence(record)]
        
        # Should filter out sequences with issues
        self.assertLess(len(filtered_records), len(records))
        
        # Check that good sequences pass
        filtered_seqs = [str(record.seq) for record in filtered_records]
        self.assertIn(self.sequences[0], filtered_seqs)  # Good sequence 1
        self.assertIn(self.sequences[4], filtered_seqs)  # Good sequence 2

def run_performance_tests():
    """Run performance benchmarks"""
    print("Running performance tests...")
    
    # Test embedding generation performance
    import time
    
    tokenizer = DNATokenizer(encoding_type="kmer", kmer_size=6)
    sequences = ["ATCGATCGATCGATCGATCG"] * 1000
    
    start_time = time.time()
    encoded = tokenizer.encode_sequences(sequences, max_length=50)
    encoding_time = time.time() - start_time
    
    print(f"Tokenization of 1000 sequences: {encoding_time:.2f} seconds")
    
    # Test clustering performance
    embeddings = np.random.randn(1000, 128)
    
    start_time = time.time()
    clusterer = EmbeddingClusterer(method="hdbscan")
    clusterer.fit(embeddings)
    clustering_time = time.time() - start_time
    
    print(f"Clustering of 1000 embeddings: {clustering_time:.2f} seconds")
    
    # Test novelty detection performance
    reference_embeddings = np.random.randn(500, 128)
    query_embeddings = np.random.randn(200, 128)
    
    start_time = time.time()
    detector = NoveltyDetector()
    detector.fit(reference_embeddings)
    predictions = detector.predict(query_embeddings)
    novelty_time = time.time() - start_time
    
    print(f"Novelty detection on 200 queries: {novelty_time:.2f} seconds")

def main():
    """Run all tests"""
    print("Running eDNA Biodiversity Assessment System Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 60)
    
    # Run performance tests
    run_performance_tests()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()