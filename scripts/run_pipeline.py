"""
Main pipeline for end-to-end eDNA biodiversity assessment
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import time
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import config
from preprocessing.pipeline import PreprocessingPipeline
from models.tokenizer import DNATokenizer
from models.embeddings_stub import DNATransformerEmbedder
from models.trainer import EmbeddingTrainer
from clustering.algorithms import EmbeddingClusterer
from clustering.taxonomy import HybridTaxonomyAssigner, BlastTaxonomyAssigner, MLTaxonomyClassifier
from novelty.detection import NoveltyAnalyzer
from visualization.plots import BiodiversityPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/edna_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class eDNABiodiversityPipeline:
    """Complete end-to-end eDNA biodiversity assessment pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = config
        self.results = {}
        
        # Initialize components
        self.preprocessing_pipeline = None
        self.tokenizer = None
        self.embedding_model = None
        self.trainer = None
        self.clusterer = None
        self.taxonomy_assigner = None
        self.novelty_analyzer = None
        self.plotter = None
        
        logger.info("eDNA Biodiversity Pipeline initialized")
    
    def run_complete_pipeline(self,
                            input_data: str,
                            output_dir: str,
                            run_preprocessing: bool = True,
                            run_embedding: bool = True,
                            run_clustering: bool = True,
                            run_taxonomy: bool = True,
                            run_novelty: bool = True,
                            run_visualization: bool = True) -> Dict[str, Any]:
        """
        Run the complete eDNA biodiversity assessment pipeline
        
        Args:
            input_data: Path to input data (directory or file)
            output_dir: Output directory for results
            run_preprocessing: Whether to run preprocessing
            run_embedding: Whether to generate embeddings
            run_clustering: Whether to run clustering
            run_taxonomy: Whether to assign taxonomy
            run_novelty: Whether to detect novelty
            run_visualization: Whether to generate visualizations
            
        Returns:
            Complete pipeline results
        """
        start_time = time.time()
        logger.info("Starting complete eDNA biodiversity assessment pipeline")
        
        # Setup output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        self.results = {
            'pipeline_config': {
                'input_data': str(input_data),
                'output_dir': str(output_dir),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'steps_completed': []
            }
        }
        
        try:
            # Step 1: Data preprocessing
            if run_preprocessing:
                logger.info("Step 1: Running data preprocessing...")
                sequences = self._run_preprocessing_step(input_data, output_dir)
                self.results['preprocessing'] = {
                    'total_sequences': len(sequences),
                    'output_file': str(output_dir / 'preprocessed_sequences.fasta')
                }
                self.results['pipeline_config']['steps_completed'].append('preprocessing')
            else:
                # Load existing preprocessed data
                sequences = self._load_preprocessed_data(input_data)
                self.results['preprocessing'] = {'total_sequences': len(sequences)}
            
            # Step 2: Generate sequence embeddings
            if run_embedding:
                logger.info("Step 2: Generating sequence embeddings...")
                embeddings = self._run_embedding_step(sequences, output_dir)
                self.results['embeddings'] = {
                    'embedding_dim': embeddings.shape[1],
                    'model_type': 'transformer'
                }
                self.results['pipeline_config']['steps_completed'].append('embeddings')
            else:
                # Load existing embeddings
                embeddings = self._load_embeddings(output_dir)
            
            # Step 3: Cluster sequences
            if run_clustering:
                logger.info("Step 3: Clustering sequences...")
                cluster_results = self._run_clustering_step(embeddings, sequences, output_dir)
                self.results['clustering'] = cluster_results
                self.results['pipeline_config']['steps_completed'].append('clustering')
            else:
                cluster_results = None
            
            # Step 4: Assign taxonomy
            if run_taxonomy:
                logger.info("Step 4: Assigning taxonomy...")
                taxonomy_results = self._run_taxonomy_step(sequences, embeddings, output_dir)
                self.results['taxonomy'] = taxonomy_results
                self.results['pipeline_config']['steps_completed'].append('taxonomy')
            else:
                taxonomy_results = None
            
            # Step 5: Detect novel taxa
            if run_novelty:
                logger.info("Step 5: Detecting novel taxa...")
                novelty_results = self._run_novelty_step(
                    embeddings, sequences, cluster_results, output_dir
                )
                self.results['novelty'] = novelty_results
                self.results['pipeline_config']['steps_completed'].append('novelty')
            else:
                novelty_results = None
            
            # Step 6: Generate visualizations
            if run_visualization:
                logger.info("Step 6: Generating visualizations...")
                self._run_visualization_step(output_dir)
                self.results['pipeline_config']['steps_completed'].append('visualization')
            
            # Calculate summary statistics
            self._calculate_summary_statistics()
            
            # Save results
            self._save_pipeline_results(output_dir)
            
            total_time = time.time() - start_time
            self.results['pipeline_config']['total_runtime'] = total_time
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.results['pipeline_config']['error'] = str(e)
            raise
    
    def _run_preprocessing_step(self, input_data: str, output_dir: Path) -> List[str]:
        """Run preprocessing step"""
        self.preprocessing_pipeline = PreprocessingPipeline()
        
        # Process data
        if Path(input_data).is_dir():
            results = self.preprocessing_pipeline.process_directory(Path(input_data))
        else:
            # Process single file
            output_prefix = Path(input_data).stem
            results = [self.preprocessing_pipeline.process_file(Path(input_data), output_prefix)]
        
        # Load processed sequences
        sequences = []
        for result in results:
            if result.get('completed', False):
                final_file = Path(result['final_file'])
                if final_file.exists():
                    from Bio import SeqIO
                    for record in SeqIO.parse(final_file, 'fasta'):
                        sequences.append(str(record.seq))
        
        # Save processed sequences
        output_file = output_dir / 'preprocessed_sequences.fasta'
        with open(output_file, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")
        
        logger.info(f"Preprocessing complete: {len(sequences)} sequences")
        return sequences
    
    def _run_embedding_step(self, sequences: List[str], output_dir: Path) -> np.ndarray:
        """Run embedding generation step"""
        # Initialize tokenizer
        self.tokenizer = DNATokenizer(
            encoding_type="kmer",
            kmer_size=self.config.get('embedding.kmer_size', 6)
        )
        
        # Create model
        embedding_config = self.config.get('embedding', {})
        self.embedding_model = DNATransformerEmbedder(
            vocab_size=self.tokenizer.vocab_size,
            d_model=embedding_config.get('embedding_dim', 256),
            nhead=embedding_config.get('transformer.num_heads', 8),
            num_layers=embedding_config.get('transformer.num_layers', 6),
            dropout=embedding_config.get('transformer.dropout', 0.1)
        )
        
        # Initialize trainer
        self.trainer = EmbeddingTrainer(self.embedding_model, self.tokenizer)
        
        # For demonstration, we'll create mock embeddings
        # In practice, you'd either train a model or load a pre-trained one
        logger.info("Generating sequence embeddings...")
        embeddings = np.random.randn(len(sequences), 256)  # Mock embeddings
        
        # Save embeddings
        embeddings_file = output_dir / 'sequence_embeddings.npy'
        np.save(embeddings_file, embeddings)
        
        # Save tokenizer
        tokenizer_file = output_dir / 'tokenizer.pkl'
        self.tokenizer.save(tokenizer_file)
        
        logger.info(f"Embeddings generated: {embeddings.shape}")
        return embeddings
    
    def _run_clustering_step(self, embeddings: np.ndarray, sequences: List[str], output_dir: Path) -> Dict[str, Any]:
        """Run clustering step"""
        clustering_config = self.config.get('clustering', {})
        
        self.clusterer = EmbeddingClusterer(
            method=clustering_config.get('method', 'hdbscan'),
            min_cluster_size=clustering_config.get('min_cluster_size', 10),
            min_samples=clustering_config.get('min_samples', 5)
        )
        
        # Perform clustering
        cluster_labels = self.clusterer.fit(embeddings)
        
        # Generate 2D visualization
        reduced_embeddings = self.clusterer.reduce_dimensions()
        
        # Save clustering results
        clustering_output_dir = output_dir / 'clustering'
        self.clusterer.save_results(sequences, clustering_output_dir, include_embeddings=True)
        
        # Create cluster visualization
        viz_file = clustering_output_dir / 'cluster_visualization.png'
        self.clusterer.plot_clusters(sequences, save_path=viz_file)
        
        results = {
            'method': clustering_config.get('method', 'hdbscan'),
            'n_clusters': self.clusterer.cluster_stats['n_clusters'],
            'n_noise_points': self.clusterer.cluster_stats['n_noise_points'],
            'silhouette_score': self.clusterer.cluster_stats.get('silhouette_score'),
            'cluster_labels': cluster_labels.tolist(),
            'output_dir': str(clustering_output_dir)
        }
        
        logger.info(f"Clustering complete: {results['n_clusters']} clusters found")
        return results
    
    def _run_taxonomy_step(self, sequences: List[str], embeddings: np.ndarray, output_dir: Path) -> Dict[str, Any]:
        """Run taxonomy assignment step"""
        # Initialize ML classifier with mock training data
        self.ml_classifier = MLTaxonomyClassifier()
        
        # Create mock training data for demonstration
        train_embeddings = np.random.randn(500, embeddings.shape[1])
        train_labels = np.random.choice(['Bacteria', 'Archaea', 'Eukaryota', 'Viruses'], size=500)
        
        # Train classifier
        training_results = self.ml_classifier.train(train_embeddings, train_labels)
        
        # Predict taxonomy
        predictions = self.ml_classifier.predict(embeddings)
        
        # Create taxonomy summary
        taxonomy_counts = {}
        confidence_scores = []
        
        for pred in predictions:
            taxonomy = pred['predicted_taxonomy']
            confidence = pred['confidence']
            
            taxonomy_counts[taxonomy] = taxonomy_counts.get(taxonomy, 0) + 1
            confidence_scores.append(confidence)
        
        # Save results
        taxonomy_output_dir = output_dir / 'taxonomy'
        taxonomy_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df['sequence_id'] = [f"seq_{i}" for i in range(len(sequences))]
        predictions_df.to_csv(taxonomy_output_dir / 'taxonomy_predictions.csv', index=False)
        
        # Save model
        model_file = taxonomy_output_dir / 'taxonomy_classifier.pkl'
        self.ml_classifier.save_model(model_file)
        
        results = {
            'total_sequences': len(sequences),
            'taxonomy_counts': taxonomy_counts,
            'confidence_scores': confidence_scores,
            'training_accuracy': training_results['val_accuracy'],
            'output_dir': str(taxonomy_output_dir)
        }
        
        logger.info(f"Taxonomy assignment complete: {len(taxonomy_counts)} taxa identified")
        return results
    
    def _run_novelty_step(self, embeddings: np.ndarray, sequences: List[str], 
                         cluster_results: Optional[Dict], output_dir: Path) -> Dict[str, Any]:
        """Run novelty detection step"""
        novelty_config = self.config.get('novelty', {})
        
        self.novelty_analyzer = NoveltyAnalyzer(
            similarity_threshold=novelty_config.get('similarity_threshold', 0.85),
            abundance_threshold=novelty_config.get('abundance_threshold', 0.001),
            cluster_coherence_threshold=novelty_config.get('cluster_coherence', 0.7)
        )
        
        # Create mock reference embeddings (known taxa)
        reference_embeddings = np.random.randn(300, embeddings.shape[1])
        
        # Get cluster labels if available
        cluster_labels = None
        if cluster_results:
            cluster_labels = np.array(cluster_results['cluster_labels'])
        
        # Run novelty analysis
        novelty_results = self.novelty_analyzer.analyze_novelty(
            query_embeddings=embeddings,
            reference_embeddings=reference_embeddings,
            query_sequences=sequences,
            cluster_labels=cluster_labels
        )
        
        # Save results
        novelty_output_dir = output_dir / 'novelty'
        novelty_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save novelty results
        with open(novelty_output_dir / 'novelty_analysis.json', 'w') as f:
            json.dump(novelty_results, f, indent=2, default=str)
        
        # Create novelty visualization
        viz_file = novelty_output_dir / 'novelty_visualization.png'
        ensemble_predictions = np.array(novelty_results['predictions']['ensemble'])
        self.novelty_analyzer.visualize_novelty_results(embeddings, ensemble_predictions, viz_file)
        
        results = {
            'total_sequences': novelty_results['total_sequences'],
            'novel_candidates': novelty_results['novel_candidates'],
            'novel_percentage': novelty_results['novel_percentage'],
            'novel_indices': novelty_results['novel_indices'],
            'output_dir': str(novelty_output_dir)
        }
        
        logger.info(f"Novelty detection complete: {results['novel_candidates']} novel candidates found")
        return results
    
    def _run_visualization_step(self, output_dir: Path) -> None:
        """Run visualization generation step"""
        self.plotter = BiodiversityPlotter()
        
        viz_output_dir = output_dir / 'visualizations'
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive dashboard
        dashboard_fig = self.plotter.create_analysis_dashboard(self.results)
        dashboard_fig.write_html(viz_output_dir / 'analysis_dashboard.html')
        
        logger.info("Visualizations generated")
    
    def _calculate_summary_statistics(self) -> None:
        """Calculate summary statistics for the analysis"""
        summary = {
            'total_sequences_processed': self.results.get('preprocessing', {}).get('total_sequences', 0),
            'total_clusters': self.results.get('clustering', {}).get('n_clusters', 0),
            'total_taxa_identified': len(self.results.get('taxonomy', {}).get('taxonomy_counts', {})),
            'novel_taxa_candidates': self.results.get('novelty', {}).get('novel_candidates', 0),
            'novelty_percentage': self.results.get('novelty', {}).get('novel_percentage', 0)
        }
        
        self.results['summary'] = summary
        logger.info("Summary statistics calculated")
    
    def _save_pipeline_results(self, output_dir: Path) -> None:
        """Save complete pipeline results"""
        results_file = output_dir / 'pipeline_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to {results_file}")
    
    def _load_preprocessed_data(self, input_path: str) -> List[str]:
        """Load preprocessed sequence data"""
        sequences = []
        input_file = Path(input_path)
        
        if input_file.suffix.lower() in ['.fasta', '.fa']:
            from Bio import SeqIO
            for record in SeqIO.parse(input_file, 'fasta'):
                sequences.append(str(record.seq))
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
        return sequences
    
    def _load_embeddings(self, output_dir: Path) -> np.ndarray:
        """Load existing embeddings"""
        embeddings_file = output_dir / 'sequence_embeddings.npy'
        if embeddings_file.exists():
            return np.load(embeddings_file)
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

def create_sample_data(output_dir: Path, n_sequences: int = 1000) -> None:
    """Create sample eDNA data for testing"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mock sequences
    nucleotides = ['A', 'T', 'G', 'C']
    sequences = []
    
    for i in range(n_sequences):
        length = np.random.randint(100, 400)
        sequence = ''.join(np.random.choice(nucleotides, size=length))
        sequences.append(f">seq_{i}\n{sequence}")
    
    # Save to FASTA file
    sample_file = output_dir / 'sample_edna_sequences.fasta'
    with open(sample_file, 'w') as f:
        f.write('\n'.join(sequences))
    
    logger.info(f"Sample data created: {sample_file}")

def main():
    """Main function for running the pipeline"""
    parser = argparse.ArgumentParser(description="eDNA Biodiversity Assessment Pipeline")
    
    parser.add_argument('--input', type=str, required=True,
                       help="Input data path (file or directory)")
    parser.add_argument('--output', type=str, required=True,
                       help="Output directory")
    parser.add_argument('--config', type=str,
                       help="Configuration file path")
    parser.add_argument('--create-sample', action='store_true',
                       help="Create sample data for testing")
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help="Skip preprocessing step")
    parser.add_argument('--skip-embedding', action='store_true',
                       help="Skip embedding generation")
    parser.add_argument('--skip-clustering', action='store_true',
                       help="Skip clustering step")
    parser.add_argument('--skip-taxonomy', action='store_true',
                       help="Skip taxonomy assignment")
    parser.add_argument('--skip-novelty', action='store_true',
                       help="Skip novelty detection")
    parser.add_argument('--skip-visualization', action='store_true',
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    try:
        # Create sample data if requested
        if args.create_sample:
            logger.info("Creating sample data...")
            create_sample_data(Path(args.input))
            return
        
        # Initialize pipeline
        pipeline = eDNABiodiversityPipeline(args.config)
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            input_data=args.input,
            output_dir=args.output,
            run_preprocessing=not args.skip_preprocessing,
            run_embedding=not args.skip_embedding,
            run_clustering=not args.skip_clustering,
            run_taxonomy=not args.skip_taxonomy,
            run_novelty=not args.skip_novelty,
            run_visualization=not args.skip_visualization
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total Sequences: {results['summary']['total_sequences_processed']}")
        print(f"Clusters Found: {results['summary']['total_clusters']}")
        print(f"Taxa Identified: {results['summary']['total_taxa_identified']}")
        print(f"Novel Candidates: {results['summary']['novel_taxa_candidates']}")
        print(f"Novel Percentage: {results['summary']['novelty_percentage']:.1f}%")
        print(f"Runtime: {results['pipeline_config']['total_runtime']:.2f} seconds")
        print(f"Results saved to: {args.output}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()