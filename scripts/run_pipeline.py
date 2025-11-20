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
from clustering.algorithms import EmbeddingClusterer
from clustering.taxonomy import HybridTaxonomyAssigner, BlastTaxonomyAssigner, MLTaxonomyClassifier, TaxonomyIndex, KNNLCATaxonomyAssigner
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
                            run_visualization: bool = True,
                            train_model: bool = False,
                            custom_model_path: Optional[str] = None) -> Dict[str, Any]:
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
            train_model: Whether to train a custom embedding model
            custom_model_path: Path to pre-trained custom model (overrides training)
            
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
            
            # Step 1.5: Train custom model if requested
            if train_model and not custom_model_path:
                logger.info("Step 1.5: Training custom embedding model...")
                model_results = self._run_training_step(sequences, output_dir)
                custom_model_path = model_results['model_path']
                self.results['model_training'] = model_results
                self.results['pipeline_config']['steps_completed'].append('model_training')
            
            # Step 2: Generate sequence embeddings
            if run_embedding:
                logger.info("Step 2: Generating sequence embeddings...")
                embeddings = self._run_embedding_step(sequences, output_dir, custom_model_path)
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

        # Check if input is SRA data
        input_path = Path(input_data)
        if self._is_sra_data(input_path):
            return self._process_sra_data(input_path, output_dir)

        # Process regular data
        if input_path.is_dir():
            results = self.preprocessing_pipeline.process_directory(input_path)
        else:
            # Process single file
            output_prefix = input_path.stem
            results = [self.preprocessing_pipeline.process_file(input_path, output_prefix)]

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

    def _is_sra_data(self, input_path: Path) -> bool:
        """Check if input path contains SRA data"""
        if input_path.is_dir():
            # Check if directory contains SRA files
            sra_files = list(input_path.rglob("*.sra"))
            fastq_files = list(input_path.rglob("*.fastq*"))
            return len(sra_files) > 0 or len(fastq_files) > 0
        else:
            # Check if file is SRA-related
            return input_path.suffix.lower() in ['.sra', '.fastq', '.fq']

    def _process_sra_data(self, input_path: Path, output_dir: Path) -> List[str]:
        """Process SRA data using SRA processor"""
        try:
            from preprocessing.sra_processor import SRAProcessor
        except ImportError as e:
            raise ImportError("SRA processor not available. Please ensure src/preprocessing/sra_processor.py exists.") from e

        logger.info("Processing SRA data...")

        sra_processor = SRAProcessor()

        # Find SRA files
        if input_path.is_dir():
            sra_files = list(input_path.rglob("*.sra"))
            fastq_files = list(input_path.rglob("*.fastq*"))

            if fastq_files:
                # Process existing FASTQ files
                logger.info(f"Found {len(fastq_files)} FASTQ files from SRA")
                results = sra_processor.integrate_with_pipeline(fastq_files, output_dir)
                sequences = [str(seq.seq) for seq in results['sequences']]
            elif sra_files:
                # Convert SRA files to FASTQ first
                logger.info(f"Found {len(sra_files)} SRA files, converting to FASTQ...")
                fastq_files = []
                for sra_file in sra_files:
                    # Use the SRA downloader to convert
                    from download_sra_data import SRADownloader
                    sra_downloader = SRADownloader()
                    converted_files = sra_downloader.convert_sra_to_fastq(sra_file)
                    fastq_files.extend(converted_files)

                if fastq_files:
                    results = sra_processor.integrate_with_pipeline(fastq_files, output_dir)
                    sequences = [str(seq.seq) for seq in results['sequences']]
                else:
                    raise ValueError("No FASTQ files could be generated from SRA files")
            else:
                raise ValueError(f"No SRA or FASTQ files found in {input_path}")
        else:
            # Process single SRA/FASTQ file
            if input_path.suffix.lower() in ['.fastq', '.fq', '.fastq.gz', '.fq.gz']:
                sequences = sra_processor.process_sra_fastq(input_path, output_dir)
                sequences = [str(seq.seq) for seq in sequences]
            else:
                raise ValueError(f"Unsupported SRA file format: {input_path.suffix}")

        logger.info(f"SRA processing complete: {len(sequences)} sequences")
        return sequences
    
    def _run_embedding_step(self, sequences: List[str], output_dir: Path, custom_model_path: Optional[str] = None) -> np.ndarray:
        """Run embedding generation step using Nucleotide Transformer or custom model"""
        
        # Use custom trained model if provided
        if custom_model_path:
            logger.info(f"Using custom trained model from: {custom_model_path}")
            return self._extract_custom_embeddings(sequences, custom_model_path, output_dir)
        
        # Otherwise use Hugging Face Nucleotide Transformer
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError(
                "Transformers/PyTorch not installed. Please install requirements (transformers, torch) "
                "to use Nucleotide Transformer embeddings."
            ) from e

        embedding_cfg = self.config.get('embedding', {})
        transformer_cfg = embedding_cfg.get('transformer', {}) or {}

        model_id = transformer_cfg.get('model_id', 'InstaDeepAI/nucleotide-transformer-250m-1000g')
        max_len = embedding_cfg.get('max_sequence_length', 512)
        stride = transformer_cfg.get('stride', 128)
        batch_size = transformer_cfg.get('batch_size', 8)

        # Device selection
        use_gpu = bool(self.config.get('performance.use_gpu', True))
        device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        logger.info(f"Loading Nucleotide Transformer model: {model_id} on device: {device}")

        # Load model and tokenizer lazily to avoid overhead when skipping step
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.to(device)
        model.eval()

        def chunk_ids(ids: list[int], max_length: int, overlap: int) -> list[list[int]]:
            chunks = []
            i = 0
            while i < len(ids):
                chunk = ids[i:i+max_length]
                if not chunk:
                    break
                chunks.append(chunk)
                if i + max_length >= len(ids):
                    break
                step = max_length - overlap
                if step <= 0:
                    step = max_length
                i += step
            return chunks

        all_embeddings: list[np.ndarray] = []

        def process_batch(batch_seqs: list[str]) -> list[np.ndarray]:
            seq_vecs: list[np.ndarray] = []
            with torch.no_grad():
                for seq in batch_seqs:
                    enc = tokenizer(seq, add_special_tokens=True, return_tensors='pt', truncation=False)
                    ids = enc['input_ids'][0].tolist()
                    # Chunk long sequences
                    id_chunks = chunk_ids(ids, max_len, stride)
                    chunk_vecs: list[torch.Tensor] = []
                    for ch in id_chunks:
                        inputs = {'input_ids': torch.tensor([ch], dtype=torch.long, device=device)}
                        outputs = model(**inputs)
                        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]  # [1, L, D]
                        # Use attention_mask if available to exclude padding and specials
                        # Here we mean-pool across tokens
                        vec = last_hidden.mean(dim=1)  # [1, D]
                        chunk_vecs.append(vec.squeeze(0).detach().cpu())
                    if chunk_vecs:
                        seq_vec = torch.stack(chunk_vecs, dim=0).mean(dim=0)
                        seq_vecs.append(seq_vec.numpy().astype('float32'))
                    else:
                        # Fallback zero vector if tokenization failed
                        hidden = getattr(model.config, 'hidden_size', 256)
                        seq_vecs.append(np.zeros((hidden,), dtype=np.float32))
            return seq_vecs

        # Batch processing to control memory footprint
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            all_embeddings.extend(process_batch(batch))
            logger.info(f"Embedded {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")

        embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)

        # Optional PCA and L2 normalization
        post_cfg = embedding_cfg.get('postprocess', {})
        pca_cfg = (post_cfg.get('pca') or {}) if isinstance(post_cfg, dict) else {}

        # PCA to reduce dimensionality if enabled
        try:
            if isinstance(post_cfg, dict) and pca_cfg.get('enabled', True):
                from sklearn.decomposition import PCA
                n_components = int(pca_cfg.get('n_components', 256))
                n_components = max(1, min(n_components, embeddings.shape[1]))
                logger.info(f"Applying PCA to {n_components} components on embeddings of shape {embeddings.shape}")
                pca = PCA(n_components=n_components, random_state=int(pca_cfg.get('random_state', 42)), whiten=bool(pca_cfg.get('whiten', False)))
                embeddings = pca.fit_transform(embeddings).astype(np.float32)
        except Exception as e:
            logger.warning(f"PCA post-processing failed ({e}). Continuing without PCA.")

        # L2 normalize row-wise (per-sequence)
        try:
            l2_norm = True if not isinstance(post_cfg, dict) else bool(post_cfg.get('l2_normalize', True))
            if l2_norm:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
                logger.info("Applied L2 normalization to embeddings")
        except Exception as e:
            logger.warning(f"L2 normalization failed ({e}). Continuing without normalization.")

        # Persist
        embeddings_file = output_dir / 'sequence_embeddings.npy'
        np.save(embeddings_file, embeddings)
        logger.info(f"Embeddings generated via {model_id}: {embeddings.shape}")
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
        """Run taxonomy assignment step using KNN-LCA with BLAST fallback and cluster consensus"""
        taxonomy_cfg = self.config.get('taxonomy', {})
        knn_cfg = taxonomy_cfg.get('knn', {})
        blast_cfg = taxonomy_cfg.get('blast_fallback', {})
        consensus_cfg = taxonomy_cfg.get('cluster_consensus', {})

        taxonomy_output_dir = output_dir / 'taxonomy'
        taxonomy_output_dir.mkdir(parents=True, exist_ok=True)

        # Attempt to build/load KNN index
        knn_assigner = None
        try:
            import os
            ref_dir = Path(knn_cfg.get('reference_dir', 'data/reference'))
            emb_path = Path(knn_cfg.get('embeddings_path', ref_dir / 'reference_embeddings.npy'))
            labels_path = Path(knn_cfg.get('labels_path', ref_dir / 'reference_labels.csv'))
            if emb_path.exists() and labels_path.exists():
                ref_embeddings = np.load(emb_path)
                labels_df = pd.read_csv(labels_path)
                index = TaxonomyIndex(ref_embeddings, labels_df, normalize=True, index_type=knn_cfg.get('index', 'flat_ip'))
                knn_assigner = KNNLCATaxonomyAssigner(
                    taxonomy_index=index,
                    k=int(knn_cfg.get('k', 50)),
                    min_similarity=float(knn_cfg.get('min_similarity', 0.65)),
                    distance_margin=float(knn_cfg.get('distance_margin', 0.07)),
                    min_agreement=knn_cfg.get('min_agreement', {"species": 0.8, "genus": 0.7, "family": 0.6})
                )
            else:
                logger.warning(f"Reference files not found for KNN taxonomy: {emb_path} or {labels_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize KNN taxonomy index: {e}")

        # Optional BLAST assigner
        blast_assigner = None
        if bool(blast_cfg.get('enable', True)):
            try:
                blast_assigner = BlastTaxonomyAssigner(
                    blast_db=str(blast_cfg.get('database', taxonomy_cfg.get('blast', {}).get('database', 'data/reference/nt'))),
                    evalue=float(taxonomy_cfg.get('blast', {}).get('evalue', 1e-5)),
                    max_targets=int(taxonomy_cfg.get('blast', {}).get('max_targets', 5)),
                    identity_threshold=float(blast_cfg.get('min_identity_species', taxonomy_cfg.get('blast', {}).get('identity_threshold', 97.0)))
                )
            except Exception as e:
                logger.warning(f"BLAST assigner unavailable: {e}")

        # Cluster labels (if clustering was run)
        cluster_labels = None
        try:
            cluster_labels = np.array(self.results.get('clustering', {}).get('cluster_labels')) if 'clustering' in self.results else None
        except Exception:
            cluster_labels = None

        sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        # If no KNN index is available but BLAST is, run pure BLAST taxonomy to avoid hybrid path issues
        if knn_assigner is None and blast_assigner is not None:
            try:
                logger.info("Running pure BLAST taxonomy assignment (no KNN index available)...")
                assignments = blast_assigner.assign_taxonomy(sequences, sequence_ids=sequence_ids)
                # Normalize fields to match downstream expectations
                for a in assignments:
                    a.update({
                        'assigned_rank': 'species' if a.get('taxonomy') and a.get('identity', 0.0) >= blast_assigner.identity_threshold else None,
                        'assigned_label': a.get('taxonomy'),
                        'confidence': min(0.99, a.get('identity', 0.0) / 100.0) if a.get('identity') is not None else 0.0,
                        'blast_identity': a.get('identity'),
                        'blast_taxid': a.get('taxid'),
                        'blast_label': a.get('taxonomy')
                    })
            except Exception as e:
                logger.warning(f"Pure BLAST taxonomy failed: {e}")
                assignments = []
        else:
            # Run hybrid assignment
            hybrid = HybridTaxonomyAssigner(
                blast_assigner=blast_assigner,
                ml_classifier=None,
                confidence_threshold=float(knn_cfg.get('species_confidence', 0.8)),
                knn_assigner=knn_assigner,
                cluster_consensus_threshold=float(consensus_cfg.get('min_cluster_agreement', 0.7))
            )
            assignments = hybrid.assign_taxonomy(
                sequences=sequences,
                embeddings=embeddings,
                sequence_ids=sequence_ids,
                cluster_labels=cluster_labels
            )

        # Summarize results
        taxonomy_counts: Dict[str, int] = {}
        confidence_scores: List[float] = []
        for a in assignments:
            label = a.get('assigned_label', 'Unknown') if a.get('assigned_label') else 'Unknown'
            taxonomy_counts[label] = taxonomy_counts.get(label, 0) + 1
            confidence_scores.append(float(a.get('confidence', 0.0)))

        # Save predictions (with optional lineage enrichment)
        predictions_df = pd.DataFrame(assignments)

        # Tie-breaker report: compare KNN species vs BLAST species
        try:
            species_conf = float(knn_cfg.get('species_confidence', 0.8))
            blast_thr = float(taxonomy_cfg.get('blast', {}).get('identity_threshold', 97.0))
            def decide(row):
                knn_sp = row.get('knn_label') if row.get('knn_rank') == 'species' else None
                blast_sp = row.get('blast_label') or row.get('taxonomy')
                blast_id = row.get('blast_identity')
                knn_conf = row.get('knn_confidence')
                conflict = (isinstance(knn_sp, str) and isinstance(blast_sp, str) and knn_sp and blast_sp and (knn_sp.strip() != blast_sp.strip()))
                if isinstance(blast_id, (int, float)) and blast_id >= blast_thr:
                    winner = 'blast'
                    reason = f'blast_identity>={blast_thr}'
                elif isinstance(knn_conf, (int, float)) and knn_conf >= species_conf:
                    winner = 'knn'
                    reason = f'knn_confidence>={species_conf}'
                else:
                    # Fallback: prefer BLAST if identity present, else KNN if confidence present, else unknown
                    if isinstance(blast_id, (int, float)):
                        winner = 'blast'
                        reason = 'blast_identity_fallback'
                    elif isinstance(knn_conf, (int, float)):
                        winner = 'knn'
                        reason = 'knn_confidence_fallback'
                    else:
                        winner = 'unknown'
                        reason = 'insufficient_evidence'
                return pd.Series({'tiebreak_winner': winner, 'tiebreak_reason': reason, 'knn_species': knn_sp, 'blast_species': blast_sp, 'conflict_flag': bool(conflict)})
            tb = predictions_df.apply(decide, axis=1)
            predictions_df = pd.concat([predictions_df.reset_index(drop=True), tb.reset_index(drop=True)], axis=1)
        except Exception as e:
            logger.warning(f"Tie-breaker report skipped due to error: {e}")

        # Optional lineage enrichment via NCBI taxdump with priority: BLAST taxid -> KNN name
        taxdump_dir = taxonomy_cfg.get('taxdump_dir')
        try:
            if taxdump_dir and Path(taxdump_dir).exists():
                from clustering.taxonomy import TaxdumpResolver
                resolver = TaxdumpResolver(taxdump_dir)
                if resolver.available():
                    enriched = []
                    for _, row in predictions_df.iterrows():
                        taxid = row.get('blast_taxid') or row.get('taxid')
                        lin = None
                        if pd.notna(taxid):
                            try:
                                lin = resolver.lineage_by_taxid(int(taxid))
                            except Exception:
                                lin = None
                        if not lin:
                            name = row.get('assigned_label') or row.get('taxonomy')
                            lin = resolver.lineage_by_name(name if isinstance(name, str) else None)
                        enriched.append(lin)
                    lin_df = pd.DataFrame(enriched)
                    predictions_df = pd.concat([predictions_df.reset_index(drop=True), lin_df.reset_index(drop=True)], axis=1)
        except Exception as e:
            logger.warning(f"Lineage enrichment skipped due to error: {e}")

        # Write main predictions
        predictions_path = taxonomy_output_dir / 'taxonomy_predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)

        # Write conflict-focused report
        try:
            conflict_df = predictions_df[predictions_df['conflict_flag'] == True].copy()
            conflict_report = taxonomy_output_dir / 'taxonomy_tiebreak_report.csv'
            conflict_df.to_csv(conflict_report, index=False)
        except Exception:
            pass

        results = {
            'total_sequences': len(sequences),
            'taxonomy_counts': taxonomy_counts,
            'confidence_scores': confidence_scores,
            'output_dir': str(taxonomy_output_dir)
        }

        logger.info(f"Taxonomy assignment complete: {len(taxonomy_counts)} unique labels identified")
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
    
    def _run_training_step(self, sequences: List[str], output_dir: Path) -> Dict[str, Any]:
        """Train a custom embedding model on the dataset"""
        try:
            from src.models.tokenizer import DNATokenizer
            from src.models.embeddings import DNATransformerEmbedder, DNAContrastiveModel, ModelFactory
            from src.models.trainer import EmbeddingTrainer
        except ImportError as e:
            raise ImportError(
                "Model training modules not available. Please ensure src/models is properly set up."
            ) from e
        
        training_config = self.config.get('embedding', {}).get('training', {})
        
        model_type = training_config.get('model_type', 'contrastive')
        kmer_size = self.config.get('embedding', {}).get('kmer_size', 6)
        batch_size = training_config.get('batch_size', 32)
        epochs = training_config.get('epochs', 100)
        learning_rate = training_config.get('learning_rate', 1e-4)
        device = training_config.get('device', 'auto')
        
        logger.info(f"Training {model_type} model with {len(sequences)} sequences")
        
        # Create tokenizer
        self.tokenizer = DNATokenizer(encoding_type='kmer', kmer_size=kmer_size)
        
        # Create model
        d_model = self.config.get('embedding', {}).get('embedding_dim', 256)
        num_layers = self.config.get('embedding', {}).get('transformer', {}).get('num_layers', 6)
        num_heads = self.config.get('embedding', {}).get('transformer', {}).get('num_heads', 8)
        
        if model_type == 'contrastive':
            backbone = DNATransformerEmbedder(
                vocab_size=self.tokenizer.vocab_size,
                d_model=d_model,
                nhead=num_heads,
                num_layers=num_layers
            )
            projection_dim = training_config.get('projection_dim', 128)
            temperature = training_config.get('temperature', 0.1)
            
            self.embedding_model = DNAContrastiveModel(
                backbone_model=backbone,
                projection_dim=projection_dim,
                temperature=temperature
            )
        else:
            self.embedding_model = DNATransformerEmbedder(
                vocab_size=self.tokenizer.vocab_size,
                d_model=d_model,
                nhead=num_heads,
                num_layers=num_layers
            )
        
        # Create trainer
        self.trainer = EmbeddingTrainer(self.embedding_model, self.tokenizer, device=device)
        
        # Prepare data
        train_loader, val_loader = self.trainer.prepare_data(
            sequences=sequences,
            labels=None,  # Unsupervised for now
            validation_split=training_config.get('validation_split', 0.2),
            batch_size=batch_size
        )
        
        # Train model
        if model_type in ['contrastive', 'transformer']:
            history = self.trainer.train_contrastive(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate
            )
        else:
            history = self.trainer.train_autoencoder(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate
            )
        
        # Save trained model
        model_dir = output_dir / 'trained_model'
        self.trainer.save_model(str(model_dir / 'model'), include_tokenizer=True)
        
        results = {
            'model_type': model_type,
            'model_path': str(model_dir / 'model'),
            'num_sequences': len(sequences),
            'epochs': epochs,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
        
        logger.info(f"Model training complete: {results}")
        return results
    
    def _extract_custom_embeddings(self, sequences: List[str], model_path: str, output_dir: Path) -> np.ndarray:
        """Extract embeddings using a custom trained model"""
        try:
            from src.models.tokenizer import DNATokenizer
            from src.models.embeddings import DNATransformerEmbedder, DNAContrastiveModel
            from src.models.trainer import EmbeddingTrainer
        except ImportError as e:
            raise ImportError(
                "Model training modules not available. Please ensure src/models is properly set up."
            ) from e
        
        logger.info(f"Loading custom model from {model_path}")
        
        # Load tokenizer
        import pickle
        tokenizer_path = Path(model_path).parent / 'tokenizer.pkl'
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Create model (will be loaded with weights)
        # For now, create a default model - the load will override weights
        backbone = DNATransformerEmbedder(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            num_layers=6
        )
        self.embedding_model = DNAContrastiveModel(
            backbone_model=backbone,
            projection_dim=128
        )
        
        # Create trainer and load model
        self.trainer = EmbeddingTrainer(self.embedding_model, self.tokenizer, device='auto')
        self.trainer.load_model(model_path)
        
        # Extract embeddings
        logger.info(f"Extracting embeddings for {len(sequences)} sequences")
        embeddings = self.trainer.extract_embeddings(sequences, batch_size=32)
        
        # Save embeddings
        embeddings_file = output_dir / 'sequence_embeddings.npy'
        np.save(embeddings_file, embeddings)
        logger.info(f"Custom model embeddings saved: {embeddings.shape}")
        
        return embeddings

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
    parser.add_argument('--train-model', action='store_true',
                       help="Train custom embedding model before analysis")
    parser.add_argument('--model-path', type=str,
                       help="Path to trained model to use for embeddings (instead of Hugging Face)")
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
            run_visualization=not args.skip_visualization,
            train_model=args.train_model,
            custom_model_path=args.model_path
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
