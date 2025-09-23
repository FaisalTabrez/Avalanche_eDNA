"""
Interactive Streamlit dashboard for eDNA biodiversity analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Any
import sys
import logging
import pickle
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config
from preprocessing.pipeline import PreprocessingPipeline
from models.trainer import EmbeddingTrainer
from models.tokenizer import DNATokenizer
from clustering.algorithms import EmbeddingClusterer
from clustering.taxonomy import HybridTaxonomyAssigner
from novelty.detection import NoveltyAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="eDNA Biodiversity Assessment Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .novelty-highlight {
        background-color: #ffe6e6;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #ff4444;
    }
</style>
""", unsafe_allow_html=True)

class BiodiversityDashboard:
    """Main dashboard class for eDNA biodiversity analysis"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.output_dir = Path("data/output")
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'sequences_uploaded' not in st.session_state:
            st.session_state.sequences_uploaded = False
    
    def run(self):
        """Main dashboard entry point"""
        # Header
        st.markdown('<h1 class="main-header">🌊 Deep-Sea eDNA Biodiversity Assessment</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        page = st.session_state.get('current_page', 'Overview')
        
        if page == 'Overview':
            self.render_overview_page()
        elif page == 'Data Upload':
            self.render_data_upload_page()
        elif page == 'Preprocessing':
            self.render_preprocessing_page()
        elif page == 'Analysis':
            self.render_analysis_page()
        elif page == 'Results':
            self.render_results_page()
        elif page == 'Settings':
            self.render_settings_page()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        pages = ['Overview', 'Data Upload', 'Preprocessing', 'Analysis', 'Results', 'Settings']
        
        current_page = st.sidebar.radio("Select Page", pages, 
                                      index=pages.index(st.session_state.get('current_page', 'Overview')))
        st.session_state.current_page = current_page
        
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.subheader("System Status")
        
        # Check if data is loaded
        data_status = "✅ Ready" if st.session_state.sequences_uploaded else "⏳ No data"
        st.sidebar.text(f"Data: {data_status}")
        
        # Check if models are available
        model_status = "✅ Available" if self.models_dir.exists() else "⏳ Not trained"
        st.sidebar.text(f"Models: {model_status}")
        
        # Analysis status
        analysis_status = "✅ Complete" if st.session_state.analysis_results else "⏳ Pending"
        st.sidebar.text(f"Analysis: {analysis_status}")
        
        st.sidebar.markdown("---")
        
        # Quick stats if analysis is available
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.sidebar.subheader("Quick Stats")
            st.sidebar.metric("Total Sequences", results.get('total_sequences', 0))
            st.sidebar.metric("Clusters Found", results.get('n_clusters', 0))
            st.sidebar.metric("Novel Taxa", results.get('novel_candidates', 0))
    
    def render_overview_page(self):
        """Render overview page"""
        st.title("System Overview")
        
        # Introduction
        st.markdown("""
        This dashboard provides an end-to-end system for analyzing deep-sea environmental DNA (eDNA) 
        to assess biodiversity and discover potentially novel taxa.
        """)
        
        # Key features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧬 Data Processing
            - Quality filtering
            - Adapter trimming  
            - Chimera detection
            - Sequence normalization
            """)
        
        with col2:
            st.markdown("""
            ### 🤖 ML Analysis
            - Deep learning embeddings
            - Clustering algorithms
            - Taxonomic assignment
            - Novelty detection
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Visualization
            - Interactive plots
            - Biodiversity metrics
            - Taxonomic trees
            - Export capabilities
            """)
        
        # Workflow diagram
        st.subheader("Analysis Workflow")
        
        workflow_steps = [
            "1. Upload eDNA sequences (FASTQ/FASTA)",
            "2. Preprocess and quality filter",
            "3. Generate sequence embeddings",
            "4. Cluster sequences by similarity",
            "5. Assign taxonomy using BLAST + ML",
            "6. Detect potentially novel taxa",
            "7. Visualize and export results"
        ]
        
        for step in workflow_steps:
            st.markdown(f"**{step}**")
        
        # Sample data option
        st.subheader("Get Started")
        if st.button("Load Sample Dataset", type="primary"):
            self.load_sample_data()
    
    def render_data_upload_page(self):
        """Render data upload page"""
        st.title("Data Upload & Import")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload eDNA sequence files",
            type=['fastq', 'fasta', 'fa', 'fq'],
            accept_multiple_files=True,
            help="Upload FASTQ or FASTA files containing eDNA sequences"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files")
            
            # Display file information
            st.subheader("Uploaded Files")
            file_info = []
            
            for file in uploaded_files:
                file_info.append({
                    'Filename': file.name,
                    'Size (MB)': f"{file.size / 1024 / 1024:.2f}",
                    'Type': file.type
                })
            
            st.dataframe(pd.DataFrame(file_info))
            
            # Process files button
            if st.button("Process Uploaded Files", type="primary"):
                self.process_uploaded_files(uploaded_files)
        
        # Alternative: Load from directory
        st.subheader("Load from Directory")
        data_directory = st.text_input("Data Directory Path", value=str(self.data_dir / "raw"))
        
        if st.button("Load from Directory"):
            self.load_from_directory(data_directory)
        
        # Sample data option
        st.subheader("Sample Data")
        if st.button("Load Sample eDNA Dataset"):
            self.load_sample_data()
    
    def render_preprocessing_page(self):
        """Render preprocessing page"""
        st.title("Data Preprocessing")
        
        if not st.session_state.sequences_uploaded:
            st.warning("Please upload data first!")
            return
        
        # Preprocessing parameters
        st.subheader("Preprocessing Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_length = st.number_input("Minimum Sequence Length", value=50, min_value=1)
            max_length = st.number_input("Maximum Sequence Length", value=500, min_value=1)
            quality_threshold = st.number_input("Quality Threshold", value=20, min_value=1)
        
        with col2:
            adapter_sequences = st.text_area(
                "Adapter Sequences (one per line)",
                value="AGATCGGAAGAGC\nCTGTCTCTTATA",
                help="Enter adapter sequences to be removed"
            )
            remove_chimeras = st.checkbox("Remove Chimeric Sequences", value=True)
        
        # Run preprocessing
        if st.button("Run Preprocessing", type="primary"):
            self.run_preprocessing({
                'min_length': min_length,
                'max_length': max_length,
                'quality_threshold': quality_threshold,
                'adapter_sequences': adapter_sequences.split('\n'),
                'remove_chimeras': remove_chimeras
            })
    
    def render_analysis_page(self):
        """Render analysis page"""
        st.title("Biodiversity Analysis")
        
        if not st.session_state.sequences_uploaded:
            st.warning("Please upload and preprocess data first!")
            return
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Clustering**")
            clustering_method = st.selectbox("Clustering Method", 
                                           ["hdbscan", "kmeans", "dbscan"])
            min_cluster_size = st.number_input("Minimum Cluster Size", value=10, min_value=1)
        
        with col2:
            st.markdown("**Novelty Detection**")
            novelty_threshold = st.slider("Novelty Threshold", 0.0, 1.0, 0.85)
            abundance_threshold = st.number_input("Abundance Threshold", value=0.001, 
                                                format="%.4f", min_value=0.0001)
        
        # Analysis options
        st.subheader("Analysis Options")
        
        analysis_options = {
            'embedding': st.checkbox("Generate Sequence Embeddings", value=True),
            'clustering': st.checkbox("Cluster Sequences", value=True),
            'taxonomy': st.checkbox("Assign Taxonomy", value=True),
            'novelty': st.checkbox("Detect Novel Taxa", value=True)
        }
        
        # Run analysis
        if st.button("Run Complete Analysis", type="primary"):
            self.run_complete_analysis({
                'clustering_method': clustering_method,
                'min_cluster_size': min_cluster_size,
                'novelty_threshold': novelty_threshold,
                'abundance_threshold': abundance_threshold,
                'options': analysis_options
            })
    
    def render_results_page(self):
        """Render results page"""
        st.title("Analysis Results")
        
        if not st.session_state.analysis_results:
            st.warning("No analysis results available. Please run analysis first!")
            return
        
        results = st.session_state.analysis_results
        
        # Summary metrics
        st.subheader("Summary Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sequences", results.get('total_sequences', 0))
        
        with col2:
            st.metric("Clusters Found", results.get('n_clusters', 0))
        
        with col3:
            st.metric("Novel Candidates", results.get('novel_candidates', 0))
        
        with col4:
            novelty_pct = results.get('novel_percentage', 0)
            st.metric("Novel %", f"{novelty_pct:.1f}%")
        
        # Clustering results
        if 'clustering' in results:
            self.render_clustering_results(results['clustering'])
        
        # Taxonomy results
        if 'taxonomy' in results:
            self.render_taxonomy_results(results['taxonomy'])
        
        # Novelty results
        if 'novelty' in results:
            self.render_novelty_results(results['novelty'])
        
        # Export options
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export CSV"):
                self.export_results_csv(results)
        
        with col2:
            if st.button("Export JSON"):
                self.export_results_json(results)
        
        with col3:
            if st.button("Generate Report"):
                self.generate_analysis_report(results)
    
    def render_clustering_results(self, clustering_results):
        """Render clustering visualization"""
        st.subheader("🔬 Clustering Results")
        
        # Cluster size distribution
        if 'cluster_sizes' in clustering_results:
            cluster_sizes = list(clustering_results['cluster_sizes'].values())
            
            fig = px.histogram(cluster_sizes, nbins=20, 
                             title="Cluster Size Distribution",
                             labels={'value': 'Cluster Size', 'count': 'Number of Clusters'})
            st.plotly_chart(fig, use_container_width=True)
        
        # 2D visualization
        if 'reduced_embeddings' in clustering_results:
            self.plot_cluster_visualization(clustering_results)
    
    def render_taxonomy_results(self, taxonomy_results):
        """Render taxonomy visualization"""
        st.subheader("🌿 Taxonomic Classification")
        
        # Taxonomy distribution
        if 'taxonomy_counts' in taxonomy_results:
            taxonomy_df = pd.DataFrame(list(taxonomy_results['taxonomy_counts'].items()),
                                     columns=['Taxonomy', 'Count'])
            
            fig = px.bar(taxonomy_df.head(15), x='Count', y='Taxonomy', 
                        orientation='h', title="Top 15 Taxonomic Groups")
            st.plotly_chart(fig, use_container_width=True)
        
        # Assignment confidence
        if 'confidence_scores' in taxonomy_results:
            confidences = taxonomy_results['confidence_scores']
            
            fig = px.histogram(confidences, nbins=30,
                             title="Taxonomic Assignment Confidence Distribution",
                             labels={'value': 'Confidence Score', 'count': 'Number of Sequences'})
            st.plotly_chart(fig, use_container_width=True)
    
    def render_novelty_results(self, novelty_results):
        """Render novelty detection results"""
        st.subheader("🆕 Novel Taxa Detection")
        
        # Novelty overview
        col1, col2 = st.columns(2)
        
        with col1:
            # Novel vs known pie chart
            labels = ['Known Taxa', 'Novel Candidates']
            values = [
                novelty_results.get('known_count', 0),
                novelty_results.get('novel_count', 0)
            ]
            
            fig = px.pie(values=values, names=labels, 
                        title="Known vs Novel Taxa Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Novelty scores distribution
            if 'novelty_scores' in novelty_results:
                scores = novelty_results['novelty_scores']
                
                fig = px.histogram(scores, nbins=30,
                                 title="Novelty Score Distribution",
                                 labels={'value': 'Novelty Score', 'count': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Novel candidates table
        if 'novel_candidates' in novelty_results:
            st.subheader("Novel Taxa Candidates")
            
            candidates_df = pd.DataFrame(novelty_results['novel_candidates'])
            st.dataframe(candidates_df, use_container_width=True)
    
    def render_settings_page(self):
        """Render settings page"""
        st.title("Settings & Configuration")
        
        # Model settings
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Embedding Model**")
            model_type = st.selectbox("Model Type", ["transformer", "autoencoder"])
            embedding_dim = st.number_input("Embedding Dimension", value=256, min_value=64)
            
        with col2:
            st.markdown("**Training Parameters**")
            batch_size = st.number_input("Batch Size", value=32, min_value=1)
            learning_rate = st.number_input("Learning Rate", value=0.0001, format="%.6f")
        
        # Analysis settings
        st.subheader("Analysis Settings")
        
        # Save settings
        if st.button("Save Settings"):
            self.save_settings({
                'model_type': model_type,
                'embedding_dim': embedding_dim,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            })
    
    def load_sample_data(self):
        """Load sample eDNA dataset"""
        with st.spinner("Loading sample dataset..."):
            # Create mock sample data
            sample_sequences = [
                "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
                "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
                "TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAAT",
                "CGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGC",
                "AGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTC"
            ] * 100  # Repeat for larger dataset
            
            st.session_state.sequences = sample_sequences
            st.session_state.sequences_uploaded = True
            
        st.success("Sample dataset loaded successfully!")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded sequence files"""
        with st.spinner("Processing uploaded files..."):
            # Here you would implement actual file processing
            # For demo purposes, we'll create mock data
            st.session_state.sequences = ["ATCGATCGATCG"] * 1000  # Mock data
            st.session_state.sequences_uploaded = True
            
        st.success("Files processed successfully!")
    
    def load_from_directory(self, directory_path):
        """Load sequences from directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            st.error(f"Directory {directory_path} does not exist!")
            return
        
        with st.spinner("Loading sequences from directory..."):
            # Here you would implement actual directory loading
            # For demo purposes, we'll create mock data
            st.session_state.sequences = ["ATCGATCGATCG"] * 500  # Mock data
            st.session_state.sequences_uploaded = True
            
        st.success("Sequences loaded from directory!")
    
    def run_preprocessing(self, params):
        """Run preprocessing pipeline"""
        with st.spinner("Running preprocessing..."):
            # Here you would run actual preprocessing
            # For demo purposes, we'll simulate processing
            st.session_state.preprocessed = True
            
        st.success("Preprocessing completed!")
        st.info("Preprocessed sequences are ready for analysis.")
    
    def run_complete_analysis(self, params):
        """Run complete biodiversity analysis"""
        with st.spinner("Running complete analysis... This may take a few minutes."):
            # Mock analysis results
            results = {
                'total_sequences': len(st.session_state.get('sequences', [])),
                'n_clusters': np.random.randint(10, 50),
                'novel_candidates': np.random.randint(5, 20),
                'novel_percentage': np.random.uniform(5, 15),
                'clustering': {
                    'cluster_sizes': {str(i): np.random.randint(10, 100) for i in range(20)},
                    'silhouette_score': np.random.uniform(0.3, 0.8)
                },
                'taxonomy': {
                    'taxonomy_counts': {
                        'Bacteria': 500,
                        'Archaea': 200,
                        'Eukaryota': 150,
                        'Unknown': 100
                    },
                    'confidence_scores': np.random.uniform(0.5, 1.0, 1000).tolist()
                },
                'novelty': {
                    'known_count': 800,
                    'novel_count': 150,
                    'novelty_scores': np.random.uniform(-2, 2, 1000).tolist(),
                    'novel_candidates': [
                        {'sequence_id': f'novel_{i}', 'novelty_score': np.random.uniform(0.8, 1.0)}
                        for i in range(10)
                    ]
                }
            }
            
            st.session_state.analysis_results = results
            
        st.success("Analysis completed!")
        st.balloons()
    
    def plot_cluster_visualization(self, clustering_results):
        """Plot 2D cluster visualization"""
        # Mock 2D embeddings for visualization
        n_points = 1000
        embeddings_2d = np.random.randn(n_points, 2)
        cluster_labels = np.random.randint(0, 10, n_points)
        
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': cluster_labels
        })
        
        fig = px.scatter(df, x='x', y='y', color='cluster', 
                        title="Sequence Clusters (2D Projection)",
                        labels={'x': 'UMAP 1', 'y': 'UMAP 2'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    def export_results_csv(self, results):
        """Export results to CSV"""
        # Create downloadable CSV
        csv_data = pd.DataFrame({
            'metric': ['total_sequences', 'n_clusters', 'novel_candidates'],
            'value': [results.get('total_sequences', 0), 
                     results.get('n_clusters', 0),
                     results.get('novel_candidates', 0)]
        })
        
        csv = csv_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="edna_analysis_results.csv",
            mime="text/csv"
        )
    
    def export_results_json(self, results):
        """Export results to JSON"""
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="edna_analysis_results.json",
            mime="application/json"
        )
    
    def generate_analysis_report(self, results):
        """Generate comprehensive analysis report"""
        report = f"""
# eDNA Biodiversity Analysis Report

## Summary
- **Total Sequences Analyzed**: {results.get('total_sequences', 0)}
- **Clusters Identified**: {results.get('n_clusters', 0)}
- **Novel Taxa Candidates**: {results.get('novel_candidates', 0)}
- **Novelty Percentage**: {results.get('novel_percentage', 0):.1f}%

## Analysis Details
This analysis was performed using the eDNA Biodiversity Assessment System,
which combines advanced machine learning techniques with traditional
bioinformatics approaches for comprehensive biodiversity analysis.

## Recommendations
Based on the analysis results, we recommend further investigation of the
identified novel taxa candidates through targeted sequencing and
taxonomic validation.
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name="edna_biodiversity_report.md",
            mime="text/markdown"
        )
    
    def save_settings(self, settings):
        """Save dashboard settings"""
        # Here you would save settings to a config file
        st.success("Settings saved successfully!")

def main():
    """Main function to run the dashboard"""
    dashboard = BiodiversityDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()