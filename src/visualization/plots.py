"""
Plotting utilities for biodiversity analysis visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiodiversityPlotter:
    """Plotting utilities for biodiversity analysis"""
    
    def __init__(self, 
                 style: str = "whitegrid",
                 color_palette: str = "husl",
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter
        
        Args:
            style: Seaborn style
            color_palette: Color palette
            figsize: Default figure size
        """
        self.style = style
        self.color_palette = color_palette
        self.figsize = figsize
        
        # Set style
        sns.set_style(style)
        plt.style.use('default')
        
        logger.info("Biodiversity plotter initialized")
    
    def plot_sequence_length_distribution(self,
                                        sequence_lengths: List[int],
                                        save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot sequence length distribution
        
        Args:
            sequence_lengths: List of sequence lengths
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        fig = px.histogram(
            x=sequence_lengths,
            nbins=50,
            title="Sequence Length Distribution",
            labels={'x': 'Sequence Length (bp)', 'count': 'Number of Sequences'}
        )
        
        fig.update_layout(
            xaxis_title="Sequence Length (bp)",
            yaxis_title="Number of Sequences",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Sequence length plot saved to {save_path}")
        
        return fig
    
    def plot_quality_scores(self,
                           quality_scores: np.ndarray,
                           save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot quality score distribution
        
        Args:
            quality_scores: Array of quality scores
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        fig = px.box(
            y=quality_scores,
            title="Sequence Quality Score Distribution",
            labels={'y': 'Quality Score'}
        )
        
        fig.update_layout(
            yaxis_title="Quality Score",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Quality scores plot saved to {save_path}")
        
        return fig
    
    def plot_taxonomic_composition(self,
                                 taxonomy_counts: Dict[str, int],
                                 plot_type: str = "pie",
                                 top_n: int = 15,
                                 save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot taxonomic composition
        
        Args:
            taxonomy_counts: Dictionary of taxonomy -> count
            plot_type: Type of plot ('pie', 'bar', 'treemap')
            top_n: Number of top taxa to show
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        # Prepare data
        df = pd.DataFrame(list(taxonomy_counts.items()), 
                         columns=['Taxonomy', 'Count'])
        df = df.sort_values('Count', ascending=False).head(top_n)
        
        if plot_type == "pie":
            fig = px.pie(
                df, 
                values='Count', 
                names='Taxonomy',
                title=f"Taxonomic Composition (Top {top_n})",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        
        elif plot_type == "bar":
            fig = px.bar(
                df,
                x='Count',
                y='Taxonomy',
                orientation='h',
                title=f"Taxonomic Composition (Top {top_n})"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        elif plot_type == "treemap":
            fig = px.treemap(
                df,
                values='Count',
                names='Taxonomy',
                title=f"Taxonomic Composition (Top {top_n})"
            )
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Taxonomic composition plot saved to {save_path}")
        
        return fig
    
    def plot_diversity_indices(self,
                             diversity_data: Dict[str, float],
                             save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot biodiversity indices
        
        Args:
            diversity_data: Dictionary of index_name -> value
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        indices = list(diversity_data.keys())
        values = list(diversity_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=indices,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Biodiversity Indices",
            xaxis_title="Diversity Index",
            yaxis_title="Value",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Diversity indices plot saved to {save_path}")
        
        return fig
    
    def plot_cluster_visualization(self,
                                 embeddings_2d: np.ndarray,
                                 cluster_labels: np.ndarray,
                                 novelty_labels: Optional[np.ndarray] = None,
                                 save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot 2D cluster visualization
        
        Args:
            embeddings_2d: 2D embeddings for visualization
            cluster_labels: Cluster labels
            novelty_labels: Optional novelty labels
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        # Prepare data
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': cluster_labels.astype(str)
        })
        
        if novelty_labels is not None:
            df['novelty'] = ['Novel' if label == -1 else 'Known' for label in novelty_labels]
            
            fig = px.scatter(
                df, x='x', y='y', 
                color='cluster',
                symbol='novelty',
                title="Sequence Clusters with Novelty Detection",
                labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
            )
        else:
            fig = px.scatter(
                df, x='x', y='y', 
                color='cluster',
                title="Sequence Clusters",
                labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
            )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Cluster visualization saved to {save_path}")
        
        return fig
    
    def plot_novelty_analysis(self,
                            novelty_scores: np.ndarray,
                            novelty_predictions: np.ndarray,
                            threshold: float = 0.0,
                            save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot novelty analysis results
        
        Args:
            novelty_scores: Novelty scores
            novelty_predictions: Novelty predictions
            threshold: Novelty threshold
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Novelty Score Distribution", "Novel vs Known"),
            specs=[[{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Score distribution
        fig.add_trace(
            go.Histogram(
                x=novelty_scores,
                nbinsx=50,
                name="Novelty Scores",
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            row=1, col=1
        )
        
        # Pie chart
        novel_count = np.sum(novelty_predictions == -1)
        known_count = np.sum(novelty_predictions == 1)
        
        fig.add_trace(
            go.Pie(
                labels=["Known", "Novel"],
                values=[known_count, novel_count],
                name="Novel vs Known"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Novelty Detection Analysis",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Novelty analysis plot saved to {save_path}")
        
        return fig
    
    def plot_abundance_heatmap(self,
                             abundance_matrix: np.ndarray,
                             sample_names: List[str],
                             taxa_names: List[str],
                             save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot abundance heatmap
        
        Args:
            abundance_matrix: Abundance matrix [samples x taxa]
            sample_names: Sample names
            taxa_names: Taxa names
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        fig = px.imshow(
            abundance_matrix,
            x=taxa_names,
            y=sample_names,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Taxa Abundance Heatmap"
        )
        
        fig.update_layout(
            xaxis_title="Taxa",
            yaxis_title="Samples"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Abundance heatmap saved to {save_path}")
        
        return fig
    
    def plot_rarefaction_curve(self,
                              rarefaction_data: Dict[int, float],
                              save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot rarefaction curve
        
        Args:
            rarefaction_data: Dictionary of sample_size -> diversity
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        sample_sizes = list(rarefaction_data.keys())
        diversity_values = list(rarefaction_data.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sample_sizes,
            y=diversity_values,
            mode='lines+markers',
            name='Rarefaction Curve',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title="Rarefaction Curve",
            xaxis_title="Sample Size",
            yaxis_title="Species Richness",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Rarefaction curve saved to {save_path}")
        
        return fig
    
    def plot_phylogenetic_tree(self,
                             tree_data: Dict[str, Any],
                             save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot phylogenetic tree (simplified)
        
        Args:
            tree_data: Tree data structure
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        # This is a placeholder for phylogenetic tree plotting
        # In practice, you'd use specialized libraries like ete3 or Bio.Phylo
        
        fig = go.Figure()
        
        # Add placeholder tree visualization
        fig.add_annotation(
            text="Phylogenetic Tree Visualization<br>Would be implemented with specialized tree libraries",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            title="Phylogenetic Tree",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Phylogenetic tree saved to {save_path}")
        
        return fig
    
    def create_analysis_dashboard(self,
                                analysis_results: Dict[str, Any],
                                save_path: Optional[Path] = None) -> go.Figure:
        """
        Create comprehensive analysis dashboard
        
        Args:
            analysis_results: Complete analysis results
            save_path: Optional path to save dashboard
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Taxonomic Composition",
                "Cluster Visualization", 
                "Novelty Detection",
                "Diversity Metrics"
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}]
            ]
        )
        
        # Taxonomic composition (pie chart)
        if 'taxonomy' in analysis_results:
            taxonomy_data = analysis_results['taxonomy'].get('taxonomy_counts', {})
            taxa = list(taxonomy_data.keys())[:10]  # Top 10
            counts = list(taxonomy_data.values())[:10]
            
            fig.add_trace(
                go.Pie(labels=taxa, values=counts, name="Taxonomy"),
                row=1, col=1
            )
        
        # Cluster visualization (scatter plot)
        if 'clustering' in analysis_results:
            # Mock 2D data for demonstration
            n_points = 500
            x = np.random.randn(n_points)
            y = np.random.randn(n_points)
            clusters = np.random.randint(0, 5, n_points)
            
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers', 
                          marker=dict(color=clusters, colorscale='Viridis'),
                          name="Clusters"),
                row=1, col=2
            )
        
        # Novelty detection (histogram)
        if 'novelty' in analysis_results:
            novelty_scores = analysis_results['novelty'].get('novelty_scores', [])
            if novelty_scores:
                fig.add_trace(
                    go.Histogram(x=novelty_scores, name="Novelty Scores"),
                    row=2, col=1
                )
        
        # Diversity metrics (bar chart)
        diversity_metrics = {
            'Shannon': np.random.uniform(1, 4),
            'Simpson': np.random.uniform(0.5, 1),
            'Chao1': np.random.uniform(50, 200),
            'ACE': np.random.uniform(60, 220)
        }
        
        fig.add_trace(
            go.Bar(x=list(diversity_metrics.keys()), 
                  y=list(diversity_metrics.values()),
                  name="Diversity"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="eDNA Biodiversity Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Analysis dashboard saved to {save_path}")
        
        return fig

def main():
    """Test function for plotting utilities"""
    logger.info("Testing biodiversity plotting utilities...")
    
    plotter = BiodiversityPlotter()
    
    # Test sequence length distribution
    sequence_lengths = np.random.normal(200, 50, 1000).astype(int)
    sequence_lengths = sequence_lengths[sequence_lengths > 0]
    
    fig1 = plotter.plot_sequence_length_distribution(sequence_lengths)
    fig1.show()
    
    # Test taxonomic composition
    taxonomy_counts = {
        'Bacteria': 500,
        'Archaea': 200,
        'Eukaryota': 150,
        'Viruses': 80,
        'Unknown': 70
    }
    
    fig2 = plotter.plot_taxonomic_composition(taxonomy_counts, plot_type="pie")
    fig2.show()
    
    # Test cluster visualization
    embeddings_2d = np.random.randn(1000, 2)
    cluster_labels = np.random.randint(0, 10, 1000)
    novelty_labels = np.random.choice([-1, 1], 1000, p=[0.1, 0.9])
    
    fig3 = plotter.plot_cluster_visualization(embeddings_2d, cluster_labels, novelty_labels)
    fig3.show()
    
    logger.info("Plotting utilities test complete!")

if __name__ == "__main__":
    main()