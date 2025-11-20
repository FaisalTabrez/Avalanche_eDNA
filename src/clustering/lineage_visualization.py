"""
Enhanced Taxonomic Lineage Visualization

This module provides comprehensive visualization capabilities for taxonomic lineage data:
- Interactive hierarchical tree visualizations
- Collapsible taxonomic trees with filtering
- Export capabilities (JSON-LD, RDF, CSV, etc.)
- Confidence and evidence visualization
- Multi-source comparison views
- Publication-ready plots and reports
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import base64
from io import StringIO, BytesIO

# Visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - some visualizations will be limited")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Import enhanced lineage components
from .enhanced_lineage import (
    EnhancedLineage, TaxonomicRank, EvidenceType, LineageEvidence,
    TaxonomicName, ExternalReference
)

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization components"""
    color_scheme: str = "viridis"
    width: int = 1200
    height: int = 800
    font_size: int = 12
    show_confidence: bool = True
    show_evidence: bool = True
    interactive: bool = True
    export_format: str = "html"

class TaxonomicTreeVisualizer:
    """Interactive taxonomic tree visualization"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Color schemes for different evidence types
        self.evidence_colors = {
            EvidenceType.NCBI_TAXDUMP: "#1f77b4",
            EvidenceType.SILVA: "#ff7f0e", 
            EvidenceType.PR2: "#2ca02c",
            EvidenceType.UNITE: "#d62728",
            EvidenceType.GTDB: "#9467bd",
            EvidenceType.BLAST_HIT: "#8c564b",
            EvidenceType.ML_PREDICTION: "#e377c2",
            EvidenceType.KNN_CONSENSUS: "#7f7f7f",
            EvidenceType.MANUAL_CURATION: "#bcbd22",
            EvidenceType.CUSTOM_DATABASE: "#17becf"
        }
        
        # Rank colors and priorities
        self.rank_colors = {
            TaxonomicRank.KINGDOM: "#1f77b4",
            TaxonomicRank.PHYLUM: "#ff7f0e",
            TaxonomicRank.CLASS: "#2ca02c",
            TaxonomicRank.ORDER: "#d62728",
            TaxonomicRank.FAMILY: "#9467bd",
            TaxonomicRank.GENUS: "#8c564b",
            TaxonomicRank.SPECIES: "#e377c2"
        }
    
    def create_hierarchical_tree(self, lineages: List[EnhancedLineage]) -> Optional[go.Figure]:
        """Create interactive hierarchical tree visualization"""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for tree visualization")
            return None
        
        # Build tree structure
        tree_data = self._build_tree_structure(lineages)
        if not tree_data:
            return None
        
        # Create Plotly tree
        fig = self._create_plotly_tree(tree_data)
        
        return fig
    
    def _build_tree_structure(self, lineages: List[EnhancedLineage]) -> Optional[Dict[str, Any]]:
        """Build hierarchical tree structure from lineages"""
        if not lineages:
            return None
        
        # Build nested dictionary structure
        tree = {}
        
        for lineage in lineages:
            current = tree
            path = []
            
            # Traverse from kingdom to species
            for rank in TaxonomicRank.basic_ranks():
                rank_name = lineage.lineage.get(rank)
                if rank_name:
                    path.append(rank_name)
                    
                    if rank_name not in current:
                        current[rank_name] = {
                            'name': rank_name,
                            'rank': rank,
                            'children': {},
                            'lineages': [],
                            'path': path.copy(),
                            'confidence': 0.0,
                            'evidence_types': set(),
                            'external_refs': []
                        }
                    
                    current[rank_name]['lineages'].append(lineage)
                    current[rank_name]['confidence'] = max(
                        current[rank_name]['confidence'], 
                        lineage.overall_confidence
                    )
                    
                    # Collect evidence types
                    for evidence in lineage.evidence:
                        current[rank_name]['evidence_types'].add(evidence.source)
                    
                    # Collect external references
                    current[rank_name]['external_refs'].extend(lineage.external_refs)
                    
                    current = current[rank_name]['children']
        
        return tree
    
    def _create_plotly_tree(self, tree_data: Dict[str, Any]) -> go.Figure:
        """Create Plotly sunburst or treemap visualization"""
        
        # Prepare data for sunburst chart
        labels = []
        parents = []
        values = []
        colors = []
        hover_texts = []
        
        def traverse_tree(node_dict: Dict[str, Any], parent: str = ""):
            for name, node in node_dict.items():
                labels.append(name)
                parents.append(parent)
                values.append(len(node['lineages']))
                
                # Color by rank or confidence
                if self.config.show_confidence:
                    colors.append(node['confidence'])
                else:
                    rank_color = self.rank_colors.get(node['rank'], "#cccccc")
                    colors.append(rank_color)
                
                # Create hover text
                hover_info = [
                    f"<b>{name}</b>",
                    f"Rank: {node['rank'].rank_name if 'rank' in node else 'Unknown'}",
                    f"Taxa: {len(node['lineages'])}",
                    f"Confidence: {node['confidence']:.3f}"
                ]
                
                if self.config.show_evidence:
                    evidence_list = [ev.value for ev in node['evidence_types']]
                    hover_info.append(f"Evidence: {', '.join(evidence_list)}")
                
                hover_texts.append("<br>".join(hover_info))
                
                # Recursively process children
                if node['children']:
                    traverse_tree(node['children'], name)
        
        traverse_tree(tree_data)
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                colorscale="Viridis" if self.config.show_confidence else None,
                colorbar=dict(title="Confidence") if self.config.show_confidence else None,
                line=dict(width=2, color="white")
            ) if self.config.show_confidence else dict(
                line=dict(width=2, color="white")
            )
        ))
        
        fig.update_layout(
            title=dict(
                text="Taxonomic Hierarchy",
                x=0.5,
                font=dict(size=16)
            ),
            width=self.config.width,
            height=self.config.height,
            font=dict(size=self.config.font_size)
        )
        
        return fig
    
    def create_confidence_heatmap(self, lineages: List[EnhancedLineage]) -> Optional[go.Figure]:
        """Create confidence heatmap across taxonomic ranks"""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Prepare data matrix
        rank_names = [rank.rank_name for rank in TaxonomicRank.basic_ranks()]
        confidence_matrix = []
        taxa_names = []
        
        for lineage in lineages:
            if not lineage.scientific_name:
                continue
                
            taxa_names.append(lineage.scientific_name[:30])  # Truncate long names
            row = []
            
            for rank in TaxonomicRank.basic_ranks():
                if rank in lineage.lineage and lineage.lineage[rank]:
                    # Use overall confidence for now - could be rank-specific
                    row.append(lineage.overall_confidence)
                else:
                    row.append(0.0)
            
            confidence_matrix.append(row)
        
        if not confidence_matrix:
            return None
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confidence_matrix,
            x=rank_names,
            y=taxa_names,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Confidence Score"),
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>%{x}<br>Confidence: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Taxonomic Assignment Confidence",
            xaxis_title="Taxonomic Rank",
            yaxis_title="Taxa",
            width=self.config.width,
            height=max(400, len(taxa_names) * 20),
            font=dict(size=self.config.font_size)
        )
        
        return fig
    
    def create_evidence_summary(self, lineages: List[EnhancedLineage]) -> Optional[go.Figure]:
        """Create evidence source summary visualization"""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Count evidence types
        evidence_counts = {}
        for lineage in lineages:
            for evidence in lineage.evidence:
                source_name = evidence.source.value
                if source_name not in evidence_counts:
                    evidence_counts[source_name] = 0
                evidence_counts[source_name] += 1
        
        if not evidence_counts:
            return None
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(evidence_counts.keys()),
            values=list(evidence_counts.values()),
            hole=0.4,
            marker_colors=[self.evidence_colors.get(EvidenceType(k), "#cccccc") 
                          for k in evidence_counts.keys()],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Evidence Sources Distribution",
            width=self.config.width // 2,
            height=self.config.height // 2,
            font=dict(size=self.config.font_size),
            showlegend=True
        )
        
        return fig

class LineageExporter:
    """Export lineage data in various formats"""
    
    def __init__(self):
        self.supported_formats = ['json', 'csv', 'tsv', 'xml', 'json-ld', 'rdf', 'phyloxml']
    
    def export_lineages(self, lineages: List[EnhancedLineage], 
                       output_path: str, 
                       format_type: str = 'json') -> bool:
        """Export lineages in specified format"""
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == 'json':
                return self._export_json(lineages, output_path)
            elif format_type.lower() == 'csv':
                return self._export_csv(lineages, output_path)
            elif format_type.lower() == 'tsv':
                return self._export_tsv(lineages, output_path)
            elif format_type.lower() == 'xml':
                return self._export_xml(lineages, output_path)
            elif format_type.lower() == 'json-ld':
                return self._export_json_ld(lineages, output_path)
            elif format_type.lower() == 'rdf':
                return self._export_rdf(lineages, output_path)
            elif format_type.lower() == 'phyloxml':
                return self._export_phyloxml(lineages, output_path)
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return False
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _export_json(self, lineages: List[EnhancedLineage], output_path: Path) -> bool:
        """Export as JSON"""
        data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_lineages': len(lineages),
                'format': 'enhanced_lineage_json'
            },
            'lineages': [lineage.to_dict() for lineage in lineages]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(lineages)} lineages to JSON: {output_path}")
        return True
    
    def _export_csv(self, lineages: List[EnhancedLineage], output_path: Path) -> bool:
        """Export as CSV"""
        rows = []
        
        for lineage in lineages:
            basic_lineage = lineage.get_basic_lineage()
            
            row = {
                'taxid': lineage.taxid,
                'scientific_name': lineage.scientific_name,
                'overall_confidence': lineage.overall_confidence,
                **basic_lineage,
                'common_names': '; '.join(lineage.names.common_names),
                'synonyms': '; '.join(lineage.names.synonyms),
                'authority': lineage.names.authority,
                'evidence_sources': '; '.join([ev.source.value for ev in lineage.evidence]),
                'external_refs': '; '.join([f"{ref.database}:{ref.identifier}" for ref in lineage.external_refs]),
                'last_updated': lineage.last_updated.isoformat() if lineage.last_updated else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported {len(lineages)} lineages to CSV: {output_path}")
        return True
    
    def _export_tsv(self, lineages: List[EnhancedLineage], output_path: Path) -> bool:
        """Export as TSV"""
        # Reuse CSV logic with tab separator
        rows = []
        
        for lineage in lineages:
            basic_lineage = lineage.get_basic_lineage()
            
            row = {
                'taxid': lineage.taxid,
                'scientific_name': lineage.scientific_name,
                'overall_confidence': lineage.overall_confidence,
                **basic_lineage,
                'common_names': '; '.join(lineage.names.common_names),
                'synonyms': '; '.join(lineage.names.synonyms),
                'authority': lineage.names.authority,
                'evidence_sources': '; '.join([ev.source.value for ev in lineage.evidence]),
                'external_refs': '; '.join([f"{ref.database}:{ref.identifier}" for ref in lineage.external_refs]),
                'last_updated': lineage.last_updated.isoformat() if lineage.last_updated else None
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')
        
        logger.info(f"Exported {len(lineages)} lineages to TSV: {output_path}")
        return True
    
    def _export_xml(self, lineages: List[EnhancedLineage], output_path: Path) -> bool:
        """Export as XML"""
        xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_content.append('<taxonomic_lineages>')
        xml_content.append(f'  <metadata>')
        xml_content.append(f'    <export_date>{datetime.now().isoformat()}</export_date>')
        xml_content.append(f'    <total_lineages>{len(lineages)}</total_lineages>')
        xml_content.append(f'  </metadata>')
        
        for lineage in lineages:
            xml_content.append('  <lineage>')
            
            if lineage.taxid:
                xml_content.append(f'    <taxid>{lineage.taxid}</taxid>')
            if lineage.scientific_name:
                xml_content.append(f'    <scientific_name>{self._escape_xml(lineage.scientific_name)}</scientific_name>')
            
            xml_content.append(f'    <confidence>{lineage.overall_confidence}</confidence>')
            
            # Taxonomic ranks
            xml_content.append('    <taxonomy>')
            for rank, name in lineage.lineage.items():
                if name:
                    xml_content.append(f'      <{rank.rank_name}>{self._escape_xml(name)}</{rank.rank_name}>')
            xml_content.append('    </taxonomy>')
            
            # Names
            if lineage.names.common_names:
                xml_content.append('    <common_names>')
                for name in lineage.names.common_names:
                    xml_content.append(f'      <name>{self._escape_xml(name)}</name>')
                xml_content.append('    </common_names>')
            
            # Evidence
            if lineage.evidence:
                xml_content.append('    <evidence>')
                for ev in lineage.evidence:
                    xml_content.append(f'      <source type="{ev.source.value}" confidence="{ev.confidence}">')
                    if ev.method_details:
                        xml_content.append(f'        <method>{self._escape_xml(ev.method_details)}</method>')
                    xml_content.append('      </source>')
                xml_content.append('    </evidence>')
            
            xml_content.append('  </lineage>')
        
        xml_content.append('</taxonomic_lineages>')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(xml_content))
        
        logger.info(f"Exported {len(lineages)} lineages to XML: {output_path}")
        return True
    
    def _export_json_ld(self, lineages: List[EnhancedLineage], output_path: Path) -> bool:
        """Export as JSON-LD (Linked Data)"""
        
        context = {
            "@context": {
                "@vocab": "http://example.org/taxonomy/",
                "dwc": "http://rs.tdwg.org/dwc/terms/",
                "taxon": "dwc:Taxon",
                "scientificName": "dwc:scientificName",
                "kingdom": "dwc:kingdom",
                "phylum": "dwc:phylum",
                "class": "dwc:class", 
                "order": "dwc:order",
                "family": "dwc:family",
                "genus": "dwc:genus",
                "species": "dwc:species"
            }
        }
        
        taxa = []
        for lineage in lineages:
            basic_lineage = lineage.get_basic_lineage()
            
            taxon = {
                "@type": "taxon",
                "@id": f"urn:taxon:{lineage.taxid}" if lineage.taxid else f"urn:taxon:{hash(lineage.scientific_name or 'unknown')}",
                "scientificName": lineage.scientific_name,
                "confidence": lineage.overall_confidence,
                **{k: v for k, v in basic_lineage.items() if v is not None},
                "commonNames": lineage.names.common_names,
                "synonyms": lineage.names.synonyms,
                "externalReferences": [
                    {
                        "database": ref.database,
                        "identifier": ref.identifier,
                        "url": ref.generate_url()
                    }
                    for ref in lineage.external_refs
                ]
            }
            
            taxa.append(taxon)
        
        data = {
            **context,
            "@type": "TaxonomicDataset",
            "dateCreated": datetime.now().isoformat(),
            "totalRecords": len(lineages),
            "taxa": taxa
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(lineages)} lineages to JSON-LD: {output_path}")
        return True
    
    def _export_rdf(self, lineages: List[EnhancedLineage], output_path: Path) -> bool:
        """Export as RDF/Turtle"""
        rdf_content = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix dwc: <http://rs.tdwg.org/dwc/terms/> .",
            "@prefix tax: <http://example.org/taxonomy/> .",
            ""
        ]
        
        for i, lineage in enumerate(lineages):
            taxon_id = f"tax:taxon{i+1}"
            rdf_content.append(f"{taxon_id} rdf:type dwc:Taxon ;")
            
            if lineage.scientific_name:
                rdf_content.append(f'    dwc:scientificName "{lineage.scientific_name}" ;')
            
            basic_lineage = lineage.get_basic_lineage()
            for rank, name in basic_lineage.items():
                if name:
                    rdf_content.append(f'    dwc:{rank} "{name}" ;')
            
            rdf_content.append(f'    tax:confidence {lineage.overall_confidence} ;')
            
            # Remove trailing semicolon from last property
            if rdf_content[-1].endswith(' ;'):
                rdf_content[-1] = rdf_content[-1][:-2] + ' .'
            
            rdf_content.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rdf_content))
        
        logger.info(f"Exported {len(lineages)} lineages to RDF: {output_path}")
        return True
    
    def _export_phyloxml(self, lineages: List[EnhancedLineage], output_path: Path) -> bool:
        """Export as PhyloXML format for phylogenetic analysis"""
        xml_content = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<phyloxml xmlns="http://www.phyloxml.org" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.phyloxml.org http://www.phyloxml.org/1.10/phyloxml.xsd">',
            '  <phylogeny rooted="true" rerootable="false">'
        ]
        
        # Create hierarchical structure for phylogenetic representation
        # This is a simplified version - full implementation would require proper tree construction
        for i, lineage in enumerate(lineages):
            xml_content.append(f'    <clade id_source="taxonomy">')
            xml_content.append(f'      <name>{self._escape_xml(lineage.scientific_name or f"taxon_{i}")}</name>')
            
            if lineage.taxid:
                xml_content.append(f'      <taxonomy>')
                xml_content.append(f'        <id provider="ncbi">{lineage.taxid}</id>')
                if lineage.scientific_name:
                    xml_content.append(f'        <scientific_name>{self._escape_xml(lineage.scientific_name)}</scientific_name>')
                xml_content.append(f'      </taxonomy>')
            
            xml_content.append(f'      <confidence type="bootstrap">{lineage.overall_confidence * 100:.1f}</confidence>')
            xml_content.append(f'    </clade>')
        
        xml_content.extend([
            '  </phylogeny>',
            '</phyloxml>'
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(xml_content))
        
        logger.info(f"Exported {len(lineages)} lineages to PhyloXML: {output_path}")
        return True
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        if not text:
            return ""
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&apos;"))

class LineageReportGenerator:
    """Generate comprehensive reports on taxonomic lineages"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.visualizer = TaxonomicTreeVisualizer(config)
        self.exporter = LineageExporter()
    
    def generate_comprehensive_report(self, 
                                    lineages: List[EnhancedLineage],
                                    output_dir: str,
                                    report_name: str = "lineage_report") -> str:
        """Generate comprehensive HTML report with all visualizations"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        tree_fig = self.visualizer.create_hierarchical_tree(lineages)
        heatmap_fig = self.visualizer.create_confidence_heatmap(lineages)
        evidence_fig = self.visualizer.create_evidence_summary(lineages)
        
        # Generate statistics
        stats = self._calculate_statistics(lineages)
        
        # Create HTML report
        html_content = self._create_html_report(
            lineages, stats, tree_fig, heatmap_fig, evidence_fig, report_name
        )
        
        # Save HTML report
        report_path = output_dir / f"{report_name}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Export data in multiple formats
        self.exporter.export_lineages(lineages, output_dir / f"{report_name}.json", "json")
        self.exporter.export_lineages(lineages, output_dir / f"{report_name}.csv", "csv")
        self.exporter.export_lineages(lineages, output_dir / f"{report_name}.xml", "xml")
        
        logger.info(f"Comprehensive report generated: {report_path}")
        return str(report_path)
    
    def _calculate_statistics(self, lineages: List[EnhancedLineage]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not lineages:
            return {}
        
        stats = {
            'total_lineages': len(lineages),
            'with_taxid': sum(1 for l in lineages if l.taxid),
            'avg_confidence': np.mean([l.overall_confidence for l in lineages]),
            'rank_coverage': {},
            'evidence_distribution': {},
            'external_refs_count': sum(len(l.external_refs) for l in lineages)
        }
        
        # Rank coverage
        for rank in TaxonomicRank.basic_ranks():
            count = sum(1 for l in lineages if rank in l.lineage and l.lineage[rank])
            stats['rank_coverage'][rank.rank_name] = {
                'count': count,
                'percentage': (count / len(lineages)) * 100
            }
        
        # Evidence distribution
        evidence_counts = {}
        for lineage in lineages:
            for evidence in lineage.evidence:
                source = evidence.source.value
                evidence_counts[source] = evidence_counts.get(source, 0) + 1
        
        stats['evidence_distribution'] = evidence_counts
        
        return stats
    
    def _create_html_report(self, 
                          lineages: List[EnhancedLineage],
                          stats: Dict[str, Any], 
                          tree_fig: Optional[go.Figure],
                          heatmap_fig: Optional[go.Figure],
                          evidence_fig: Optional[go.Figure],
                          report_name: str) -> str:
        """Create HTML report content"""
        
        html_parts = [
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Taxonomic Lineage Report</title>
                <meta charset="utf-8">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .section { margin: 30px 0; }
                    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                    .stat-card { background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }
                    .stat-number { font-size: 2em; font-weight: bold; color: #2c3e50; }
                    .stat-label { color: #7f8c8d; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
            """,
            
            f"""
            <div class="header">
                <h1>Taxonomic Lineage Report: {report_name}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """,
            
            # Summary statistics
            '<div class="section">',
            '<h2>Summary Statistics</h2>',
            '<div class="stat-grid">'
        ]
        
        # Add stat cards
        if stats:
            html_parts.extend([
                f'<div class="stat-card"><div class="stat-number">{stats["total_lineages"]}</div><div class="stat-label">Total Lineages</div></div>',
                f'<div class="stat-card"><div class="stat-number">{stats.get("with_taxid", 0)}</div><div class="stat-label">With Taxid</div></div>',
                f'<div class="stat-card"><div class="stat-number">{stats.get("avg_confidence", 0):.3f}</div><div class="stat-label">Avg Confidence</div></div>',
                f'<div class="stat-card"><div class="stat-number">{stats.get("external_refs_count", 0)}</div><div class="stat-label">External References</div></div>'
            ])
        
        html_parts.extend([
            '</div>',
            '</div>'
        ])
        
        # Visualizations
        if PLOTLY_AVAILABLE:
            html_parts.append('<div class="section">')
            html_parts.append('<h2>Visualizations</h2>')
            
            if tree_fig:
                html_parts.append('<h3>Hierarchical Tree</h3>')
                html_parts.append(f'<div id="tree-plot">{tree_fig.to_html(include_plotlyjs=False, div_id="tree-plot")}</div>')
            
            if heatmap_fig:
                html_parts.append('<h3>Confidence Heatmap</h3>')
                html_parts.append(f'<div id="heatmap-plot">{heatmap_fig.to_html(include_plotlyjs=False, div_id="heatmap-plot")}</div>')
            
            if evidence_fig:
                html_parts.append('<h3>Evidence Distribution</h3>')
                html_parts.append(f'<div id="evidence-plot">{evidence_fig.to_html(include_plotlyjs=False, div_id="evidence-plot")}</div>')
            
            html_parts.append('</div>')
        
        # Rank coverage table
        if stats and 'rank_coverage' in stats:
            html_parts.extend([
                '<div class="section">',
                '<h2>Rank Coverage</h2>',
                '<table>',
                '<tr><th>Rank</th><th>Count</th><th>Percentage</th></tr>'
            ])
            
            for rank, data in stats['rank_coverage'].items():
                html_parts.append(f'<tr><td>{rank.title()}</td><td>{data["count"]}</td><td>{data["percentage"]:.1f}%</td></tr>')
            
            html_parts.extend(['</table>', '</div>'])
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)

# Factory functions
def create_visualizer_from_config() -> TaxonomicTreeVisualizer:
    """Create visualizer from configuration"""
    viz_config = config.get('visualization', {})
    
    config_obj = VisualizationConfig(
        color_scheme=viz_config.get('color_scheme', 'viridis'),
        width=viz_config.get('width', 1200),
        height=viz_config.get('height', 800),
        font_size=viz_config.get('font_size', 12),
        show_confidence=viz_config.get('show_confidence', True),
        show_evidence=viz_config.get('show_evidence', True),
        interactive=viz_config.get('interactive', True)
    )
    
    return TaxonomicTreeVisualizer(config_obj)