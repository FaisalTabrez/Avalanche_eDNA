"""
Interactive Report Management Dashboard.

This module provides a comprehensive Streamlit-based dashboard for managing,
browsing, and analyzing eDNA analysis reports.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import os
import json

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.manager import DatabaseManager
from src.database.queries import ReportQueryEngine
from src.report_management.catalogue_manager import ReportCatalogueManager
from src.similarity.cross_analysis_engine import CrossAnalysisEngine
from src.analysis.dataset_analyzer import DatasetAnalyzer

# Page configuration
st.set_page_config(
    page_title="eDNA Report Management Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ReportManagementDashboard:
    """Main dashboard class for interactive report management."""
    
    def __init__(self):
        """Initialize dashboard with required managers."""
        self.db_manager = DatabaseManager()
        self.query_engine = ReportQueryEngine(self.db_manager)
        self.catalogue_manager = ReportCatalogueManager(db_manager=self.db_manager)
        self.cross_analysis_engine = CrossAnalysisEngine(self.db_manager)
        
        # Initialize session state
        if 'selected_reports' not in st.session_state:
            st.session_state.selected_reports = []
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Overview'
    
    def run(self):
        """Main dashboard entry point."""
        st.markdown('<h1 style="text-align: center; color: #1f77b4;">🧬 eDNA Report Management Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        self.render_sidebar()
        
        # Main content
        page = st.session_state.current_page
        
        if page == 'Overview':
            self.render_overview_page()
        elif page == 'Dataset Analyzer':
            self.render_dataset_analyzer_page()
        elif page == 'Report Browser':
            self.render_report_browser_page()
        elif page == 'Complete Report View':
            self.render_complete_report_view_page()
        elif page == 'Report Comparison':
            self.render_report_comparison_page()
        elif page == 'Organism Profiles':
            self.render_organism_profiles_page()
        elif page == 'Similarity Analysis':
            self.render_similarity_analysis_page()
    
    def render_sidebar(self):
        """Render sidebar with navigation and quick stats."""
        st.sidebar.title("Navigation")
        
        pages = ['Overview', 'Dataset Analyzer', 'Report Browser', 'Complete Report View', 'Report Comparison', 'Organism Profiles', 'Similarity Analysis']
        current_page = st.sidebar.radio("Select Page", pages, index=pages.index(st.session_state.current_page))
        st.session_state.current_page = current_page
        
        st.sidebar.markdown("---")
        
        # Quick statistics
        st.sidebar.subheader("Quick Statistics")
        try:
            stats = self.db_manager.get_database_statistics()
            st.sidebar.metric("Total Reports", stats.get('total_reports', 0))
            st.sidebar.metric("Total Organisms", stats.get('total_organisms', 0))
            st.sidebar.metric("Novel Candidates", stats.get('novel_organisms', 0))
        except Exception as e:
            st.sidebar.error(f"Failed to load statistics: {str(e)}")
    
    def render_overview_page(self):
        """Render overview page with system summary."""
        st.title("System Overview")
        
        # Get database statistics
        stats = self.db_manager.get_database_statistics()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Analysis Reports", stats.get('total_reports', 0))
        with col2:
            st.metric("Unique Organisms", stats.get('total_organisms', 0))
        with col3:
            st.metric("Novel Candidates", stats.get('novel_organisms', 0))
        with col4:
            st.metric("Datasets", stats.get('total_datasets', 0))
        
        # Recent activity
        st.subheader("Recent Analysis Reports")
        
        # Debug information (can be removed after testing)
        with st.expander("🔍 Debug Information", expanded=False):
            try:
                recent_reports = self.catalogue_manager.list_reports(limit=10)
                st.write(f"Number of reports found: {len(recent_reports)}")
                if recent_reports:
                    st.write("Sample report data:")
                    st.json(recent_reports[0])
                    st.write(f"All available keys: {list(recent_reports[0].keys())}")
            except Exception as e:
                st.error(f"Error in debug: {str(e)}")
        
        # Main report display
        recent_reports = self.catalogue_manager.list_reports(limit=10)
        
        if recent_reports:
            try:
                df = pd.DataFrame(recent_reports)
                
                # Convert created_at to datetime if it exists
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Show dataframe info
                st.write(f"📊 Displaying {len(df)} reports")
                
                # Display the dataframe with available columns
                display_columns = ['report_name', 'dataset_name', 'analysis_type', 'shannon_diversity', 'created_at']
                available_columns = [col for col in display_columns if col in df.columns]
                
                if available_columns:
                    st.dataframe(df[available_columns], use_container_width=True)
                else:
                    st.warning("Expected columns not found, showing all available data:")
                    st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error displaying reports: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                # Fallback: try to show raw data
                try:
                    st.subheader("Raw Data (Fallback)")
                    for i, report in enumerate(recent_reports[:3]):
                        st.write(f"Report {i+1}:")
                        st.json(report)
                except:
                    st.error("Could not display raw data either")
        else:
            st.info("No reports found. Upload and analyze some datasets to get started!")
    
    def render_report_browser_page(self):
        """Render report browser with filtering and search."""
        st.title("Report Browser")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now())
            )
        
        with col2:
            analysis_types = ["All", "comprehensive", "clustering", "taxonomy"]
            selected_analysis_type = st.selectbox("Analysis Type", analysis_types)
        
        # Get reports
        filter_kwargs = {}
        if len(date_range) == 2:
            filter_kwargs['date_range'] = (
                datetime.combine(date_range[0], datetime.min.time()),
                datetime.combine(date_range[1], datetime.max.time())
            )
        if selected_analysis_type != "All":
            filter_kwargs['analysis_type'] = selected_analysis_type
        
        reports = self.catalogue_manager.list_reports(limit=100, **filter_kwargs)
        
        # Debug information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write(f"Filter kwargs: {filter_kwargs}")
            st.write(f"Number of reports found: {len(reports)}")
            if reports:
                st.write("Sample report data:")
                st.json(reports[0])
        
        st.subheader(f"Found {len(reports)} reports")
        
        if reports:
            df = pd.DataFrame(reports)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Selection
            selected_indices = st.multiselect(
                "Select reports for comparison:",
                options=range(len(df)),
                format_func=lambda x: f"{df.iloc[x]['report_name']} ({df.iloc[x]['dataset_name']})"
            )
            
            st.session_state.selected_reports = [df.iloc[i]['report_id'] for i in selected_indices]
            
            # Display table
            st.dataframe(
                df[['report_name', 'dataset_name', 'analysis_type', 'created_at', 'shannon_diversity']],
                use_container_width=True
            )
            
            # Comparison button
            if len(st.session_state.selected_reports) >= 2:
                if st.button("Compare Selected Reports", type="primary"):
                    st.session_state.current_page = 'Report Comparison'
                    st.rerun()
        else:
            st.info("No reports found matching the criteria.")
    
    def render_report_comparison_page(self):
        """Render report comparison page."""
        st.title("Report Comparison")
        
        if len(st.session_state.selected_reports) < 2:
            st.warning("Please select at least 2 reports from the Report Browser page.")
            return
        
        selected_reports = st.session_state.selected_reports
        st.subheader(f"Comparing {len(selected_reports)} reports")
        
        # Debug information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write(f"Selected report IDs: {selected_reports}")
            
            # Check if reports exist in database
            for report_id in selected_reports:
                try:
                    report = self.db_manager.get_analysis_report(report_id)
                    if report:
                        st.write(f"✅ Report {report_id}: Found")
                        st.write(f"   - Name: {report.report_name}")
                        st.write(f"   - Shannon Diversity: {report.shannon_diversity}")
                    else:
                        st.write(f"❌ Report {report_id}: Not found")
                except Exception as e:
                    st.write(f"⚠️ Report {report_id}: Error checking - {str(e)}")
        
        # Pairwise comparisons
        comparison_count = 0
        for i in range(len(selected_reports)):
            for j in range(i + 1, len(selected_reports)):
                report_1 = selected_reports[i]
                report_2 = selected_reports[j]
                
                st.subheader(f"📊 Comparison {comparison_count + 1}: Report {i+1} vs Report {j+1}")
                
                try:
                    # Show comparison attempt
                    with st.spinner(f"Comparing {report_1} with {report_2}..."):
                        similarity_matrix = self.cross_analysis_engine.compare_reports(report_1, report_2)
                    
                    if similarity_matrix:
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Similarity", f"{similarity_matrix.similarity_score:.3f}")
                        with col2:
                            st.metric("Jaccard Similarity", f"{similarity_matrix.jaccard_similarity:.3f}")
                        with col3:
                            st.metric("Shared Organisms", similarity_matrix.organism_overlap_count)
                        with col4:
                            st.metric("Cosine Similarity", f"{similarity_matrix.cosine_similarity:.3f}")
                        
                        # Additional details
                        with st.expander("📋 Detailed Comparison Results", expanded=False):
                            details_col1, details_col2 = st.columns(2)
                            
                            with details_col1:
                                st.write("**Taxonomic Similarities:**")
                                st.write(f"Kingdom: {similarity_matrix.kingdom_similarity:.3f}")
                                st.write(f"Phylum: {similarity_matrix.phylum_similarity:.3f}")
                                st.write(f"Genus: {similarity_matrix.genus_similarity:.3f}")
                                
                                st.write("**Diversity Differences:**")
                                st.write(f"Shannon: {similarity_matrix.shannon_diversity_diff:.3f}")
                                st.write(f"Simpson: {similarity_matrix.simpson_diversity_diff:.3f}")
                            
                            with details_col2:
                                st.write("**Environmental Context:**")
                                if similarity_matrix.location_distance_km:
                                    st.write(f"Distance: {similarity_matrix.location_distance_km:.1f} km")
                                if similarity_matrix.depth_difference_m:
                                    st.write(f"Depth diff: {similarity_matrix.depth_difference_m:.1f} m")
                                if similarity_matrix.temporal_difference_days:
                                    st.write(f"Time diff: {similarity_matrix.temporal_difference_days:.0f} days")
                                
                                st.write("**Comparison Details:**")
                                st.write(f"Method: {similarity_matrix.comparison_method}")
                                st.write(f"Comparison ID: {similarity_matrix.comparison_id}")
                        
                        st.success(f"✅ Comparison {comparison_count + 1} completed successfully")
                        
                    else:
                        st.error(f"❌ Comparison {comparison_count + 1} failed: No similarity matrix returned")
                        st.info("This could be due to:")
                        st.write("- Missing organism data for one or both reports")
                        st.write("- Incomplete analysis results")
                        st.write("- Database connectivity issues")
                        
                except Exception as e:
                    st.error(f"❌ Comparison {comparison_count + 1} failed with error: {str(e)}")
                    
                    with st.expander("Show Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                
                comparison_count += 1
                st.markdown("---")
        
        if comparison_count == 0:
            st.info("No comparisons could be performed with the selected reports.")
    
    def render_complete_report_view_page(self):
        """Render comprehensive report view with all parameters and analysis details."""
        st.title("Complete Report View")
        st.markdown("📋 **Comprehensive Analysis Report with All Parameters**")
        
        # Report selection
        st.subheader("Select Report")
        
        # Get all reports
        reports = self.catalogue_manager.list_reports(limit=100)
        
        if not reports:
            st.warning("No reports found. Please analyze some datasets first.")
            return
        
        # Create report selection dropdown
        report_options = {}
        for report in reports:
            display_name = f"{report.get('report_name', 'Unknown')} - {report.get('dataset_name', 'Unknown')} ({report.get('created_at', 'Unknown date')})"
            report_options[display_name] = report.get('report_id')
        
        selected_report_display = st.selectbox(
            "Choose a report to view:",
            options=list(report_options.keys())
        )
        
        if not selected_report_display:
            return
        
        selected_report_id = report_options[selected_report_display]
        
        # Load complete report data
        try:
            with st.spinner("Loading complete report data..."):
                report_data = self.db_manager.get_analysis_report(selected_report_id)
                
                if not report_data:
                    st.error("Report not found in database.")
                    return
                
                # Get dataset information
                dataset_info = self._get_dataset_info(report_data.dataset_id)
                
                # Get additional data
                organisms = self._get_organisms_for_report(selected_report_id)
                similarities = self._get_similarities_for_report(selected_report_id)
                
        except Exception as e:
            st.error(f"Error loading report data: {str(e)}")
            return
        
        # Display comprehensive report
        self._display_complete_report(report_data, dataset_info, organisms, similarities)
    
    def _display_complete_report(self, report_data, dataset_info, organisms, similarities):
        """Display complete report with all sections and parameters."""
        
        # Report Header
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### 📊 {report_data.report_name or 'Unknown Report'}")
            dataset_name = dataset_info.dataset_name if dataset_info else f"Dataset ID: {report_data.dataset_id}"
            st.markdown(f"**Dataset:** {dataset_name}")
        with col2:
            st.metric("Report ID", report_data.report_id)
        with col3:
            st.metric("Analysis Type", report_data.analysis_type.title())
        
        # Basic Information Section
        st.markdown("---")
        st.subheader("📋 Basic Information")
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.metric("Created", report_data.created_at.strftime("%Y-%m-%d %H:%M") if report_data.created_at else "Unknown")
        with info_col2:
            total_sequences = dataset_info.total_sequences if dataset_info else getattr(report_data, 'total_sequences', None)
            st.metric("Total Sequences", f"{total_sequences:,}" if total_sequences else "N/A")
        with info_col3:
            st.metric("Mean Length", f"{report_data.mean_length:.1f}" if report_data.mean_length else "N/A")
        with info_col4:
            sequence_type = report_data.sequence_type_detected.value if report_data.sequence_type_detected else "Unknown"
            st.metric("Sequence Type", sequence_type)
        
        # Biodiversity Metrics Section
        st.markdown("---")
        st.subheader("🌿 Biodiversity Metrics")
        
        bio_col1, bio_col2, bio_col3, bio_col4 = st.columns(4)
        
        with bio_col1:
            st.metric("Shannon Diversity", f"{report_data.shannon_diversity:.4f}" if report_data.shannon_diversity else "N/A")
        with bio_col2:
            st.metric("Simpson Diversity", f"{report_data.simpson_diversity:.4f}" if report_data.simpson_diversity else "N/A")
        with bio_col3:
            st.metric("Species Richness", report_data.species_richness if report_data.species_richness else "N/A")
        with bio_col4:
            st.metric("Evenness", f"{report_data.evenness:.4f}" if report_data.evenness else "N/A")
        
        # Environmental Context Section
        st.markdown("---")
        st.subheader("🌍 Environmental Context")
        
        env_col1, env_col2, env_col3, env_col4 = st.columns(4)
        
        with env_col1:
            location = dataset_info.collection_location if dataset_info else "Unknown"
            st.metric("Location", location)
        with env_col2:
            depth = dataset_info.depth_meters if dataset_info else None
            st.metric("Depth (m)", f"{depth:.1f}" if depth else "N/A")
        with env_col3:
            temperature = dataset_info.temperature_celsius if dataset_info else None
            st.metric("Temperature (°C)", f"{temperature:.1f}" if temperature else "N/A")
        with env_col4:
            salinity = dataset_info.salinity if dataset_info else None
            st.metric("Salinity", f"{salinity:.2f}" if salinity else "N/A")
        
        # Additional Environmental Parameters
        ph = dataset_info.ph_level if dataset_info else None
        if ph or (dataset_info and any([dataset_info.ph_level])):
            env_col5, env_col6, env_col7, env_col8 = st.columns(4)
            
            with env_col5:
                st.metric("pH", f"{ph:.2f}" if ph else "N/A")
            with env_col6:
                st.metric("Dissolved O₂", "N/A")  # Not available in dataset_info
            with env_col7:
                st.metric("Conductivity", "N/A")  # Not available in dataset_info
            with env_col8:
                st.metric("Turbidity", "N/A")  # Not available in dataset_info
        
        # Organisms Section
        st.markdown("---")
        st.subheader(f"🦠 Detected Organisms ({len(organisms)} total)")
        
        if organisms:
            # Summary statistics
            novel_count = sum(1 for org in organisms if org.is_novel_candidate)
            kingdoms = set(org.kingdom for org in organisms if org.kingdom)
            phyla = set(org.phylum for org in organisms if org.phylum)
            
            org_col1, org_col2, org_col3, org_col4 = st.columns(4)
            
            with org_col1:
                st.metric("Total Organisms", len(organisms))
            with org_col2:
                st.metric("Novel Candidates", novel_count)
            with org_col3:
                st.metric("Kingdoms", len(kingdoms))
            with org_col4:
                st.metric("Phyla", len(phyla))
            
            # Organisms table
            with st.expander(f"📋 View All {len(organisms)} Organisms", expanded=False):
                organism_data = []
                for org in organisms:
                    organism_data.append({
                        'Organism Name': org.organism_name or 'Unknown',
                        'Kingdom': org.kingdom or 'Unknown',
                        'Phylum': org.phylum or 'Unknown',
                        'Genus': org.genus or 'Unknown',
                        'Detection Count': org.detection_count,
                        'Novel Candidate': 'Yes' if org.is_novel_candidate else 'No',
                        'Confidence Score': f"{org.confidence_score:.3f}" if org.confidence_score else 'N/A'
                    })
                
                df_organisms = pd.DataFrame(organism_data)
                st.dataframe(df_organisms, use_container_width=True)
            
            # Top organisms chart
            top_organisms = sorted(organisms, key=lambda x: x.detection_count or 0, reverse=True)[:10]
            if top_organisms:
                st.subheader("🔝 Top 10 Most Abundant Organisms")
                
                org_names = [org.organism_name or f'Unknown_{i+1}' for i, org in enumerate(top_organisms)]
                counts = [org.detection_count or 0 for org in top_organisms]
                
                fig = px.bar(
                    x=counts,
                    y=org_names,
                    orientation='h',
                    title="Organism Abundance",
                    labels={'x': 'Detection Count', 'y': 'Organism'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No organisms found for this report.")
        
        # Quality Metrics Section
        st.markdown("---")
        st.subheader("🔍 Quality Metrics")
        
        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
        
        with qual_col1:
            st.metric("Processing Time", f"{report_data.processing_time_seconds:.1f}s" if report_data.processing_time_seconds else "N/A")
        with qual_col2:
            file_size = dataset_info.file_size_mb if dataset_info else None
            st.metric("File Size", f"{file_size:.2f} MB" if file_size else "N/A")
        with qual_col3:
            st.metric("Quality Score", "N/A")  # Not available in current model
        with qual_col4:
            st.metric("Data Completeness", "N/A")  # Not available in current model
        
        # Analysis Parameters Section
        st.markdown("---")
        st.subheader("⚙️ Analysis Parameters")
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            st.markdown("**Sequence Analysis:**")
            st.write(f"• Minimum Length: {report_data.min_length or 'N/A'}")
            st.write(f"• Maximum Length: {report_data.max_length or 'N/A'}")
            st.write(f"• Standard Deviation: {report_data.std_length:.2f}" if report_data.std_length else "• Standard Deviation: N/A")
        
        with param_col2:
            st.markdown("**Taxonomic Analysis:**")
            st.write(f"• Analysis Method: {report_data.analysis_type or 'Unknown'}")
            st.write(f"• Confidence Threshold: N/A")  # Not available in current model
            st.write(f"• Database Version: N/A")  # Not available in current model
        
        # Similarity Analysis Section
        if similarities:
            st.markdown("---")
            st.subheader(f"🔗 Similarity Comparisons ({len(similarities)} total)")
            
            similarity_data = []
            for sim in similarities:
                similarity_data.append({
                    'Compared With': f"Report {sim.report_id_2}",
                    'Overall Similarity': f"{sim.similarity_score:.3f}",
                    'Jaccard Similarity': f"{sim.jaccard_similarity:.3f}",
                    'Shared Organisms': sim.organism_overlap_count,
                    'Comparison Date': sim.created_at.strftime("%Y-%m-%d") if sim.created_at else 'Unknown'
                })
            
            df_similarities = pd.DataFrame(similarity_data)
            st.dataframe(df_similarities, use_container_width=True)
        
        # Additional Metadata Section
        st.markdown("---")
        st.subheader("📊 Additional Metadata")
        
        with st.expander("🔧 Technical Details", expanded=False):
            metadata_col1, metadata_col2 = st.columns(2)
            
            with metadata_col1:
                st.markdown("**System Information:**")
                st.write(f"• Report ID: {report_data.report_id}")
                st.write(f"• Dataset ID: {report_data.dataset_id}")
                st.write(f"• Analysis Version: N/A")  # Not available in current model
            
            with metadata_col2:
                st.markdown("**Timestamps:**")
                st.write(f"• Created: {report_data.created_at}")
                st.write(f"• Last Updated: {report_data.updated_at if report_data.updated_at else 'N/A'}")
                st.write(f"• Analysis Duration: {report_data.processing_time_seconds:.1f}s" if report_data.processing_time_seconds else "• Analysis Duration: N/A")
        
        # Export Options
        st.markdown("---")
        st.subheader("📤 Export Options")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📋 Export as CSV", type="secondary"):
                # Prepare export data
                export_data = self._prepare_export_data(report_data, dataset_info, organisms)
                csv = export_data.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"report_{report_data.report_id}_complete.csv",
                    "text/csv"
                )
        
        with export_col2:
            if st.button("📊 Export Summary", type="secondary"):
                summary = self._generate_report_summary(report_data, dataset_info, organisms)
                st.download_button(
                    "Download Summary",
                    summary,
                    f"report_{report_data.report_id}_summary.txt",
                    "text/plain"
                )
        
        with export_col3:
            if st.button("🔄 Refresh Data", type="secondary"):
                st.rerun()
    
    def _prepare_export_data(self, report_data, dataset_info, organisms):
        """Prepare comprehensive data for CSV export."""
        export_rows = []
        
        # Add report metadata row
        dataset_name = dataset_info.dataset_name if dataset_info else "Unknown"
        location = dataset_info.collection_location if dataset_info else "Unknown"
        depth = dataset_info.depth_meters if dataset_info else None
        temperature = dataset_info.temperature_celsius if dataset_info else None
        
        export_rows.append({
            'Type': 'Report_Metadata',
            'Name': report_data.report_name,
            'Dataset': dataset_name,
            'Value': '',
            'Shannon_Diversity': report_data.shannon_diversity,
            'Simpson_Diversity': report_data.simpson_diversity,
            'Species_Richness': report_data.species_richness,
            'Location': location,
            'Depth_m': depth,
            'Temperature_C': temperature
        })
        
        # Add organism rows
        for org in organisms:
            export_rows.append({
                'Type': 'Organism',
                'Name': org.organism_name,
                'Dataset': dataset_name,
                'Value': org.detection_count,
                'Shannon_Diversity': '',
                'Simpson_Diversity': '',
                'Species_Richness': '',
                'Location': org.kingdom,
                'Depth_m': org.phylum,
                'Temperature_C': org.genus
            })
        
        return pd.DataFrame(export_rows)
    
    def _generate_report_summary(self, report_data, dataset_info, organisms):
        """Generate a text summary of the report."""
        dataset_name = dataset_info.dataset_name if dataset_info else f"Dataset ID: {report_data.dataset_id}"
        
        summary_lines = [
            f"eDNA Analysis Report Summary",
            f"=" * 40,
            f"",
            f"Report: {report_data.report_name or 'Unknown Report'}",
            f"Dataset: {dataset_name}",
            f"Created: {report_data.created_at}",
            f"Analysis Type: {report_data.analysis_type}",
            f"",
            f"Biodiversity Metrics:",
            f"- Shannon Diversity: {report_data.shannon_diversity:.4f}" if report_data.shannon_diversity else "- Shannon Diversity: N/A",
            f"- Simpson Diversity: {report_data.simpson_diversity:.4f}" if report_data.simpson_diversity else "- Simpson Diversity: N/A",
            f"- Species Richness: {report_data.species_richness}" if report_data.species_richness else "- Species Richness: N/A",
            f"- Evenness: {report_data.evenness:.4f}" if report_data.evenness else "- Evenness: N/A",
            f"",
            f"Environmental Context:",
            f"- Location: {dataset_info.collection_location if dataset_info else 'Unknown'}",
            f"- Depth: {dataset_info.depth_meters:.1f} m" if dataset_info and dataset_info.depth_meters else "- Depth: N/A",
            f"- Temperature: {dataset_info.temperature_celsius:.1f} °C" if dataset_info and dataset_info.temperature_celsius else "- Temperature: N/A",
            f"- Salinity: {dataset_info.salinity:.2f}" if dataset_info and dataset_info.salinity else "- Salinity: N/A",
            f"",
            f"Organisms Detected: {len(organisms)}",
            f"Novel Candidates: {sum(1 for org in organisms if org.is_novel_candidate)}",
            f"",
            f"Top 10 Most Abundant Organisms:"
        ]
        
        # Add top organisms
        top_organisms = sorted(organisms, key=lambda x: x.detection_count or 0, reverse=True)[:10]
        for i, org in enumerate(top_organisms, 1):
            summary_lines.append(f"{i:2d}. {org.organism_name or 'Unknown'} ({org.detection_count} detections)")
        
        return "\n".join(summary_lines)
    
    def _get_organisms_for_report(self, report_id: str):
        """Get organisms associated with a specific report."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT op.*
                    FROM organism_profiles op
                    JOIN sequences s ON op.organism_id = s.organism_id
                    WHERE s.report_id = ?
                    ORDER BY op.detection_count DESC
                """, (report_id,))
                
                organisms = []
                for row in cursor.fetchall():
                    organism = self.db_manager._row_to_organism_profile(row, cursor.description)
                    if organism:
                        organisms.append(organism)
                
                return organisms
        except Exception as e:
            st.error(f"Error loading organisms: {str(e)}")
            return []
    
    def _get_similarities_for_report(self, report_id: str):
        """Get similarity comparisons for a specific report."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        comparison_id, report_id_1, report_id_2, 
                        similarity_score, jaccard_similarity, 
                        organism_overlap_count, created_at
                    FROM similarity_matrices
                    WHERE report_id_1 = ? OR report_id_2 = ?
                    ORDER BY created_at DESC
                """, (report_id, report_id))
                
                similarities = []
                for row in cursor.fetchall():
                    # Create a simple similarity object
                    similarity = type('Similarity', (), {
                        'comparison_id': row[0],
                        'report_id_1': row[1],
                        'report_id_2': row[2],
                        'similarity_score': row[3],
                        'jaccard_similarity': row[4],
                        'organism_overlap_count': row[5],
                        'created_at': datetime.fromisoformat(row[6]) if row[6] else None
                    })()
                    similarities.append(similarity)
                
                return similarities
        except Exception as e:
            st.error(f"Error loading similarities: {str(e)}")
            return []
    
    def _get_dataset_info(self, dataset_id: str):
        """Get dataset information by dataset_id."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM datasets WHERE dataset_id = ?
                """, (dataset_id,))
                
                row = cursor.fetchone()
                if row:
                    # Convert row to dictionary
                    columns = [col[0] for col in cursor.description]
                    data = dict(zip(columns, row))
                    
                    # Create a simple dataset object
                    return type('DatasetInfo', (), {
                        'dataset_id': data.get('dataset_id'),
                        'dataset_name': data.get('dataset_name'),
                        'collection_location': data.get('collection_location'),
                        'depth_meters': data.get('depth_meters'),
                        'temperature_celsius': data.get('temperature_celsius'),
                        'ph_level': data.get('ph_level'),
                        'salinity': data.get('salinity'),
                        'file_size_mb': data.get('file_size_mb'),
                        'total_sequences': data.get('total_sequences'),
                        'collection_date': data.get('collection_date')
                    })()
                    
                return None
        except Exception as e:
            st.error(f"Error loading dataset info: {str(e)}")
            return None
    
    def render_organism_profiles_page(self):
        """Render organism profiles page."""
        st.title("Organism Profiles")
        
        # Search organisms
        col1, col2 = st.columns(2)
        
        with col1:
            search_query = st.text_input("Search organisms...")
        with col2:
            kingdom_filter = st.selectbox("Kingdom", ["All", "Bacteria", "Archaea", "Eukaryota"])
        
        # Get organisms
        search_kwargs = {}
        if search_query:
            search_kwargs['query'] = search_query
        if kingdom_filter != "All":
            search_kwargs['kingdom'] = kingdom_filter
        
        organisms = self.query_engine.search_organisms(limit=20, **search_kwargs)
        
        st.subheader(f"Found {len(organisms)} organisms")
        
        if organisms:
            for organism in organisms:
                with st.expander(f"🦠 {organism.organism_name or 'Unnamed organism'}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**ID:** {organism.organism_id}")
                        st.write(f"**Kingdom:** {organism.kingdom or 'Unknown'}")
                        st.write(f"**Genus:** {organism.genus or 'Unknown'}")
                    
                    with col2:
                        st.write(f"**Detection Count:** {organism.detection_count}")
                        st.write(f"**Novel Candidate:** {'Yes' if organism.is_novel_candidate else 'No'}")
                    
                    with col3:
                        if organism.first_detected:
                            st.write(f"**First Detected:** {organism.first_detected.strftime('%Y-%m-%d')}")
        else:
            st.info("No organisms found.")
    
    def render_similarity_analysis_page(self):
        """Render similarity analysis page."""
        st.title("Similarity Analysis")
        
        # Get similarity trends
        time_period = st.selectbox("Time Period (days)", [30, 60, 90])
        trends = self.cross_analysis_engine.get_similarity_trends(time_period)
        
        if trends and trends.get('daily_trends'):
            df_trends = pd.DataFrame(trends['daily_trends'])
            df_trends['date'] = pd.to_datetime(df_trends['date'])
            
            fig = px.line(df_trends, x='date', y='avg_similarity', title="Similarity Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Similarity", f"{trends['overall_statistics']['average_similarity']:.3f}")
            with col2:
                st.metric("Std Deviation", f"{trends['overall_statistics']['similarity_std']:.3f}")
            with col3:
                st.metric("Total Comparisons", trends['overall_statistics']['total_comparisons'])
        
        # Recent comparisons
        st.subheader("Recent Comparisons")
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT similarity_score, jaccard_similarity, created_at
                    FROM similarity_matrices 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                
                comparisons = []
                for row in cursor.fetchall():
                    comparisons.append({
                        'similarity_score': row[0],
                        'jaccard_similarity': row[1],
                        'created_at': row[2]
                    })
                
                if comparisons:
                    df_comp = pd.DataFrame(comparisons)
                    st.dataframe(df_comp, use_container_width=True)
                else:
                    st.info("No comparisons found.")
        except Exception as e:
            st.error(f"Failed to load comparisons: {str(e)}")
    
    def render_dataset_analyzer_page(self):
        """Render dataset analyzer page for uploading and analyzing new datasets."""
        st.title("📁 Dataset Analyzer")
        st.markdown("Upload and analyze new eDNA datasets to create analysis reports")
        
        # File upload section
        st.subheader("1. Upload Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a biological sequence file",
            type=['fasta', 'fa', 'fas', 'fastq', 'fq', 'swiss', 'gb', 'gbk', 'embl', 'em', 'gz'],
            help="Supported formats: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL (including gzipped files)"
        )
        
        # File information
        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 File: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Analysis configuration
        st.subheader("2. Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "Dataset Name",
                value=uploaded_file.name.split('.')[0] if uploaded_file else "My Dataset",
                help="Custom name for your analysis"
            )
            
            report_name = st.text_input(
                "Report Name (Optional)",
                help="Custom name for the analysis report"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["comprehensive", "taxonomy", "clustering", "novelty"],
                help="Choose the type of analysis to perform"
            )
            
            max_sequences = st.number_input(
                "Max Sequences (0 = all)",
                min_value=0,
                value=0,
                help="Limit the number of sequences to analyze"
            )
        
        # Environmental context
        with st.expander("🌍 Environmental Context (Optional)"):
            col1, col2 = st.columns(2)
            
            with col1:
                location = st.text_input("Collection Location")
                depth = st.number_input("Depth (m)", value=None, help="Sampling depth in meters")
            
            with col2:
                temperature = st.number_input("Temperature (°C)", value=None)
                ph = st.number_input("pH", value=None, min_value=0.0, max_value=14.0)
        
        # Analysis execution
        st.subheader("3. Run Analysis")
        
        if uploaded_file is not None:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                self.run_dataset_analysis(
                    uploaded_file, dataset_name, report_name, analysis_type, 
                    max_sequences, location, depth, temperature, ph
                )
        else:
            st.info("Please upload a dataset file to start analysis")
    
    def run_dataset_analysis(self, uploaded_file, dataset_name, report_name, analysis_type, 
                           max_sequences, location, depth, temperature, ph):
        """Execute dataset analysis and store results."""
        
        # Create temporary file
        tmp_file_path = None
        output_path = None
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize analyzer
            status_text.text("Initializing analyzer...")
            analyzer = DatasetAnalyzer()
            progress_bar.progress(20)
            
            # Initialize output path
            output_path = None
            
            # Create output file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as output_file:
                output_path = output_file.name
            
            # Run analysis
            status_text.text("Running analysis...")
            progress_bar.progress(40)
            
            analysis_results = analyzer.analyze_dataset(
                input_path=tmp_file_path,
                output_path=output_path,
                dataset_name=dataset_name,
                max_sequences=max_sequences if max_sequences > 0 else None
            )
            
            progress_bar.progress(70)
            status_text.text("Storing results...")
            
            # Prepare environmental context
            environmental_context = {}
            if location:
                environmental_context['location'] = location
            if depth is not None:
                environmental_context['depth'] = depth
            if temperature is not None:
                environmental_context['temperature'] = temperature
            if ph is not None:
                environmental_context['ph'] = ph
            
            # Store analysis report
            report_id, storage_path = self.catalogue_manager.store_analysis_report(
                dataset_file_path=tmp_file_path,
                analysis_results=analysis_results,
                report_name=report_name or f"{dataset_name}_analysis",
                environmental_context=environmental_context if environmental_context else None
            )
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Display success message
            st.success(f"✅ Analysis completed successfully!")
            st.info(f"📋 Report ID: {report_id}")
            st.info(f"💾 Storage Path: {storage_path}")
            
            # Display quick results summary
            if analysis_results and 'basic_stats' in analysis_results:
                st.subheader("📊 Quick Results Summary")
                
                stats = analysis_results['basic_stats']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Sequences", f"{stats.get('total_sequences', 0):,}")
                with col2:
                    st.metric("Avg Length", f"{stats.get('avg_length', 0):.1f}")
                with col3:
                    st.metric("GC Content", f"{stats.get('gc_content', 0):.1f}%")
                with col4:
                    st.metric("Unique Sequences", f"{stats.get('unique_sequences', 0):,}")
                
                # Show a button to view the full report
                if st.button("View Full Report", type="secondary"):
                    st.session_state.current_page = 'Report Browser'
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())
        
        finally:
            # Cleanup temporary files
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass


def main():
    """Main function to run the dashboard."""
    dashboard = ReportManagementDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()