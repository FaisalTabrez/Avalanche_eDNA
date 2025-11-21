"""
Results Viewer Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
import streamlit.components.v1 as components
from src.utils.config import config as app_config

def render():
    """Display results viewer page"""
    st.title("Results Viewer")

    # Choose results directory (support prefill from homepage)
    default_dir = str(Path(app_config.get('storage.runs_dir', 'runs')).resolve())
    prefill = st.session_state.get('prefill_results_dir')
    if prefill and Path(prefill).exists():
        default_dir = prefill
    
    results_dir = st.text_input(
        "Results directory",
        value=default_dir,
        help="Folder containing pipeline_results.json and subfolders: clustering, taxonomy, novelty, visualizations",
        key="results_dir_input"
    )
    # Persist last used results dir for quick navigation
    if results_dir:
        st.session_state.prefill_results_dir = results_dir

    if not results_dir:
        return
    base = Path(results_dir)
    if not base.exists():
        st.warning(f"Directory not found: {base}")
        return

    # Helper to check files
    def p(*parts):
        return base.joinpath(*parts)

    st.markdown("---")
    # 1) Pipeline summary
    st.subheader("Pipeline Summary")
    summary_cols = st.columns(5)
    pr_path = p('pipeline_results.json')
    ui_report_path = p('analysis_report.txt')
    try:
        if pr_path.exists():
            with open(pr_path, 'r', encoding='utf-8') as f:
                pr = json.load(f)
            s = pr.get('summary', {})
            summary_cols[0].metric("Total sequences", f"{s.get('total_sequences_processed', 0):,}")
            summary_cols[1].metric("Clusters", f"{s.get('total_clusters', 0):,}")
            summary_cols[2].metric("Taxa identified", f"{s.get('total_taxa_identified', 0):,}")
            summary_cols[3].metric("Novel candidates", f"{s.get('novel_taxa_candidates', 0):,}")
            summary_cols[4].metric("Novel %", f"{s.get('novelty_percentage', 0)}%")
        else:
            # Fallback to UI report-only runs
            if ui_report_path.exists():
                summary_cols[0].metric("Report only", "Yes")
                st.info("Detected UI analysis report (no pipeline_results.json). Showing the text report below.")
            else:
                st.info("pipeline_results.json not found; showing sections based on available files")
    except Exception as e:
        st.warning(f"Could not read pipeline_results.json: {e}")

    # UI text report display
    if ui_report_path.exists():
        st.markdown("### Analysis Report (UI)")
        try:
            st.text_area("analysis_report.txt", ui_report_path.read_text(encoding='utf-8', errors='ignore'), height=240)
            st.download_button("Download analysis_report.txt", data=ui_report_path.read_bytes(), file_name="analysis_report.txt")
        except Exception:
            st.write(f"Open report: {ui_report_path}")

    st.markdown("---")
    # 2) Clustering
    st.subheader("Clustering")
    clus_img = p('clustering', 'cluster_visualization.png')
    clus_stats = p('clustering', 'cluster_stats.txt')
    clus_csv = p('clustering', 'cluster_assignments.csv')

    cols = st.columns([2,1])
    with cols[0]:
        if clus_img.exists():
            st.image(str(clus_img), caption="Cluster visualization")
        else:
            st.info("No cluster_visualization.png found")
    with cols[1]:
        if clus_stats.exists():
            try:
                st.text_area("Cluster stats", clus_stats.read_text(encoding='utf-8'), height=200)
            except Exception:
                st.text("cluster_stats.txt present (could not display)")
        if clus_csv.exists():
            try:
                dfc = pd.read_csv(clus_csv)
                st.write("Cluster assignments (first 10 rows):")
                st.dataframe(dfc.head(10), hide_index=True, use_container_width=True)
                st.download_button("Download cluster_assignments.csv", data=clus_csv.read_bytes(), file_name="cluster_assignments.csv")
            except Exception as e:
                st.warning(f"Could not read cluster_assignments.csv: {e}")

    st.markdown("---")
    # 3) Taxonomy
    st.subheader("Taxonomy")
    tax_csv = p('taxonomy', 'taxonomy_predictions.csv')
    if tax_csv.exists():
        try:
            dft = pd.read_csv(tax_csv)
            # Summary
            tcols = st.columns(4)
            tcols[0].metric("Rows", f"{len(dft):,}")
            if 'assigned_label' in dft.columns:
                tcols[1].metric("Unique taxa", f"{dft['assigned_label'].dropna().nunique():,}")
            if 'tiebreak_winner' in dft.columns:
                try:
                    tb_counts = dft['tiebreak_winner'].value_counts()
                    tcols[2].metric("BLAST wins", f"{int(tb_counts.get('blast',0)):,}")
                    tcols[3].metric("KNN wins", f"{int(tb_counts.get('knn',0)):,}")
                except Exception:
                    pass
            st.write("Top taxa (bar chart)")
            if 'assigned_label' in dft.columns:
                top_counts = dft['assigned_label'].fillna('Unknown').value_counts().head(15)
                chart_df = pd.DataFrame({"Taxon": top_counts.index, "Count": top_counts.values})
                fig = px.bar(chart_df, x='Taxon', y='Count', title="Top 15 Assigned Taxa")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dft.head(50), use_container_width=True, hide_index=True)
            st.download_button("Download taxonomy_predictions.csv", data=tax_csv.read_bytes(), file_name="taxonomy_predictions.csv")
            # Conflicts report
            tb_report = p('taxonomy', 'taxonomy_tiebreak_report.csv')
            if tb_report.exists():
                st.download_button("Download taxonomy_tiebreak_report.csv", data=tb_report.read_bytes(), file_name="taxonomy_tiebreak_report.csv")
        except Exception as e:
            st.warning(f"Could not parse taxonomy_predictions.csv: {e}")
    else:
        st.info("No taxonomy_predictions.csv found")

    st.markdown("---")
    # 4) Novelty
    st.subheader("Novelty")
    nov_json = p('novelty', 'novelty_analysis.json')
    nov_img = p('novelty', 'novelty_visualization.png')
    if nov_json.exists():
        try:
            nov = json.loads(nov_json.read_text(encoding='utf-8'))
            ncols = st.columns(4)
            ncols[0].metric("Total sequences", f"{nov.get('total_sequences',0):,}")
            ncols[1].metric("Novel candidates", f"{nov.get('novel_candidates',0):,}")
            ncols[2].metric("Novel %", f"{nov.get('novel_percentage',0):.2f}%")
            if nov_img.exists():
                st.image(str(nov_img), caption="Novelty visualization")
        except Exception as e:
            st.warning(f"Could not read novelty_analysis.json: {e}")
    else:
        st.info("No novelty analysis found")

    st.markdown("---")
    # 5) Dashboard
    st.subheader("Dashboard")
    dash_html = p('visualizations', 'analysis_dashboard.html')
    if dash_html.exists():
        try:
            components.html(dash_html.read_text(encoding='utf-8'), height=600, scrolling=True)
        except Exception:
            st.write(f"Open dashboard: {dash_html}")
    else:
        st.info("No analysis_dashboard.html found")
    
    st.title("Results Viewer")
    st.info("Results viewer coming soon! This will allow you to browse and compare previous analyses.")
    
    # Placeholder for results viewer
    st.markdown("""
    ### Planned Features:
    - **Analysis history**
    - **Result comparison**
    - **Trend analysis**
    - **Export utilities**
    """)
