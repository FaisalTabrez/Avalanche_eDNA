"""
Taxonomy Viewer Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

def render():
    """Display taxonomy viewer page with conflict and lineage summaries"""
    st.title("Taxonomy Viewer")
    st.info("Load taxonomy_predictions.csv (and optional tiebreak report) from a results directory.")

    col1, col2 = st.columns([2, 1])
    with col1:
        default_dir = str(Path("results").resolve())
        results_dir = st.text_input("Results directory", value=default_dir, help="Folder containing taxonomy/taxonomy_predictions.csv")
    with col2:
        refresh = st.button("Load", type="primary")

    if results_dir:
        tax_csv = Path(results_dir) / "taxonomy" / "taxonomy_predictions.csv"
        report_csv = Path(results_dir) / "taxonomy" / "taxonomy_tiebreak_report.csv"
        if not tax_csv.exists():
            st.warning(f"taxonomy_predictions.csv not found at {tax_csv}")
            return
        try:
            df = pd.read_csv(tax_csv)
        except Exception as e:
            st.error(f"Failed to read taxonomy_predictions.csv: {e}")
            return

        # Summary cards
        total = len(df)
        unique_labels = df['assigned_label'].dropna().nunique() if 'assigned_label' in df.columns else 0
        conflicts = 0
        if 'conflict_flag' in df.columns:
            try:
                conflicts = int(df['conflict_flag'].astype(str).str.lower().eq('true').sum())
            except Exception:
                conflicts = 0
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Total Sequences", f"{total:,}")
        colB.metric("Unique Taxa", f"{unique_labels:,}")
        colC.metric("Conflicts", f"{conflicts:,}")
        if report_csv.exists():
            colD.download_button("Download Tiebreak Report", data=report_csv.read_bytes(), file_name="taxonomy_tiebreak_report.csv")

        st.markdown("---")

        # Filters
        with st.expander("Filters"):
            only_conflicts = st.checkbox("Show only conflicts", value=False)
            top_n = st.number_input("Top N taxa chart", min_value=5, max_value=100, value=15)

        view_df = df.copy()
        if only_conflicts and 'conflict_flag' in view_df.columns:
            try:
                view_df = view_df[view_df['conflict_flag'].astype(str).str.lower() == 'true']
            except Exception:
                pass

        # Top taxa chart
        try:
            if 'assigned_label' in view_df.columns and len(view_df) > 0:
                top_counts = view_df['assigned_label'].fillna('Unknown').value_counts().head(int(top_n))
                chart_df = pd.DataFrame({"Taxon": top_counts.index, "Count": top_counts.values})
                fig = px.bar(chart_df, x='Taxon', y='Count', title=f"Top {int(top_n)} Assigned Taxa")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        # Simplified data table with core columns
        preferred_cols = [
            'sequence_id', 'assigned_rank', 'assigned_label', 'confidence',
            'tiebreak_winner', 'conflict_flag'
        ]
        cols = [c for c in preferred_cols if c in view_df.columns]
        st.dataframe(view_df[cols], use_container_width=True, hide_index=True)
        
        st.caption("Download the complete, unfiltered taxonomy table below.")
        
        # Download full (complete) table
        st.download_button(
            label="Download full taxonomy_predictions.csv (complete table)",
            data=tax_csv.read_bytes(),
            file_name="taxonomy_predictions.csv",
            use_container_width=True
        )
