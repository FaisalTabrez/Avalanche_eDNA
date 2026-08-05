from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render():
    st.title("🔍 Data Explorer")

    # Load Data
    data_path = Path("consolidated_data/results/demo_taxonomy.csv")

    if not data_path.exists():
        st.warning("No analysis results found. Please run an analysis first.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar Filters
    with st.sidebar:
        st.header("Filters")
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.9)
        selected_phyla = st.multiselect("Filter Phylum", df["Phylum"].unique())

    # Apply Filters
    filtered_df = df[df["Confidence"] >= min_conf]
    if selected_phyla:
        filtered_df = filtered_df[filtered_df["Phylum"].isin(selected_phyla)]

    # Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sequences", len(filtered_df))
    col2.metric("Unique Species", filtered_df["Species"].nunique())
    col3.metric("Avg Confidence", f"{filtered_df['Confidence'].mean():.2f}")

    st.markdown("---")

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Taxonomy Sunburst", "Species Abundance", "Raw Data"])

    with tab1:
        st.subheader("Taxonomic Distribution")
        if not filtered_df.empty:
            fig = px.sunburst(
                filtered_df,
                path=["Phylum", "Class", "Order", "Family", "Genus"],
                values="Confidence",  # Using confidence as a proxy for weight/count if count col missing
                color="Confidence",
                color_continuous_scale="RdBu",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data matches filters.")

    with tab2:
        st.subheader("Top Species")
        if not filtered_df.empty:
            species_counts = filtered_df["Species"].value_counts().reset_index()
            species_counts.columns = ["Species", "Count"]

            fig = px.bar(
                species_counts.head(10),
                x="Count",
                y="Species",
                orientation="h",
                color="Count",
                title="Top 10 Identified Species",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data matches filters.")

    with tab3:
        st.subheader("Sequence Data")
        st.dataframe(filtered_df, use_container_width=True)
