"""
SRA Browser Page
"""
import streamlit as st
from pathlib import Path
try:
    from src.utils.sra_integration import SRAIntegrationUI
except ImportError:
    SRAIntegrationUI = None

def render():
    """Display SRA dataset browser and batch download interface"""
    
    st.title("NCBI SRA Dataset Browser")
    st.markdown("""
    Search and download datasets from NCBI Sequence Read Archive (SRA).
    Find eDNA and metabarcoding datasets for analysis or model training.
    """)
    
    if not SRAIntegrationUI:
        st.error("SRA integration module not available. Please check installation.")
        return
    
    sra_ui = SRAIntegrationUI()
    
    # Check toolkit status
    sra_ui.show_sra_toolkit_status()
    
    if not sra_ui.sra_toolkit_available:
        st.info("Install SRA Toolkit to enable dataset downloads.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Search & Browse", "Batch Download", "Downloaded Datasets"])
    
    with tab1:
        st.markdown("### Search NCBI SRA")
        
        # Search interface
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_terms = st.text_input(
                "Search Keywords",
                placeholder="eDNA, 18S rRNA, marine metabarcoding",
                help="Enter comma-separated keywords to search"
            )
        
        with col2:
            max_results = st.number_input("Max Results", min_value=10, max_value=100, value=30)
        
        with col3:
            st.markdown("&nbsp;")  # Spacing
            search_button = st.button("Search", type="primary", use_container_width=True)
        
        if search_button:
            keywords = [term.strip() for term in search_terms.split(',') if term.strip()]
            
            with st.spinner("Searching NCBI SRA database..."):
                results = sra_ui.search_sra_datasets(keywords, max_results)
                st.session_state.sra_search_results = results
        
        # Display results
        if 'sra_search_results' in st.session_state and st.session_state.sra_search_results:
            results = st.session_state.sra_search_results
            st.success(f"Found {len(results)} datasets")
            
            # Add filters
            st.markdown("#### Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                organism_filter = st.multiselect(
                    "Filter by Organism",
                    options=list(set(r.get('organism', 'Unknown') for r in results)),
                    default=[]
                )
            
            with col2:
                platform_filter = st.multiselect(
                    "Filter by Platform",
                    options=list(set(r.get('platform', 'Unknown') for r in results)),
                    default=[]
                )
            
            # Apply filters
            filtered_results = results
            if organism_filter:
                filtered_results = [r for r in filtered_results if r.get('organism') in organism_filter]
            if platform_filter:
                filtered_results = [r for r in filtered_results if r.get('platform') in platform_filter]
            
            st.markdown(f"#### Results ({len(filtered_results)} datasets)")
            
            # Display in expandable cards
            for idx, study in enumerate(filtered_results):
                with st.expander(
                    f"**{study.get('accession', 'Unknown')}** - {study.get('title', 'No title')[:100]}..."
                ):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Accession:** `{study.get('accession', 'N/A')}`")
                        st.markdown(f"**Organism:** {study.get('organism', 'N/A')}")
                        st.markdown(f"**Platform:** {study.get('platform', 'N/A')}")
                        st.markdown(f"**Title:** {study.get('title', 'N/A')}")
                    
                    with col2:
                        spots = study.get('spots', '0')
                        bases = study.get('bases', '0')
                        st.metric("Spots", f"{int(spots):,}" if spots.isdigit() else spots)
                        st.metric("Bases", f"{int(bases):,}" if bases.isdigit() else bases)
                    
                    with col3:
                        # Download button
                        if st.button(f"Download", key=f"dl_{idx}"):
                            accession = study.get('accession')
                            output_dir = Path("data/sra") / accession
                            
                            status_text = st.empty()
                            progress_bar = st.progress(0)
                            
                            def update_progress(msg):
                                status_text.text(msg)
                            
                            progress_bar.progress(10)
                            success, file_path = sra_ui.download_sra_dataset(
                                accession,
                                output_dir,
                                progress_callback=update_progress
                            )
                            
                            if success:
                                progress_bar.progress(100)
                                status_text.text("Download complete!")
                                st.success(f"Downloaded to {file_path}")
                                
                                # Add to batch download list
                                if 'downloaded_sra' not in st.session_state:
                                    st.session_state.downloaded_sra = []
                                st.session_state.downloaded_sra.append({
                                    'accession': accession,
                                    'path': str(file_path),
                                    'metadata': study
                                })
                            else:
                                status_text.text("Download failed")
                                st.error("Download failed")
                        
                        # Add to batch queue
                        if st.button(f"Add to Queue", key=f"queue_{idx}"):
                            if 'sra_batch_queue' not in st.session_state:
                                st.session_state.sra_batch_queue = []
                            
                            if study not in st.session_state.sra_batch_queue:
                                st.session_state.sra_batch_queue.append(study)
                                st.success(f"Added {study.get('accession')} to queue")
                            else:
                                st.warning("Already in queue")
    
    with tab2:
        st.markdown("### Batch Download Queue")
        
        if 'sra_batch_queue' not in st.session_state or not st.session_state.sra_batch_queue:
            st.info("No datasets in queue. Add datasets from the Search tab.")
        else:
            queue = st.session_state.sra_batch_queue
            st.success(f"{len(queue)} datasets in queue")
            
            # Display queue
            for idx, study in enumerate(queue):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{idx+1}.** {study.get('accession')} - {study.get('title', 'No title')[:80]}...")
                with col2:
                    if st.button(f"Remove", key=f"remove_{idx}"):
                        st.session_state.sra_batch_queue.pop(idx)
                        st.rerun()
            
            st.markdown("---")
            
            # Batch download controls
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Download All", type="primary", use_container_width=True):
                    st.markdown("### Download Progress")
                    
                    for idx, study in enumerate(queue):
                        accession = study.get('accession')
                        st.markdown(f"**{idx+1}/{len(queue)}:** Downloading {accession}...")
                        
                        output_dir = Path("data/sra") / accession
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(msg):
                            status_text.text(msg)
                        
                        progress_bar.progress(10)
                        success, file_path = sra_ui.download_sra_dataset(
                            accession,
                            output_dir,
                            progress_callback=update_progress
                        )
                        
                        if success:
                            progress_bar.progress(100)
                            status_text.text("Complete")
                            st.success(f"Downloaded {accession}")
                            
                            # Add to downloaded list
                            if 'downloaded_sra' not in st.session_state:
                                st.session_state.downloaded_sra = []
                            st.session_state.downloaded_sra.append({
                                'accession': accession,
                                'path': str(file_path),
                                'metadata': study
                            })
                        else:
                            status_text.text("Failed")
                            st.error(f"Failed to download {accession}")
                    
                    st.success("Batch download complete!")
                    st.session_state.sra_batch_queue = []
            
            with col2:
                if st.button("Clear Queue", use_container_width=True):
                    st.session_state.sra_batch_queue = []
                    st.rerun()
    
    with tab3:
        st.markdown("### Downloaded Datasets")
        
        if 'downloaded_sra' not in st.session_state or not st.session_state.downloaded_sra:
            st.info("No datasets downloaded yet.")
        else:
            downloads = st.session_state.downloaded_sra
            st.success(f"{len(downloads)} datasets downloaded")
            
            for idx, item in enumerate(downloads):
                with st.expander(f"**{item['accession']}** - {item.get('metadata', {}).get('title', 'No title')[:80]}..."):
                    st.markdown(f"**Path:** `{item['path']}`")
                    st.markdown(f"**Organism:** {item.get('metadata', {}).get('organism', 'N/A')}")
                    st.markdown(f"**Platform:** {item.get('metadata', {}).get('platform', 'N/A')}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"Analyze Dataset", key=f"analyze_{idx}"):
                            st.info(f"Navigate to 'Dataset Analysis' page to analyze {item['path']}")
                    
                    with col2:
                        if st.button(f"Use for Training", key=f"train_{idx}"):
                            st.info(f"Navigate to 'Model Training' page to use this dataset")
