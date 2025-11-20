"""
About Page
"""
import streamlit as st

def render():
    """Display about page"""
    
    st.title("About eDNA Biodiversity Assessment System")
    
    st.markdown("""
    ## Project Overview
    
    The eDNA Biodiversity Assessment System is an end-to-end platform for identifying 
    taxonomic diversity and assessing biological richness in deep-sea environmental DNA 
    (eDNA) datasets using advanced machine learning and bioinformatics techniques.
    
    ## Key Features
    
    - **Universal Format Support**: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL
    - **Large File Processing**: Supports datasets up to 2GB in size
    - **Automatic Analysis**: Intelligent sequence type detection and analysis
    - **Performance Optimized**: Parallel processing and vectorized computations
    - **Interactive Visualizations**: Real-time charts and plots
    - **Comprehensive Reports**: Detailed analysis summaries
    
    ## Technology Stack
    
    - **Python 3.13** - Core programming language
    - **BioPython** - Biological sequence analysis
    - **Streamlit** - Web interface framework
    - **Plotly** - Interactive visualizations
    - **NumPy & Pandas** - Data processing
    - **PyTorch** - Deep learning capabilities
    
    ## Support
    
    For questions, issues, or feature requests, please contact the development team.
    """)
