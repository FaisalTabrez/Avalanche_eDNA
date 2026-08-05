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
    
    ### 🚀 **Dynamic Scaling System (NEW!)**
    - **Auto-Scaling**: Automatically adapts from 10 to 10,000+ clusters
    - **Hybrid Memory Buffers**: Exemplar + Uncertainty + Recent buffers
    - **Memory Management**: Intelligent budget allocation and monitoring
    - **Auto-Adaptation**: System adjusts configuration as dataset grows
    - **Advanced Refinements**: Temperature scaling, reservoir sampling, mini-retrieval, LoRA
    
    ### 📊 **Analysis Features**
    - **Universal Format Support**: FASTA, FASTQ, Swiss-Prot, GenBank, EMBL
    - **Large File Processing**: Supports datasets up to 10GB
    - **Automatic Analysis**: Intelligent sequence type detection
    - **Performance Optimized**: Parallel processing and vectorized computations
    
    ### 🧬 **Taxonomy Classification**
    - **Continual Learning**: Sequential cluster training with memory replay
    - **BLAST Integration**: Traditional homology-based assignment
    - **ML Classification**: Advanced neural network classifiers
    - **Hybrid Approach**: Combines BLAST and ML for best results
    
    ### 📈 **Visualization & Monitoring**
    - **Interactive Charts**: Real-time training metrics and adaptation events
    - **Buffer Composition**: Visual breakdown of memory allocation
    - **Memory Tracking**: Live memory usage monitoring
    - **Configuration History**: Track all system adaptations
    
    ## Technology Stack
    
    - **Python 3.13** - Core programming language
    - **BioPython** - Biological sequence analysis
    - **PyTorch** - Deep learning and continual learning
    - **DNABERT-2** - Pre-trained DNA sequence embeddings
    - **Streamlit** - Web interface framework
    - **Plotly** - Interactive visualizations
    - **NumPy & Pandas** - Data processing
    
    ## Dynamic Scaling Performance
    
    ### Validated Results
    - **Real eDNA**: 71.5% accuracy on actual environmental samples
    - **SwissProt**: 60.7% accuracy on 4,703 sequences
    - **Memory Efficiency**: 0.2-0.3% budget usage per cluster
    - **Adaptation**: Auto-adjusts every ~10-20 clusters
    
    ### Scaling Capabilities
    | Clusters | Accuracy | Memory (2GB budget) |
    |----------|----------|---------------------|
    | 10       | 96%      | ~20 MB             |
    | 25       | 89%      | ~50 MB             |
    | 50       | 85%      | ~100 MB            |
    | 100+     | 75-80%   | ~200 MB            |
    | 1000+    | 70-75%   | ~2 GB              |
    
    ## System Architecture
    
    ### Pipeline v2 with Dynamic Scaling
    ```
    Input FASTA
        ↓
    DNABERT-2 Embeddings
        ↓
    Clustering (k-means/hierarchical)
        ↓
    Dynamic Scaling Controller
        ├─ Auto-Scale Configuration
        ├─ Monitor Memory Budget
        └─ Trigger Adaptations
        ↓
    Continual Learning Training
        ├─ Exemplar Buffer (diverse samples)
        ├─ Uncertainty Buffer (hard cases)
        └─ Recent Buffer (temporal patterns)
        ↓
    Taxonomy Assignment
        ├─ BLAST (homology)
        ├─ ML Classifier (neural network)
        └─ Hybrid (combined)
        ↓
    Results & Visualizations
    ```
    
    ## Advanced Features
    
    ### 1. **Temperature-Scaled Confidence**
    - Identifies uncertain predictions using scaled softmax
    - Dynamic temperature adjustment based on cluster count
    - Populates uncertainty buffer with challenging samples
    
    ### 2. **Reservoir Sampling**
    - Ensures diverse exemplar selection within clusters
    - Prevents redundant similar samples
    - Weighted sampling based on sequence diversity
    
    ### 3. **Mini-Retrieval**
    - k-NN retrieval of most relevant replay samples
    - Embedding-space similarity for smart replay
    - Improves training effectiveness
    
    ### 4. **Centroid Tracking**
    - Maintains running cluster centroids
    - Enables drift detection
    - Supports adaptive re-clustering
    
    ### 5. **LoRA Adaptation**
    - Low-Rank Adaptation for efficient fine-tuning
    - Reduces trainable parameters
    - Faster adaptation to new clusters
    
    ## Getting Started
    
    ### Quick Start
    1. **Upload Dataset**: Use the Analysis page to upload FASTA files
    2. **Configure**: Choose dynamic scaling preset or customize
    3. **Run Pipeline**: Execute full taxonomy classification
    4. **Monitor**: Watch real-time adaptation and training metrics
    5. **Review**: Analyze results and download reports
    
    ### Dynamic Scaling Configuration
    - Visit **🚀 Dynamic Scaling** page for configuration presets
    - Choose preset based on dataset size
    - Or build custom configuration with fine-grained control
    - Export/import configurations for reproducibility
    
    ## Documentation & Support
    
    ### Internal Documentation
    - `DYNAMIC_SCALING_INTEGRATION.md` - Integration guide
    - `EDNA_REPORT_MANAGEMENT_SYSTEM_GUIDE.md` - Report system
    - `BLAST_INTEGRATION_GUIDE.md` - BLAST taxonomy setup
    - `docs/` - API reference and user guides
    
    ### Key Files
    - `scripts/run_taxonomy_pipeline_v2.py` - Main pipeline with dynamic scaling
    - `src/models/dynamic_hybrid_buffer.py` - Dynamic scaling controller
    - `src/models/hybrid_memory_buffer.py` - Hybrid buffer implementation
    - `src/clustering/taxonomy.py` - Taxonomy assignment
    
    ## Version History
    
    ### v2.0 (Current) - Dynamic Scaling Release
    - ✨ **NEW**: Complete dynamic scaling system
    - ✨ **NEW**: Auto-adaptation for 10-10,000+ clusters
    - ✨ **NEW**: Hybrid memory buffers
    - ✨ **NEW**: 5 advanced refinements
    - ✨ **NEW**: Interactive configuration UI
    - ✨ **NEW**: Real-time monitoring and visualization
    - ✅ Validated on real eDNA and SwissProt data
    
    ### v1.0 - Legacy System
    - Basic continual learning with fixed buffers
    - Active replay mechanism
    - EWC regularization
    - BLAST and ML taxonomy
    
    ## Credits & Acknowledgments
    
    **Development Team**: eDNA Analysis Platform
    
    **Technologies**:
    - DNABERT-2: Pre-trained DNA language model
    - Avalanche: Continual learning library
    - BioPython: Sequence analysis tools
    - NCBI BLAST: Homology-based taxonomy
    
    ## License
    
    See LICENSE file for details.
    
    ## Contact
    
    For questions, issues, or feature requests, please contact the development team.
    """)
