#!/usr/bin/env python3
"""
SRA Integration Example for eDNA Biodiversity Assessment

This script demonstrates how to use NCBI SRA data with the eDNA analysis pipeline.
It shows the complete workflow from SRA data download to biodiversity analysis.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import config
from scripts.download_sra_data import SRADownloader
from src.preprocessing.sra_processor import SRAProcessor
from scripts.run_pipeline import eDNABiodiversityPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_sra_integration():
    """Demonstrate complete SRA integration workflow"""

    logger.info("Starting SRA Integration Demonstration")
    logger.info("=" * 60)

    # Step 1: Search and download SRA data
    logger.info("Step 1: Searching for eDNA-relevant studies...")
    downloader = SRADownloader()

    # Search for marine sediment eDNA studies
    studies = downloader.search_edna_studies(max_results=5)
    logger.info(f"Found {len(studies)} relevant eDNA studies")

    if not studies:
        logger.warning("No studies found. Using predefined study list...")
        # Use predefined studies from config
        studies = downloader.sra_config.get('edna_studies', {}).get('marine_sediment', [])

    # Download first study
    if studies:
        study_accession = studies[0].get('accession', studies[0]) if isinstance(studies[0], dict) else studies[0]

        logger.info(f"Downloading study: {study_accession}")
        success = downloader.download_sra_run(study_accession)

        if success:
            logger.info("✅ SRA download completed successfully")

            # Step 2: Process SRA data
            logger.info("Step 2: Processing SRA data...")
            sra_processor = SRAProcessor()

            # Find downloaded FASTQ files
            sra_dir = Path("data/sra") / study_accession
            fastq_files = list(sra_dir.rglob("*.fastq.gz"))

            if fastq_files:
                logger.info(f"Found {len(fastq_files)} FASTQ files")

                # Process with SRA processor
                results = sra_processor.integrate_with_pipeline(fastq_files[:2])  # Process first 2 files

                logger.info("✅ SRA processing completed")
                logger.info(f"   - Processed {results['results']['total_processed_sequences']} sequences")
                logger.info(f"   - From {results['results']['total_sra_files']} SRA files")

                # Step 3: Run full biodiversity analysis
                logger.info("Step 3: Running biodiversity analysis...")

                # Prepare sequences for pipeline
                sequences = [str(seq.seq) for seq in results['sequences']]

                if sequences:
                    # Save sequences for pipeline
                    temp_fasta = Path("temp_sra_sequences.fasta")
                    with open(temp_fasta, 'w') as f:
                        for i, seq in enumerate(sequences):
                            f.write(f">SRA_seq_{i}\n{seq}\n")

                    # Run analysis pipeline
                    output_dir = Path("results/sra_analysis")

                    try:
                        pipeline = eDNABiodiversityPipeline()
                        analysis_results = pipeline.run_complete_pipeline(
                            input_data=str(temp_fasta),
                            output_dir=str(output_dir),
                            run_preprocessing=False,  # Skip since already processed
                            run_embedding=True,
                            run_clustering=True,
                            run_taxonomy=True,
                            run_novelty=True,
                            run_visualization=True
                        )

                        logger.info("✅ Biodiversity analysis completed")
                        logger.info(f"   - Total sequences: {analysis_results['summary']['total_sequences_processed']}")
                        logger.info(f"   - Clusters found: {analysis_results['summary']['total_clusters']}")
                        logger.info(f"   - Taxa identified: {analysis_results['summary']['total_taxa_identified']}")
                        logger.info(f"   - Novel candidates: {analysis_results['summary']['novel_taxa_candidates']}")
                        logger.info(f"   - Results saved to: {output_dir}")

                        # Cleanup
                        temp_fasta.unlink()

                        return True

                    except Exception as e:
                        logger.error(f"❌ Analysis pipeline failed: {e}")
                        return False
                else:
                    logger.error("❌ No sequences to analyze")
                    return False
            else:
                logger.error("❌ No FASTQ files found after download")
                return False
        else:
            logger.error("❌ SRA download failed")
            return False
    else:
        logger.error("❌ No studies available for download")
        return False

def show_sra_configuration():
    """Display current SRA configuration"""
    logger.info("Current SRA Configuration:")
    logger.info("-" * 40)

    sra_config = config.get('databases', {}).get('sra', {})

    logger.info(f"SRA Base URL: {sra_config.get('sra_base_url', 'Not configured')}")
    logger.info(f"SRA FTP: {sra_config.get('sra_ftp', 'Not configured')}")

    edna_studies = sra_config.get('edna_studies', {})
    for study_type, studies in edna_studies.items():
        logger.info(f"{study_type}: {len(studies)} studies configured")

    search_config = sra_config.get('search', {})
    logger.info(f"Search keywords: {search_config.get('edna_keywords', [])}")
    logger.info(f"Min spots: {search_config.get('min_spots', 'Not set')}")

def main():
    """Main demonstration function"""
    logger.info("NCBI SRA Integration Demonstration")
    logger.info("===================================")

    # Show configuration
    show_sra_configuration()
    logger.info("")

    # Check for SRA Toolkit
    try:
        import subprocess
        result = subprocess.run(['which', 'prefetch'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ SRA Toolkit found")
        else:
            logger.warning("⚠️ SRA Toolkit not found. Install for better performance:")
            logger.info("   conda install -c bioconda sra-tools")
    except Exception:
        logger.warning("⚠️ Could not check for SRA Toolkit")

    logger.info("")

    # Run demonstration
    start_time = time.time()
    success = demonstrate_sra_integration()
    elapsed_time = time.time() - start_time

    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ SRA Integration Demonstration Completed Successfully!")
        logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Explore results in: results/sra_analysis/")
        logger.info("2. View interactive dashboard: results/sra_analysis/visualizations/analysis_dashboard.html")
        logger.info("3. Check SRA metadata: data/sra/")
        logger.info("4. Try with your own SRA accessions using: python scripts/download_sra_data.py --accession YOUR_ACCESSION")
    else:
        logger.error("❌ SRA Integration Demonstration Failed")
        logger.error("Check the logs above for details")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
