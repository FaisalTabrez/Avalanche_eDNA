#!/usr/bin/env python
"""
Demo script showing complete eDNA biodiversity analysis workflow
"""

import os
import sys
import subprocess
from pathlib import Path
import time

def main():
    """Run complete demo of the eDNA biodiversity assessment system"""
    
    print("🌊" + "="*60)
    print("  Deep-Sea eDNA Biodiversity Assessment System Demo")
    print("="*60 + "🌊")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Steps
    steps = [
        "🏗️  Setting up environment",
        "📊 Creating sample eDNA dataset", 
        "⚙️  Running preprocessing pipeline",
        "🧠 Generating sequence embeddings",
        "🔗 Clustering sequences", 
        "🏷️  Assigning taxonomy",
        "🆕 Detecting novel taxa",
        "📈 Creating visualizations",
        "🎯 Generating final report"
    ]
    
    print("\n📋 Demo Workflow:")
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    
    print("\n" + "="*60)
    input("Press Enter to start the demo...")
    
    try:
        # Step 1: Create sample data
        print("\n" + "="*60)
        print("📊 STEP 1: Creating Sample eDNA Dataset")
        print("="*60)
        
        sample_dir = Path("data/sample")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable, "scripts/run_pipeline.py",
            "--create-sample",
            "--input", str(sample_dir),
            "--output", "results/demo"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error creating sample data: {result.stderr}")
            return
        
        print("✅ Sample eDNA dataset created successfully!")
        print(f"📁 Location: {sample_dir / 'sample_edna_sequences.fasta'}")
        
        time.sleep(2)
        
        # Step 2: Run complete analysis
        print("\n" + "="*60)
        print("⚙️  STEP 2: Running Complete eDNA Analysis Pipeline")
        print("="*60)
        
        sample_file = sample_dir / "sample_edna_sequences.fasta"
        output_dir = "results/demo"
        
        cmd = [
            sys.executable, "scripts/run_pipeline.py",
            "--input", str(sample_file),
            "--output", output_dir
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("This may take a few minutes...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error in analysis pipeline: {result.stderr}")
            return
        
        print("✅ Complete analysis pipeline finished successfully!")
        
        # Parse and display results
        results_file = Path(output_dir) / "pipeline_results.json"
        if results_file.exists():
            import json
            with open(results_file) as f:
                results = json.load(f)
            
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"   🧬 Total sequences processed: {results.get('summary', {}).get('total_sequences_processed', 'N/A')}")
            print(f"   🔗 Clusters identified: {results.get('summary', {}).get('total_clusters', 'N/A')}")
            print(f"   🏷️  Taxa identified: {results.get('summary', {}).get('total_taxa_identified', 'N/A')}")
            print(f"   🆕 Novel candidates: {results.get('summary', {}).get('novel_taxa_candidates', 'N/A')}")
            print(f"   📈 Novelty percentage: {results.get('summary', {}).get('novelty_percentage', 'N/A'):.1f}%")
            print(f"   ⏱️  Runtime: {results.get('pipeline_config', {}).get('total_runtime', 'N/A'):.1f} seconds")
        
        time.sleep(2)
        
        # Step 3: Launch dashboard
        print("\n" + "="*60)
        print("🌐 STEP 3: Launching Interactive Dashboard")
        print("="*60)
        
        print("The interactive dashboard will open in your web browser.")
        print("You can explore the results, visualizations, and run additional analyses.")
        print("")
        print("Dashboard features:")
        print("   • Upload and analyze your own eDNA data")
        print("   • Interactive clustering visualizations") 
        print("   • Taxonomic composition analysis")
        print("   • Novel taxa detection results")
        print("   • Export results and generate reports")
        print("")
        
        launch_dashboard = input("Launch dashboard now? (y/n): ").lower().strip()
        
        if launch_dashboard == 'y':
            print("\n🚀 Launching dashboard...")
            print("📱 Opening http://localhost:8501 in your browser")
            print("⌨️  Press Ctrl+C in this terminal to stop the dashboard")
            
            try:
                subprocess.run([sys.executable, "scripts/launch_dashboard.py"])
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped by user")
        
        # Summary
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n📁 All results saved to: {Path(output_dir).absolute()}")
        print(f"📊 Pipeline results: {results_file}")
        print(f"📈 Visualizations: {Path(output_dir) / 'visualizations'}")
        print(f"🔗 Clustering: {Path(output_dir) / 'clustering'}")
        print(f"🏷️  Taxonomy: {Path(output_dir) / 'taxonomy'}")
        print(f"🆕 Novelty: {Path(output_dir) / 'novelty'}")
        
        print(f"\n📚 Next Steps:")
        print(f"   • Read the user guide: docs/user_guide.md")
        print(f"   • Check API documentation: docs/api_reference.md")
        print(f"   • Try the Jupyter demo: notebooks/demo_analysis.py")
        print(f"   • Analyze your own eDNA data using the pipeline")
        
        print(f"\n🧪 For your own data analysis:")
        print(f"   python scripts/run_pipeline.py --input your_sequences.fasta --output results/my_analysis")
        
        print("\n" + "="*60)
        print("Thank you for trying the eDNA Biodiversity Assessment System! 🌊")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check the logs and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()