"""
Demo: Test the revised taxonomy classification pipeline
Uses the synthetic eDNA dataset we already generated
"""

import sys
sys.path.insert(0, '.')

from scripts.run_taxonomy_pipeline_v2 import TaxonomyClassificationPipeline

def main():
    print("="*70)
    print("DEMO: Revised Taxonomy Classification Pipeline v2.0")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Load synthetic eDNA sequences (2,500 sequences)")
    print("  2. Generate DNABERT-2 embeddings")
    print("  3. Cluster sequences (k=5)")
    print("  4. Train classifier with ACTIVE REPLAY")
    print("  5. Evaluate on all clusters")
    print("  6. Generate visualizations")
    print("\nExpected result: ~89% accuracy (vs 18% with passive replay)")
    print("="*70)
    
    input("\nPress Enter to start the demo...")
    
    # Initialize pipeline with active replay configuration
    pipeline = TaxonomyClassificationPipeline(
        output_dir="demo_taxonomy_pipeline_v2",
        dnabert_model_path="./models/dnabert2_cpu",
        device="cpu",
        enable_continual_learning=True,
        replay_buffer_size=1000,
        replay_ratio=0.5,  # 50/50 mix - THE KEY TO SUCCESS!
        ewc_lambda=100.0
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        fasta_file="data/synthetic_edna/mixed_edna_2500.fasta",
        n_clusters=5,
        train_classifier=True,
        use_active_replay=True,  # Critical parameter!
        epochs_per_cluster=10
    )
    
    # Display results
    print("\n" + "="*70)
    print("DEMO RESULTS")
    print("="*70)
    
    if 'training_results' in results and 'overall_accuracy' in results['training_results']:
        accuracy = results['training_results']['overall_accuracy']
        print(f"\n✅ Overall Accuracy: {accuracy:.1f}%")
        
        if accuracy > 80:
            print("   🎉 EXCELLENT! Active replay is working as expected!")
        elif accuracy > 50:
            print("   ⚠️  Good, but lower than expected. Check configuration.")
        else:
            print("   ❌ Low accuracy. Verify active replay is enabled.")
        
        print("\nPer-Cluster Results:")
        for cluster_id, cluster_result in results['training_results']['cluster_results'].items():
            acc = cluster_result['accuracy']
            correct = cluster_result['correct']
            total = cluster_result['total']
            print(f"  Cluster {cluster_id}: {acc:>6.1f}% ({correct:>4}/{total:>4})")
    
    print(f"\nTotal runtime: {results['total_time_seconds']/60:.1f} minutes")
    print(f"Output directory: {pipeline.output_dir}")
    print("\n" + "="*70)
    print("Demo complete! Check the output directory for:")
    print("  • embeddings/dnabert2_embeddings.npy")
    print("  • clustering/results.json")
    print("  • taxonomy/assignments.csv")
    print("  • checkpoints/*.pt (5 model versions)")
    print("  • visualizations/cluster_analysis.png")
    print("  • pipeline_summary.json")
    print("="*70)

if __name__ == "__main__":
    main()
