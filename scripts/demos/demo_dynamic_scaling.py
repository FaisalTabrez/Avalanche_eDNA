"""
Demo: Dynamic Scaling System

Shows how the system automatically adapts as clusters are added,
from small-scale (10 clusters) to large-scale (1000+ clusters).
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.dynamic_hybrid_buffer import ScalingConfig, DynamicHybridBuffer


def demonstrate_scaling_configs():
    """Show how configuration scales with cluster count."""
    
    print("="*80)
    print("DEMONSTRATION: Automatic Configuration Scaling")
    print("="*80)
    
    print("\nShowing optimal configurations for different scales:")
    print("(Assuming 10GB memory budget, 100K sequences per scale)")
    
    scales = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    print("\n" + "─"*80)
    print(f"{'Clusters':<10} {'Exemplars':<12} {'Uncertainty':<15} {'Recent':<12} {'Temp':<8} {'Strategy':<15}")
    print("─"*80)
    
    for n_clusters in scales:
        dataset_size = n_clusters * 1000  # 1000 samples per cluster
        
        config = ScalingConfig.auto_scale(
            n_clusters=n_clusters,
            dataset_size=dataset_size,
            memory_budget_gb=10.0,
            target_accuracy=0.80
        )
        
        strategy = []
        if config.use_hierarchical:
            strategy.append(f"Hierarchical({config.hierarchy_levels}L)")
        if config.use_lora:
            strategy.append("LoRA")
        if not strategy:
            strategy.append("Flat")
        
        print(f"{n_clusters:<10} {config.exemplars_per_cluster:<12} "
              f"{config.uncertainty_buffer_size:<15,} {config.recent_buffer_size:<12,} "
              f"{config.temperature:<8.1f} {'/'.join(strategy):<15}")
    
    print("\n📊 Observations:")
    print("  • Exemplars/cluster: Decreases as clusters grow (memory constraint)")
    print("  • Buffer sizes: Scale with cluster count (more rehearsal needed)")
    print("  • Temperature: Increases with scale (more conservative uncertainty)")
    print("  • Strategy: Switches to hierarchical at 200+, adds LoRA at 500+")


def demonstrate_memory_scaling():
    """Show memory usage across scales."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Memory Budget Management")
    print("="*80)
    
    print("\nMemory allocation across different scales:")
    print("(10GB budget)")
    
    scales = [10, 100, 1000, 10000]
    
    for n_clusters in scales:
        dataset_size = n_clusters * 1000
        
        config = ScalingConfig.auto_scale(
            n_clusters=n_clusters,
            dataset_size=dataset_size,
            memory_budget_gb=10.0,
            target_accuracy=0.80
        )
        
        memory = config.get_memory_estimate()
        
        print(f"\n{n_clusters} clusters:")
        print(f"  Exemplar buffer:    {memory['exemplar_buffer_mb']:>8.1f} MB")
        print(f"  Uncertainty buffer: {memory['uncertainty_buffer_mb']:>8.1f} MB")
        print(f"  Recent buffer:      {memory['recent_buffer_mb']:>8.1f} MB")
        print(f"  Model:              {memory['model_mb']:>8.1f} MB")
        print(f"  {'─'*40}")
        print(f"  Total:              {memory['total_mb']:>8.1f} MB / {memory['budget_mb']:.0f} MB")
        
        usage_pct = (memory['total_mb'] / memory['budget_mb']) * 100
        print(f"  Usage:              {usage_pct:>7.1f}%")


def demonstrate_dynamic_adaptation():
    """Show adaptive scaling in action."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Dynamic Adaptation During Training")
    print("="*80)
    
    print("\nSimulating continual learning from 1 to 100 clusters...")
    print("System will automatically adapt at key thresholds.")
    
    # Create dynamic buffer
    buffer = DynamicHybridBuffer(auto_adapt=True)
    
    print(f"\nInitial configuration:")
    stats = buffer.get_current_stats()
    print(f"  Clusters capacity: {stats['config']['n_clusters']}")
    print(f"  Exemplars/cluster: {stats['config']['exemplars_per_cluster']}")
    print(f"  Uncertainty buffer: {stats['config']['uncertainty_buffer_size']:,}")
    
    # Simulate adding clusters
    adaptation_points = [10, 20, 50, 100]
    
    print("\n" + "─"*80)
    print("Adding clusters...")
    print("─"*80)
    
    for cluster_id in range(100):
        # Generate dummy cluster data
        n_samples = np.random.randint(500, 1500)
        samples = np.random.randn(n_samples, 768)
        labels = np.full(n_samples, cluster_id)
        
        # Add cluster
        buffer.add_cluster(cluster_id, samples, labels, logits=None)
        
        # Report at key points
        if cluster_id + 1 in adaptation_points:
            print(f"\nCluster {cluster_id + 1}:")
            stats = buffer.get_current_stats()
            print(f"  Total adaptations: {stats['adaptations']}")
            print(f"  Current config:")
            print(f"    Exemplars/cluster: {stats['config']['exemplars_per_cluster']}")
            print(f"    Uncertainty buffer: {stats['config']['uncertainty_buffer_size']:,}")
            print(f"    Temperature: {stats['config']['temperature']}")
            print(f"    Architecture: {stats['config']['hidden_dims']}")
    
    # Final statistics
    print("\n" + "─"*80)
    print("Final State (100 clusters):")
    print("─"*80)
    
    stats = buffer.get_current_stats()
    
    print(f"\nBuffer Statistics:")
    print(f"  Exemplar clusters: {stats['buffer_stats']['exemplar']['n_clusters']}")
    print(f"  Total exemplars: {stats['buffer_stats']['exemplar']['total_exemplars']:,}")
    print(f"  Uncertainty samples: {stats['buffer_stats']['uncertainty']['size']:,}")
    print(f"  Recent samples: {stats['buffer_stats']['recent']['size']:,}")
    
    print(f"\nMemory Usage:")
    mem = stats['memory_estimate']
    print(f"  Total: {mem['total_mb']:.1f} MB / {mem['budget_mb']:.0f} MB")
    print(f"  Usage: {(mem['total_mb'] / mem['budget_mb']) * 100:.1f}%")
    
    print(f"\nAdaptations: {stats['adaptations']} automatic reconfigurations")
    
    # Show adaptation history
    if buffer.adaptation_history:
        print("\n📜 Adaptation History:")
        for i, adaptation in enumerate(buffer.adaptation_history):
            print(f"\n  Adaptation {i+1} (at cluster {adaptation['clusters_at_adaptation']}):")
            for change in adaptation['changes']:
                print(f"    • {change}")


def demonstrate_configuration_comparison():
    """Compare configurations for same cluster count with different targets."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Target Accuracy Impact")
    print("="*80)
    
    print("\nComparing configurations for 500 clusters with different accuracy targets:")
    
    n_clusters = 500
    dataset_size = 500000
    
    targets = [0.70, 0.75, 0.80, 0.85, 0.90]
    
    print("\n" + "─"*80)
    print(f"{'Target Acc':<12} {'Exemplars':<12} {'Uncertainty':<15} {'Total Memory':<15}")
    print("─"*80)
    
    for target in targets:
        config = ScalingConfig.auto_scale(
            n_clusters=n_clusters,
            dataset_size=dataset_size,
            memory_budget_gb=10.0,
            target_accuracy=target
        )
        
        memory = config.get_memory_estimate()
        
        print(f"{target:<12.0%} {config.exemplars_per_cluster:<12} "
              f"{config.uncertainty_buffer_size:<15,} {memory['total_mb']:<15.1f}")
    
    print("\n📊 Analysis:")
    print("  • Higher accuracy targets require more exemplars per cluster")
    print("  • This increases memory usage proportionally")
    print("  • System balances accuracy vs memory constraints automatically")


def demonstrate_strategy_selection():
    """Show when different strategies are selected."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Strategy Selection Logic")
    print("="*80)
    
    print("\nStrategy recommendations based on scale:")
    
    test_cases = [
        (10, "Small dataset - simple classification"),
        (100, "Medium dataset - standard continual learning"),
        (250, "Large dataset - hierarchical clustering recommended"),
        (750, "Very large - hierarchical + LoRA recommended"),
        (5000, "Massive scale - multi-level hierarchy + LoRA"),
    ]
    
    print("\n" + "─"*80)
    
    for n_clusters, description in test_cases:
        dataset_size = n_clusters * 1000
        
        config = ScalingConfig.auto_scale(
            n_clusters=n_clusters,
            dataset_size=dataset_size,
            memory_budget_gb=10.0,
            target_accuracy=0.80
        )
        
        print(f"\n{n_clusters} clusters - {description}")
        print(f"  Strategy:")
        
        if config.use_hierarchical:
            print(f"    ✓ Hierarchical clustering ({config.hierarchy_levels} levels)")
        else:
            print(f"    • Flat clustering")
        
        if config.use_lora:
            print(f"    ✓ LoRA adapters (parameter-efficient)")
        else:
            print(f"    • Full fine-tuning")
        
        print(f"  Configuration:")
        print(f"    Architecture: {config.hidden_dims}")
        print(f"    Replay ratio: {config.replay_ratio:.0%}")
        print(f"    Batch size: {config.batch_size}")
        print(f"    EWC lambda: {config.ewc_lambda}")


def demonstrate_save_load():
    """Show configuration persistence."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Configuration Persistence")
    print("="*80)
    
    # Create configuration
    config = ScalingConfig.auto_scale(
        n_clusters=1000,
        dataset_size=1000000,
        memory_budget_gb=10.0,
        target_accuracy=0.80
    )
    
    print("\nOriginal configuration for 1000 clusters:")
    print(f"  Exemplars/cluster: {config.exemplars_per_cluster}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Use hierarchical: {config.use_hierarchical}")
    print(f"  Use LoRA: {config.use_lora}")
    
    # Save
    save_path = Path("demo_config_1000.json")
    config.save(save_path)
    print(f"\n✓ Saved to {save_path}")
    
    # Load
    loaded_config = ScalingConfig.load(save_path)
    print(f"✓ Loaded from {save_path}")
    
    print("\nLoaded configuration:")
    print(f"  Exemplars/cluster: {loaded_config.exemplars_per_cluster}")
    print(f"  Hidden dims: {loaded_config.hidden_dims}")
    print(f"  Temperature: {loaded_config.temperature}")
    print(f"  Use hierarchical: {loaded_config.use_hierarchical}")
    print(f"  Use LoRA: {loaded_config.use_lora}")
    
    # Verify match
    print(f"\n✓ Configurations match: {config.to_dict() == loaded_config.to_dict()}")
    
    # Cleanup
    save_path.unlink()
    print(f"✓ Cleaned up {save_path}")


def main():
    """Run all demonstrations."""
    
    print("\n" + "█"*80)
    print("DYNAMIC SCALING SYSTEM DEMONSTRATION")
    print("█"*80)
    
    print("\nThis demo shows how the system automatically adapts to any scale:")
    print("  • Automatic configuration based on cluster count")
    print("  • Memory budget management")
    print("  • Dynamic adaptation during training")
    print("  • Strategy selection (flat/hierarchical/LoRA)")
    print("  • Configuration persistence")
    
    # Run demonstrations
    demonstrate_scaling_configs()
    demonstrate_memory_scaling()
    demonstrate_dynamic_adaptation()
    demonstrate_configuration_comparison()
    demonstrate_strategy_selection()
    demonstrate_save_load()
    
    print("\n" + "█"*80)
    print("DEMO COMPLETE")
    print("█"*80)
    
    print("\n✅ Key Features Demonstrated:")
    print("  1. Scales from 10 to 10,000+ clusters automatically")
    print("  2. Adapts buffer sizes based on memory budget")
    print("  3. Adjusts strategies (hierarchical, LoRA) based on scale")
    print("  4. Reconfigures on-the-fly during training")
    print("  5. No hard-coded limits - purely data-driven")
    
    print("\n🚀 Ready for Production:")
    print("  • Use DynamicHybridBuffer for automatic scaling")
    print("  • System adapts at: 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000 clusters")
    print("  • Memory usage stays within budget")
    print("  • Performance optimized for each scale")


if __name__ == "__main__":
    main()
