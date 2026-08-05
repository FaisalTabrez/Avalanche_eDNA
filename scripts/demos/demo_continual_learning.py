"""
Continual Learning Simulation Demo
Demonstrates checkpoint management, model registry, and continual learning strategies
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import continual learning components
from src.models.checkpoint_manager import CheckpointManager
from src.models.continual_learning import ContinualLearner, ExperienceReplayBuffer
from src.models.model_registry import ModelRegistry

print("=" * 80)
print("CONTINUAL LEARNING SIMULATION - DNABERT-2 eDNA Analysis")
print("=" * 80)
print()

# ============================================================================
# 1. SETUP: Create Demo Model and Directories
# ============================================================================
print("📁 Step 1: Setting up demo environment...")

# Create output directories
demo_dir = Path("demo_outputs")
checkpoint_dir = demo_dir / "checkpoints"
registry_dir = demo_dir / "registry"

for dir_path in [demo_dir, checkpoint_dir, registry_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Created directory: {dir_path}")


# Create a simple DNA sequence classifier model
class DNASequenceClassifier(nn.Module):
    """Simple neural network for DNA sequence classification"""

    def __init__(self, input_dim=100, hidden_dim=64, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


model = DNASequenceClassifier()
print(
    f"   ✓ Created model with {sum(p.numel() for p in model.parameters()):,} parameters"
)
print()

# ============================================================================
# 2. CHECKPOINT MANAGER: Save and Load Training States
# ============================================================================
print("💾 Step 2: Testing Checkpoint Manager...")

checkpoint_manager = CheckpointManager(
    checkpoint_dir=str(checkpoint_dir), max_checkpoints=3, keep_best=True
)
print(f"   ✓ Initialized CheckpointManager at {checkpoint_dir}")

# Simulate training on Dataset 1: Marine bacteria
print("\n   📊 Simulating training on Dataset 1: Marine bacteria...")
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    # Simulate training
    loss = 0.5 - (epoch * 0.1)  # Decreasing loss
    accuracy = 0.7 + (epoch * 0.05)  # Increasing accuracy

    # Save checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={"loss": loss, "accuracy": accuracy},
        dataset_info={"name": "marine_bacteria", "samples": 1000},
    )

    print(
        f"      Epoch {epoch}: loss={loss:.3f}, acc={accuracy:.3f} → Saved to {Path(checkpoint_path).name}"
    )

# Get checkpoint history
history = checkpoint_manager.get_checkpoint_history()
print(f"\n   ✓ Total checkpoints saved: {len(history)}")

# Get best checkpoint
best = checkpoint_manager.get_best_checkpoint_info(metric="loss", minimize=True)
if best:
    print(
        f"   ✓ Best checkpoint: epoch {best['epoch']}, loss={best['metrics']['loss']:.3f}"
    )
print()

# ============================================================================
# 3. MODEL REGISTRY: Track Model Versions and Lineage
# ============================================================================
print("📋 Step 3: Testing Model Registry...")

registry = ModelRegistry(registry_dir=str(registry_dir), backend="json")
print(f"   ✓ Initialized ModelRegistry at {registry_dir}")

# Register initial model
v1_path = demo_dir / "model_v1.0.pt"
torch.save(model.state_dict(), v1_path)

registry.register_model(
    version="v1.0",
    model_path=str(v1_path),
    datasets=["marine_bacteria"],
    metrics={"loss": 0.3, "accuracy": 0.80},
    description="Initial model trained on marine bacteria",
)
print("   ✓ Registered model v1.0")

# Simulate fine-tuning and register new version
model_v2 = DNASequenceClassifier()
v2_path = demo_dir / "model_v2.0.pt"
torch.save(model_v2.state_dict(), v2_path)

registry.register_model(
    version="v2.0",
    model_path=str(v2_path),
    datasets=["marine_bacteria", "freshwater_algae"],
    metrics={"loss": 0.25, "accuracy": 0.85},
    parent_version="v1.0",
    description="Fine-tuned on marine bacteria + freshwater algae",
)
print("   ✓ Registered model v2.0 (child of v1.0)")

# List all models
models = registry.list_models()
print(f"\n   📊 Registered models: {len(models)}")
for model_info in models:
    print(
        f"      - {model_info['version']}: {model_info.get('description', 'No description')}"
    )

# Get lineage
lineage = registry.get_lineage("v2.0")
lineage_versions = [v["version"] if isinstance(v, dict) else v for v in lineage]
print(f"\n   🌳 Model v2.0 lineage: {' → '.join(lineage_versions)}")
print()

# ============================================================================
# 4. EXPERIENCE REPLAY BUFFER: Store and Sample Past Data
# ============================================================================
print("🔄 Step 4: Testing Experience Replay Buffer...")

buffer = ExperienceReplayBuffer(max_size=100)
print(f"   ✓ Initialized buffer with max_size=100")

# Add samples from Dataset 1
sequences_d1 = [f"MARINE_SEQ_{i:03d}" for i in range(60)]
labels_d1 = [i % 5 for i in range(60)]

buffer.add_samples(sequences_d1, labels_d1)
print(f"   ✓ Added 60 samples from Dataset 1 (marine)")

# Add samples from Dataset 2 (will trigger reservoir sampling)
sequences_d2 = [f"FRESH_SEQ_{i:03d}" for i in range(80)]
labels_d2 = [(i + 2) % 5 for i in range(80)]

buffer.add_samples(sequences_d2, labels_d2)
print(f"   ✓ Added 80 samples from Dataset 2 (freshwater)")
print(f"   ✓ Buffer size capped at: {len(buffer)} (max_size enforced)")

# Sample from buffer
sampled_seqs, sampled_labels = buffer.sample(batch_size=10)
print(f"\n   📦 Sampled batch:")
print(f"      - Batch size: {len(sampled_seqs)}")
print(f"      - Example sequences: {sampled_seqs[:3]}")
print(f"      - Example labels: {sampled_labels[:3]}")
print()

# ============================================================================
# 5. CONTINUAL LEARNING: Anti-Forgetting Strategies
# ============================================================================
print("🧠 Step 5: Testing Continual Learning Strategies...")

# Test Experience Replay strategy
print("\n   A) Experience Replay Strategy:")
learner_replay = ContinualLearner(strategy="experience_replay", buffer_size=1000)
print("      ✓ Initialized with experience_replay strategy")

# Store samples
learner_replay.store_samples(
    sequences=["ATCGATCG", "GCTAGCTA", "TTAATTAA"], labels=[0, 1, 2]
)
print(f"      ✓ Stored 3 samples in replay buffer")

# Get replay samples
replay_seqs, replay_labels = learner_replay.get_replay_samples(batch_size=2)
print(f"      ✓ Sampled {len(replay_seqs)} sequences for replay")

# Test EWC strategy
print("\n   B) Elastic Weight Consolidation (EWC) Strategy:")
learner_ewc = ContinualLearner(strategy="ewc", ewc_lambda=0.5)
print("      ✓ Initialized with EWC strategy (λ=0.5)")

# Create dummy dataloader for Fisher computation
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(50, 100)  # 50 samples, 100 features
y = torch.randint(0, 5, (50,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)


# Wrap dataloader to return dict format
class DictDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch_x, batch_y in self.dataloader:
            yield {"input_ids": batch_x, "labels": batch_y}


dict_dataloader = DictDataLoader(dataloader)


# Create model wrapper
class ModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, input_ids, **kwargs):
        return self.base(input_ids)


wrapped_model = ModelWrapper(model)

# Compute Fisher information
print("      ⏳ Computing Fisher Information Matrix...")
learner_ewc.compute_fisher_information(
    model=wrapped_model, dataloader=dict_dataloader, device="cpu"
)
print(
    f"      ✓ Computed Fisher information for {len(learner_ewc.fisher_dict)} parameters"
)

# Test Combined strategy
print("\n   C) Combined Strategy (Replay + EWC + LwF):")
learner_combined = ContinualLearner(
    strategy="combined", buffer_size=500, ewc_lambda=0.4
)
print("      ✓ Initialized with combined strategy")
print("      ✓ Uses: Experience Replay + EWC + Learning without Forgetting")

print()

# ============================================================================
# 6. INTEGRATION TEST: Complete Workflow
# ============================================================================
print("🔗 Step 6: Integration Test - Complete Workflow...")
print()

# Create new model for sequential training
integration_model = DNASequenceClassifier()
print("   📝 Workflow: Train on Dataset 1 → Dataset 2 → Dataset 3")
print()

datasets = [
    {"name": "marine_bacteria", "samples": 100, "color": "🌊"},
    {"name": "freshwater_algae", "samples": 150, "color": "🌿"},
    {"name": "soil_fungi", "samples": 120, "color": "🍄"},
]

integration_learner = ContinualLearner(strategy="combined", buffer_size=500)
integration_optimizer = optim.Adam(integration_model.parameters())

for idx, dataset in enumerate(datasets, 1):
    print(f"   {dataset['color']} Training on Dataset {idx}: {dataset['name']}")

    # Simulate training
    loss = 0.5 - (idx * 0.08)
    accuracy = 0.75 + (idx * 0.04)

    # Store samples for replay
    sample_seqs = [f"{dataset['name'].upper()}_SEQ_{i:03d}" for i in range(20)]
    sample_labels = [i % 5 for i in range(20)]
    integration_learner.store_samples(sample_seqs, sample_labels)

    # Save checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=integration_model,
        optimizer=integration_optimizer,
        epoch=idx,
        metrics={"loss": loss, "accuracy": accuracy},
        dataset_info=dataset,
    )

    # Register in registry
    version = f"v1.{idx}"
    model_path = demo_dir / f"model_{version}.pt"
    torch.save(integration_model.state_dict(), model_path)

    parent = f"v1.{idx-1}" if idx > 1 else None
    registry.register_model(
        version=version,
        model_path=str(model_path),
        datasets=[d["name"] for d in datasets[:idx]],
        metrics={"loss": loss, "accuracy": accuracy},
        parent_version=parent,
    )

    print(f"      ✓ Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
    print(f"      ✓ Checkpoint saved, model registered as {version}")
    print(f"      ✓ Replay buffer size: {len(integration_learner.replay_buffer)}")
    print()

# Final summary
print("=" * 80)
print("📊 SIMULATION SUMMARY")
print("=" * 80)
print()

print("✅ Checkpoint Manager:")
history = checkpoint_manager.get_checkpoint_history()
print(f"   - Total checkpoints: {len(history)}")
print(
    f"   - Best checkpoint: epoch {checkpoint_manager.get_best_checkpoint_info('loss', minimize=True)['epoch']}"
)
print()

print("✅ Model Registry:")
all_models = registry.list_models()
print(f"   - Total models: {len(all_models)}")
print(f"   - Latest version: {all_models[-1]['version']}")
latest_lineage = registry.get_lineage(all_models[-1]["version"])
latest_lineage_versions = [
    v["version"] if isinstance(v, dict) else v for v in latest_lineage
]
print(f"   - Training lineage: {' → '.join(latest_lineage_versions)}")
print()

print("✅ Continual Learning:")
print(
    f"   - Replay buffer capacity: {len(integration_learner.replay_buffer)}/{integration_learner.buffer_size}"
)
print(f"   - Strategy: {integration_learner.strategy}")
print(f"   - Datasets trained: {len(datasets)}")
print()

print("=" * 80)
print("🎉 SIMULATION COMPLETED SUCCESSFULLY!")
print("=" * 80)
print()

print("📁 Output files created in:", demo_dir)
print("   - Checkpoints:", checkpoint_dir)
print("   - Model registry:", registry_dir)
print()

# Save summary
summary = {
    "simulation_date": "2025-11-26",
    "total_checkpoints": len(history),
    "total_models": len(all_models),
    "datasets_trained": [d["name"] for d in datasets],
    "final_metrics": all_models[-1]["metrics"],
    "replay_buffer_size": len(integration_learner.replay_buffer),
    "strategy_used": integration_learner.strategy,
}

summary_path = demo_dir / "simulation_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"💾 Simulation summary saved to: {summary_path}")
print()
print("✨ All continual learning features validated!")
