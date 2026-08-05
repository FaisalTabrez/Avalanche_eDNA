"""
Test Suite for Continual Learning Features
Tests checkpoint management, fine-tuning, model registry, and continual learning strategies
"""

import json
import shutil

# Import continual learning components
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.checkpoint_manager import CheckpointManager
from src.models.continual_learning import ContinualLearner, ExperienceReplayBuffer
from src.models.finetuner import ContrastiveLearningHead, DNABERTFineTuner
from src.models.model_registry import ModelRegistry


class TestCheckpointManager:
    """Test checkpoint saving, loading, and management"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager instance"""
        return CheckpointManager(checkpoint_dir=temp_dir)

    def test_checkpoint_initialization(self, checkpoint_manager, temp_dir):
        """Test checkpoint manager initialization"""
        assert checkpoint_manager.checkpoint_dir == Path(temp_dir)
        assert checkpoint_manager.checkpoint_dir.exists()

    def test_save_checkpoint(self, checkpoint_manager):
        """Test saving a checkpoint"""
        # Create dummy model and optimizer (API takes objects, not state dicts)
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        metrics = {"loss": 0.5, "accuracy": 0.85}

        # Save checkpoint (uses 'model' and 'optimizer' parameters, not 'model_state' and 'optimizer_state')
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics=metrics,
            model_config={"batch_size": 32},
        )

        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()
        assert "epoch5" in str(checkpoint_path) or "checkpoint" in str(checkpoint_path)

    def test_load_checkpoint(self, checkpoint_manager):
        """Test loading a checkpoint"""
        # Create and save a checkpoint first
        import torch.nn as nn

        model = nn.Linear(5, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model, optimizer=optimizer, epoch=3, metrics={"loss": 0.3}
        )

        # Load checkpoint
        loaded_data = checkpoint_manager.load_checkpoint(checkpoint_path)

        assert loaded_data is not None
        assert "model_state_dict" in loaded_data
        assert "optimizer_state_dict" in loaded_data
        assert "epoch" in loaded_data
        assert loaded_data["epoch"] == 3

    def test_checkpoint_history(self, checkpoint_manager):
        """Test retrieving checkpoint history"""
        # Save multiple checkpoints
        import torch.nn as nn

        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(1, 4):
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"loss": 1.0 / epoch},
            )

        # Get history
        history = checkpoint_manager.get_checkpoint_history()

        assert len(history) == 3
        assert all("epoch" in ckpt for ckpt in history)
        assert all("timestamp" in ckpt for ckpt in history)

    def test_best_checkpoint(self, checkpoint_manager):
        """Test getting best checkpoint by metric"""
        # Save checkpoints with different losses
        import torch.nn as nn

        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())

        for i, loss in enumerate([0.5, 0.3, 0.7]):
            checkpoint_manager.save_checkpoint(
                model=model, optimizer=optimizer, epoch=i, metrics={"val_loss": loss}
            )

        # Get best checkpoint info
        best = checkpoint_manager.get_best_checkpoint_info(
            metric="val_loss", minimize=True
        )

        assert best is not None
        assert best["metrics"]["val_loss"] == 0.3

    def test_cleanup_old_checkpoints(self, temp_dir):
        """Test automatic cleanup of old checkpoints"""
        # Create manager with max_checkpoints=5
        manager = CheckpointManager(checkpoint_dir=temp_dir, max_checkpoints=5)

        import torch.nn as nn

        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Create many checkpoints (should auto-cleanup)
        for epoch in range(10):
            manager.save_checkpoint(
                model=model, optimizer=optimizer, epoch=epoch, metrics={"loss": 0.1}
            )

        # Check that only max_checkpoints are kept
        history = manager.get_checkpoint_history()
        assert len(history) <= 5


class TestDNABERTFineTuner:
    """Test DNABERT fine-tuning functionality"""

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.mark.skip(reason="Requires DNABERT-2 model download and triton dependency")
    def test_finetuner_initialization(self, temp_dir):
        """Test fine-tuner initialization"""
        # Note: API uses 'model_id' not 'model_name', and doesn't take 'output_dir' in __init__
        finetuner = DNABERTFineTuner(
            model_id="zhihan1996/DNABERT-2-117M", freeze_embeddings=True, device="cpu"
        )

        assert finetuner.model_id == "zhihan1996/DNABERT-2-117M"
        assert finetuner.device == "cpu"
        assert finetuner.freeze_embeddings == True

    @pytest.mark.skip(reason="Requires DNABERT-2 model download and triton dependency")
    def test_freeze_layers(self, temp_dir):
        """Test layer freezing strategies"""
        # Initialize with specific freeze configuration
        finetuner = DNABERTFineTuner(
            model_id="zhihan1996/DNABERT-2-117M",
            freeze_layers=[0, 1, 2],  # Freeze first 3 layers
            freeze_embeddings=True,
            device="cpu",
        )

        # Test different freeze strategy helpers
        strategies = ["all", "none", "half", "gradual", "top3"]

        for strategy in strategies:
            try:
                freeze_config = finetuner.get_freezing_strategy(strategy=strategy)
                # Verify it returns a list
                assert isinstance(freeze_config, list)
            except Exception as e:
                pytest.fail(f"Freeze strategy '{strategy}' failed: {e}")

    def test_contrastive_head(self):
        """Test contrastive learning head"""
        input_dim = 768
        projection_dim = 128

        head = ContrastiveLearningHead(
            input_dim=input_dim, projection_dim=projection_dim
        )

        # Test forward pass
        batch_size = 4
        embeddings = torch.randn(batch_size, input_dim)
        projections = head(embeddings)

        assert projections.shape == (batch_size, projection_dim)

        # Test contrastive loss (method is 'contrastive_loss', not 'nt_xent_loss')
        z1 = head(torch.randn(batch_size, input_dim))
        z2 = head(torch.randn(batch_size, input_dim))
        loss = head.contrastive_loss(z1, z2)
        assert loss.item() >= 0


class TestContinualLearner:
    """Test continual learning strategies"""

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        return model

    def test_continual_learner_initialization(self, dummy_model, temp_dir):
        """Test continual learner initialization"""
        # Note: ContinualLearner does NOT take 'model' or 'memory_dir' in __init__
        learner = ContinualLearner(
            strategy="experience_replay", buffer_size=5000, ewc_lambda=0.4
        )

        assert learner.strategy == "experience_replay"
        assert learner.buffer_size == 5000
        assert learner.replay_buffer is not None

    def test_experience_replay_buffer(self):
        """Test experience replay buffer"""
        # Note: ExperienceReplayBuffer uses 'max_size' parameter, not 'buffer_size'
        max_size = 100
        buffer = ExperienceReplayBuffer(max_size=max_size)

        # Add samples (uses add_samples with sequences and labels)
        sequences = [f'ATCG{"N" * i}' for i in range(150)]
        labels = [i % 5 for i in range(150)]

        buffer.add_samples(sequences, labels)

        # Check buffer size (should be capped at max_size)
        assert len(buffer) == max_size

        # Sample from buffer
        sampled_seqs, sampled_labels = buffer.sample(batch_size=10)
        assert len(sampled_seqs) == 10
        assert len(sampled_labels) == 10

    def test_ewc_fisher_computation(self, dummy_model, temp_dir):
        """Test EWC Fisher information computation"""
        # ContinualLearner doesn't take model in __init__
        learner = ContinualLearner(strategy="ewc", ewc_lambda=0.5)

        # Create dummy dataloader
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.randn(50, 10)
        y = torch.randint(0, 5, (50,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10)

        # Create a model wrapper that mimics BERT-style output
        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model

            def forward(self, input_ids, **kwargs):
                # Process input_ids and return logits
                logits = self.base(input_ids)
                return logits

        wrapped_model = ModelWrapper(dummy_model)

        # Modify dataloader to return dict format expected by compute_fisher_information
        class DictDataLoader:
            def __init__(self, dataloader):
                self.dataloader = dataloader

            def __iter__(self):
                for batch_x, batch_y in self.dataloader:
                    yield {"input_ids": batch_x, "labels": batch_y}

        dict_dataloader = DictDataLoader(dataloader)

        # Compute Fisher information (pass model as parameter)
        learner.compute_fisher_information(
            model=wrapped_model, dataloader=dict_dataloader, device="cpu"
        )

        assert learner.fisher_dict is not None
        assert len(learner.fisher_dict) > 0

    def test_save_load_task_memory(self, dummy_model, temp_dir):
        """Test storing samples in continual learner"""
        learner = ContinualLearner(strategy="experience_replay", buffer_size=1000)

        # Store samples
        sequences = ["ATCGATCG", "GCTAGCTA", "TTAATTAA"]
        labels = [0, 1, 2]
        learner.store_samples(sequences, labels)

        # Verify samples were stored
        assert len(learner.replay_buffer) == 3

        # Get replay samples
        replay_seqs, replay_labels = learner.get_replay_samples(batch_size=2)
        assert len(replay_seqs) == 2
        assert len(replay_labels) == 2


class TestModelRegistry:
    """Test model registry and versioning"""

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def registry(self, temp_dir):
        """Create model registry instance"""
        return ModelRegistry(registry_dir=temp_dir, backend="json")

    def test_registry_initialization(self, registry, temp_dir):
        """Test registry initialization"""
        assert registry.registry_dir == Path(temp_dir)
        assert registry.backend == "json"

    def test_register_model(self, registry, temp_dir):
        """Test registering a new model"""
        version = registry.register_model(
            version="v1.0",
            model_path=str(Path(temp_dir) / "model.pt"),
            base_model="zhihan1996/DNABERT-2-117M",
            datasets=["marine_edna"],
            metrics={"val_loss": 0.25, "val_accuracy": 0.89},
            description="First model on marine dataset",
        )

        assert version == "v1.0"

        # Retrieve model
        model_info = registry.get_model("v1.0")
        assert model_info is not None
        assert model_info["version"] == "v1.0"
        assert "marine_edna" in model_info["datasets"]
        assert model_info["metrics"]["val_loss"] == 0.25

    def test_list_models(self, registry, temp_dir):
        """Test listing all models"""
        # Register multiple models
        for i in range(3):
            registry.register_model(
                version=f"v1.{i}",
                model_path=str(Path(temp_dir) / f"model_{i}.pt"),
                datasets=[f"dataset_{i}"],
                metrics={"loss": 0.1 * (i + 1)},
            )

        # List all models
        models = registry.list_models()
        assert len(models) == 3
        assert all("version" in m for m in models)

    def test_model_lineage(self, registry, temp_dir):
        """Test model lineage tracking"""
        # Create parent-child relationship
        registry.register_model(
            version="v1.0",
            model_path=str(Path(temp_dir) / "model_v1.pt"),
            datasets=["dataset1"],
            metrics={"loss": 0.5},
        )

        registry.register_model(
            version="v1.1",
            model_path=str(Path(temp_dir) / "model_v1.1.pt"),
            parent_version="v1.0",
            datasets=["dataset2"],
            metrics={"loss": 0.3},
        )

        # Get lineage
        lineage = registry.get_lineage("v1.1")
        assert len(lineage) == 2
        assert lineage[0]["version"] == "v1.0"
        assert lineage[1]["version"] == "v1.1"

        # Get children
        children = registry.get_children("v1.0")
        assert len(children) == 1
        assert children[0]["version"] == "v1.1"

    def test_compare_models(self, registry, temp_dir):
        """Test model comparison"""
        # Register two models
        registry.register_model(
            version="v1.0",
            model_path=str(Path(temp_dir) / "model1.pt"),
            datasets=["dataset1"],
            metrics={"val_loss": 0.5, "val_accuracy": 0.8},
        )

        registry.register_model(
            version="v2.0",
            model_path=str(Path(temp_dir) / "model2.pt"),
            datasets=["dataset1", "dataset2"],
            metrics={"val_loss": 0.3, "val_accuracy": 0.9},
        )

        # Compare models
        comparison = registry.compare_models("v1.0", "v2.0")

        assert comparison is not None
        assert "metric_differences" in comparison
        assert "dataset_differences" in comparison

        # Check metric differences
        assert "val_loss" in comparison["metric_differences"]
        assert comparison["metric_differences"]["val_loss"]["change"] < 0  # Improved

    def test_get_best_model(self, registry, temp_dir):
        """Test getting best model by metric"""
        # Register models with different performance
        for i, loss in enumerate([0.5, 0.2, 0.8]):
            registry.register_model(
                version=f"v1.{i}",
                model_path=str(Path(temp_dir) / f"model_{i}.pt"),
                datasets=["test"],
                metrics={"val_loss": loss},
            )

        # Get best model
        best = registry.get_best_model(metric="val_loss", minimize=True)

        assert best is not None
        assert best["metrics"]["val_loss"] == 0.2
        assert best["version"] == "v1.1"

    def test_update_model_status(self, registry, temp_dir):
        """Test updating model status"""
        # Register a model
        registry.register_model(
            version="v1.0",
            model_path=str(Path(temp_dir) / "model.pt"),
            datasets=["test"],
            metrics={"loss": 0.3},
        )

        # Update status
        registry.update_model_status("v1.0", "archived")

        # Verify update
        model_info = registry.get_model("v1.0")
        assert model_info["status"] == "archived"


class TestIntegration:
    """Integration tests for complete continual learning workflow"""

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_end_to_end_workflow(self, temp_dir):
        """Test complete continual learning workflow"""
        # 1. Initialize components
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(Path(temp_dir) / "checkpoints")
        )

        registry = ModelRegistry(
            registry_dir=str(Path(temp_dir) / "registry"), backend="json"
        )

        # 2. Create dummy model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )

        # 3. Save checkpoint (using model object, not model.state_dict())
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            epoch=5,
            metrics={"val_loss": 0.25, "val_accuracy": 0.85},
        )

        assert checkpoint_path is not None

        # 4. Register model in registry
        version = registry.register_model(
            version="v1.0",
            model_path=str(Path(temp_dir) / "model.pt"),
            checkpoint_path=checkpoint_path,
            datasets=["marine_dataset"],
            metrics={"val_loss": 0.25, "val_accuracy": 0.85},
        )

        assert version == "v1.0"

        # 5. Load checkpoint and verify
        loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
        assert loaded_checkpoint is not None
        assert loaded_checkpoint["epoch"] == 5

        # 6. Get model from registry and verify
        model_info = registry.get_model("v1.0")
        assert model_info is not None
        assert model_info["checkpoint_path"] == checkpoint_path
        assert "marine_dataset" in model_info["datasets"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
