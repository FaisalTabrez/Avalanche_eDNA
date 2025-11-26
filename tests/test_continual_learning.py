"""
Test Suite for Continual Learning Features
Tests checkpoint management, fine-tuning, model registry, and continual learning strategies
"""

import pytest
import torch
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil

# Import continual learning components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.checkpoint_manager import CheckpointManager
from src.models.finetuner import DNABERTFineTuner, ContrastiveLearningHead
from src.models.continual_learning import ContinualLearner, ExperienceReplayBuffer
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
        # Create dummy model state
        model_state = {
            'layer1.weight': torch.randn(10, 10),
            'layer1.bias': torch.randn(10)
        }
        
        optimizer_state = {'lr': 0.001, 'momentum': 0.9}
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            epoch=5,
            metrics=metrics,
            config={'batch_size': 32}
        )
        
        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()
        assert 'epoch_5' in str(checkpoint_path)
    
    def test_load_checkpoint(self, checkpoint_manager):
        """Test loading a checkpoint"""
        # Save a checkpoint first
        model_state = {'param': torch.randn(5, 5)}
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=None,
            epoch=3,
            metrics={'loss': 0.3}
        )
        
        # Load checkpoint
        loaded_data = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        assert loaded_data is not None
        assert 'model_state_dict' in loaded_data
        assert 'epoch' in loaded_data
        assert loaded_data['epoch'] == 3
        assert torch.allclose(
            loaded_data['model_state_dict']['param'],
            model_state['param']
        )
    
    def test_checkpoint_history(self, checkpoint_manager):
        """Test retrieving checkpoint history"""
        # Save multiple checkpoints
        for epoch in range(1, 4):
            checkpoint_manager.save_checkpoint(
                model_state={'param': torch.randn(2, 2)},
                optimizer_state=None,
                epoch=epoch,
                metrics={'loss': 1.0 / epoch}
            )
        
        # Get history
        history = checkpoint_manager.get_checkpoint_history()
        
        assert len(history) == 3
        assert all('epoch' in ckpt for ckpt in history)
        assert all('timestamp' in ckpt for ckpt in history)
    
    def test_best_checkpoint(self, checkpoint_manager):
        """Test getting best checkpoint by metric"""
        # Save checkpoints with different losses
        for i, loss in enumerate([0.5, 0.3, 0.7]):
            checkpoint_manager.save_checkpoint(
                model_state={'param': torch.randn(2, 2)},
                optimizer_state=None,
                epoch=i,
                metrics={'val_loss': loss}
            )
        
        # Get best checkpoint
        best = checkpoint_manager.get_best_checkpoint(metric='val_loss')
        
        assert best is not None
        assert best['metrics']['val_loss'] == 0.3
    
    def test_cleanup_old_checkpoints(self, checkpoint_manager):
        """Test automatic cleanup of old checkpoints"""
        # Create many checkpoints
        for epoch in range(15):
            checkpoint_manager.save_checkpoint(
                model_state={'param': torch.randn(2, 2)},
                optimizer_state=None,
                epoch=epoch,
                metrics={'loss': 0.1}
            )
        
        # Cleanup (max_checkpoints default is 10)
        checkpoint_manager.cleanup_old_checkpoints(max_keep=5)
        
        history = checkpoint_manager.get_checkpoint_history()
        assert len(history) <= 5


class TestDNABERTFineTuner:
    """Test DNABERT fine-tuning functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_finetuner_initialization(self, temp_dir):
        """Test fine-tuner initialization"""
        finetuner = DNABERTFineTuner(
            model_name='zhihan1996/DNABERT-2-117M',
            output_dir=temp_dir,
            device='cpu'
        )
        
        assert finetuner.model_name == 'zhihan1996/DNABERT-2-117M'
        assert finetuner.device == 'cpu'
        assert finetuner.output_dir == Path(temp_dir)
    
    def test_freeze_layers(self, temp_dir):
        """Test layer freezing strategies"""
        finetuner = DNABERTFineTuner(
            model_name='zhihan1996/DNABERT-2-117M',
            output_dir=temp_dir,
            device='cpu'
        )
        
        # Test different freeze strategies
        strategies = ['all', 'none', 'half']
        
        for strategy in strategies:
            try:
                finetuner.freeze_layers(strategy=strategy)
                # Verify freezing worked (at least it doesn't crash)
                assert True
            except Exception as e:
                pytest.fail(f"Freeze strategy '{strategy}' failed: {e}")
    
    def test_contrastive_head(self):
        """Test contrastive learning head"""
        input_dim = 768
        projection_dim = 128
        
        head = ContrastiveLearningHead(
            input_dim=input_dim,
            projection_dim=projection_dim
        )
        
        # Test forward pass
        batch_size = 4
        embeddings = torch.randn(batch_size, input_dim)
        projections = head(embeddings)
        
        assert projections.shape == (batch_size, projection_dim)
        
        # Test NT-Xent loss
        loss = head.nt_xent_loss(projections, temperature=0.07)
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
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        return model
    
    def test_continual_learner_initialization(self, dummy_model, temp_dir):
        """Test continual learner initialization"""
        learner = ContinualLearner(
            model=dummy_model,
            strategy='experience_replay',
            memory_dir=temp_dir
        )
        
        assert learner.strategy == 'experience_replay'
        assert learner.model is not None
    
    def test_experience_replay_buffer(self):
        """Test experience replay buffer"""
        buffer_size = 100
        buffer = ExperienceReplayBuffer(buffer_size=buffer_size)
        
        # Add samples
        for i in range(150):
            sample = {
                'sequence': f'ATCG' * (i % 10),
                'embedding': np.random.randn(128),
                'label': i % 5
            }
            buffer.add_sample(sample)
        
        # Check buffer size (should be capped at 100)
        assert len(buffer.buffer) == buffer_size
        
        # Sample from buffer
        samples = buffer.sample(batch_size=10)
        assert len(samples) == 10
    
    def test_ewc_fisher_computation(self, dummy_model, temp_dir):
        """Test EWC Fisher information computation"""
        learner = ContinualLearner(
            model=dummy_model,
            strategy='ewc',
            memory_dir=temp_dir
        )
        
        # Create dummy dataset
        X = torch.randn(50, 10)
        y = torch.randint(0, 5, (50,))
        
        # Compute Fisher information
        fisher_dict = learner.compute_fisher_information(
            dataloader=[(X[i:i+10], y[i:i+10]) for i in range(0, 50, 10)]
        )
        
        assert fisher_dict is not None
        assert len(fisher_dict) > 0
    
    def test_save_load_task_memory(self, dummy_model, temp_dir):
        """Test saving and loading task memory"""
        learner = ContinualLearner(
            model=dummy_model,
            strategy='combined',
            memory_dir=temp_dir
        )
        
        # Save task memory
        task_name = 'marine_dataset'
        learner.save_task_memory(
            task_name=task_name,
            model_state=dummy_model.state_dict(),
            fisher_dict={'param1': torch.randn(10)},
            replay_buffer=[{'seq': 'ATCG', 'emb': np.random.randn(128)}]
        )
        
        # Check file exists
        memory_file = Path(temp_dir) / f'{task_name}_memory.pt'
        assert memory_file.exists()
        
        # Load task memory
        loaded_memory = learner.load_task_memory(task_name)
        assert loaded_memory is not None
        assert 'model_state' in loaded_memory
        assert 'fisher_dict' in loaded_memory


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
        return ModelRegistry(registry_dir=temp_dir, backend='json')
    
    def test_registry_initialization(self, registry, temp_dir):
        """Test registry initialization"""
        assert registry.registry_dir == Path(temp_dir)
        assert registry.backend == 'json'
    
    def test_register_model(self, registry, temp_dir):
        """Test registering a new model"""
        version = registry.register_model(
            version='v1.0',
            model_path=str(Path(temp_dir) / 'model.pt'),
            base_model='zhihan1996/DNABERT-2-117M',
            datasets=['marine_edna'],
            metrics={'val_loss': 0.25, 'val_accuracy': 0.89},
            description='First model on marine dataset'
        )
        
        assert version == 'v1.0'
        
        # Retrieve model
        model_info = registry.get_model('v1.0')
        assert model_info is not None
        assert model_info['version'] == 'v1.0'
        assert 'marine_edna' in model_info['datasets']
        assert model_info['metrics']['val_loss'] == 0.25
    
    def test_list_models(self, registry, temp_dir):
        """Test listing all models"""
        # Register multiple models
        for i in range(3):
            registry.register_model(
                version=f'v1.{i}',
                model_path=str(Path(temp_dir) / f'model_{i}.pt'),
                datasets=[f'dataset_{i}'],
                metrics={'loss': 0.1 * (i + 1)}
            )
        
        # List all models
        models = registry.list_models()
        assert len(models) == 3
        assert all('version' in m for m in models)
    
    def test_model_lineage(self, registry, temp_dir):
        """Test model lineage tracking"""
        # Create parent-child relationship
        registry.register_model(
            version='v1.0',
            model_path=str(Path(temp_dir) / 'model_v1.pt'),
            datasets=['dataset1'],
            metrics={'loss': 0.5}
        )
        
        registry.register_model(
            version='v1.1',
            model_path=str(Path(temp_dir) / 'model_v1.1.pt'),
            parent_version='v1.0',
            datasets=['dataset2'],
            metrics={'loss': 0.3}
        )
        
        # Get lineage
        lineage = registry.get_lineage('v1.1')
        assert len(lineage) == 2
        assert lineage[0]['version'] == 'v1.0'
        assert lineage[1]['version'] == 'v1.1'
        
        # Get children
        children = registry.get_children('v1.0')
        assert len(children) == 1
        assert children[0]['version'] == 'v1.1'
    
    def test_compare_models(self, registry, temp_dir):
        """Test model comparison"""
        # Register two models
        registry.register_model(
            version='v1.0',
            model_path=str(Path(temp_dir) / 'model1.pt'),
            datasets=['dataset1'],
            metrics={'val_loss': 0.5, 'val_accuracy': 0.8}
        )
        
        registry.register_model(
            version='v2.0',
            model_path=str(Path(temp_dir) / 'model2.pt'),
            datasets=['dataset1', 'dataset2'],
            metrics={'val_loss': 0.3, 'val_accuracy': 0.9}
        )
        
        # Compare models
        comparison = registry.compare_models('v1.0', 'v2.0')
        
        assert comparison is not None
        assert 'metric_differences' in comparison
        assert 'dataset_differences' in comparison
        
        # Check metric differences
        assert 'val_loss' in comparison['metric_differences']
        assert comparison['metric_differences']['val_loss']['change'] < 0  # Improved
    
    def test_get_best_model(self, registry, temp_dir):
        """Test getting best model by metric"""
        # Register models with different performance
        for i, loss in enumerate([0.5, 0.2, 0.8]):
            registry.register_model(
                version=f'v1.{i}',
                model_path=str(Path(temp_dir) / f'model_{i}.pt'),
                datasets=['test'],
                metrics={'val_loss': loss}
            )
        
        # Get best model
        best = registry.get_best_model(metric='val_loss', minimize=True)
        
        assert best is not None
        assert best['metrics']['val_loss'] == 0.2
        assert best['version'] == 'v1.1'
    
    def test_update_model_status(self, registry, temp_dir):
        """Test updating model status"""
        # Register a model
        registry.register_model(
            version='v1.0',
            model_path=str(Path(temp_dir) / 'model.pt'),
            datasets=['test'],
            metrics={'loss': 0.3}
        )
        
        # Update status
        registry.update_model_status('v1.0', 'archived')
        
        # Verify update
        model_info = registry.get_model('v1.0')
        assert model_info['status'] == 'archived'


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
            checkpoint_dir=str(Path(temp_dir) / 'checkpoints')
        )
        
        registry = ModelRegistry(
            registry_dir=str(Path(temp_dir) / 'registry'),
            backend='json'
        )
        
        # 2. Create dummy model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        # 3. Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=None,
            epoch=5,
            metrics={'val_loss': 0.25, 'val_accuracy': 0.85}
        )
        
        assert checkpoint_path is not None
        
        # 4. Register model in registry
        version = registry.register_model(
            version='v1.0',
            model_path=str(Path(temp_dir) / 'model.pt'),
            checkpoint_path=checkpoint_path,
            datasets=['marine_dataset'],
            metrics={'val_loss': 0.25, 'val_accuracy': 0.85}
        )
        
        assert version == 'v1.0'
        
        # 5. Load checkpoint and verify
        loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
        assert loaded_checkpoint is not None
        assert loaded_checkpoint['epoch'] == 5
        
        # 6. Get model from registry and verify
        model_info = registry.get_model('v1.0')
        assert model_info is not None
        assert model_info['checkpoint_path'] == checkpoint_path
        assert 'marine_dataset' in model_info['datasets']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
