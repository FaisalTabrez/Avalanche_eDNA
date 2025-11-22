"""
Integration tests for Celery tasks
Tests task execution, error handling, retries, and result storage
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================================
# Analysis Tasks Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.celery
class TestAnalysisTasks:
    """Tests for analysis-related Celery tasks"""
    
    def test_run_analysis_success(self, mock_celery_app, temp_dir, dataset_factory):
        """Test successful analysis task execution"""
        from src.tasks.analysis_tasks import run_analysis
        
        dataset = dataset_factory()
        config = {
            'analysis_type': 'taxonomic',
            'parameters': {
                'clustering_method': 'hdbscan',
                'min_cluster_size': 10
            }
        }
        
        with patch('src.tasks.analysis_tasks.perform_analysis') as mock_perform:
            mock_perform.return_value = {
                'status': 'success',
                'clusters': 50,
                'sequences_processed': 1000
            }
            
            result = run_analysis(str(dataset.id), config)
            
            assert result['status'] == 'success'
            assert 'clusters' in result
            mock_perform.assert_called_once()
    
    def test_run_analysis_file_not_found(self, mock_celery_app):
        """Test analysis task with missing input file"""
        from src.tasks.analysis_tasks import run_analysis
        
        with pytest.raises(FileNotFoundError):
            run_analysis('nonexistent_dataset_id', {})
    
    def test_run_analysis_retry_on_failure(self, mock_celery_app, dataset_factory):
        """Test analysis task retry mechanism"""
        from src.tasks.analysis_tasks import run_analysis
        
        dataset = dataset_factory()
        config = {'analysis_type': 'taxonomic'}
        
        with patch('src.tasks.analysis_tasks.perform_analysis') as mock_perform:
            mock_perform.side_effect = [
                Exception("Temporary error"),
                {'status': 'success'}
            ]
            
            # First call should fail and retry
            with pytest.raises(Exception):
                run_analysis(str(dataset.id), config)
    
    def test_run_blast_search_success(self, mock_celery_app, sample_fasta_file):
        """Test BLAST search task execution"""
        from src.tasks.analysis_tasks import run_blast_search
        
        query_sequences = str(sample_fasta_file)
        database = 'nt'
        
        with patch('src.tasks.analysis_tasks.run_blast') as mock_blast:
            mock_blast.return_value = {
                'hits': [
                    {'query_id': 'seq1', 'subject_id': 'hit1', 'evalue': 1e-50},
                    {'query_id': 'seq2', 'subject_id': 'hit2', 'evalue': 1e-45}
                ]
            }
            
            result = run_blast_search(query_sequences, database)
            
            assert 'hits' in result
            assert len(result['hits']) == 2
            mock_blast.assert_called_once()
    
    def test_run_multiple_analyses_parallel(self, mock_celery_app, dataset_factory):
        """Test parallel execution of multiple analyses"""
        from src.tasks.analysis_tasks import run_multiple_analyses
        
        datasets = [dataset_factory() for _ in range(3)]
        dataset_ids = [str(d.id) for d in datasets]
        
        with patch('src.tasks.analysis_tasks.run_analysis.delay') as mock_task:
            mock_task.return_value.id = 'task_123'
            
            result = run_multiple_analyses(dataset_ids, {})
            
            assert 'task_ids' in result
            assert len(result['task_ids']) == 3
            assert mock_task.call_count == 3


# ============================================================================
# Training Tasks Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.celery
class TestTrainingTasks:
    """Tests for model training Celery tasks"""
    
    def test_train_model_success(self, mock_celery_app, temp_dir):
        """Test successful model training"""
        from src.tasks.training_tasks import train_model
        
        training_config = {
            'model_type': 'transformer',
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        dataset_path = str(temp_dir / 'train_data.fasta')
        
        with patch('src.tasks.training_tasks.build_and_train_model') as mock_train:
            mock_train.return_value = {
                'model_path': '/models/model_123.pt',
                'final_loss': 0.05,
                'final_accuracy': 0.95,
                'epochs_completed': 10
            }
            
            result = train_model(dataset_path, training_config)
            
            assert result['final_accuracy'] == 0.95
            assert result['epochs_completed'] == 10
            mock_train.assert_called_once()
    
    def test_train_model_progress_updates(self, mock_celery_app, mock_celery_task, temp_dir):
        """Test training task sends progress updates"""
        from src.tasks.training_tasks import train_model
        
        training_config = {'model_type': 'cnn', 'epochs': 5}
        dataset_path = str(temp_dir / 'train_data.fasta')
        
        with patch('src.tasks.training_tasks.build_and_train_model') as mock_train:
            def training_callback(epoch, metrics):
                # Simulate progress callback
                mock_celery_task.update_state(
                    state='PROGRESS',
                    meta={'epoch': epoch, 'loss': metrics.get('loss')}
                )
            
            mock_train.return_value = {'final_accuracy': 0.90}
            
            result = train_model.apply(args=[dataset_path, training_config]).get()
            
            assert 'final_accuracy' in result
    
    def test_evaluate_model_success(self, mock_celery_app, temp_dir):
        """Test model evaluation task"""
        from src.tasks.training_tasks import evaluate_model
        
        model_path = str(temp_dir / 'model.pt')
        test_data_path = str(temp_dir / 'test_data.fasta')
        
        with patch('src.tasks.training_tasks.load_and_evaluate') as mock_eval:
            mock_eval.return_value = {
                'accuracy': 0.92,
                'precision': 0.91,
                'recall': 0.93,
                'f1_score': 0.92
            }
            
            result = evaluate_model(model_path, test_data_path)
            
            assert result['accuracy'] == 0.92
            assert 'precision' in result
            mock_eval.assert_called_once()
    
    def test_hyperparameter_tuning_grid_search(self, mock_celery_app, temp_dir):
        """Test hyperparameter tuning task"""
        from src.tasks.training_tasks import hyperparameter_tuning
        
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32],
            'epochs': [5, 10]
        }
        dataset_path = str(temp_dir / 'train_data.fasta')
        
        with patch('src.tasks.training_tasks.grid_search') as mock_search:
            mock_search.return_value = {
                'best_params': {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10},
                'best_score': 0.95,
                'all_results': []
            }
            
            result = hyperparameter_tuning(dataset_path, param_grid)
            
            assert 'best_params' in result
            assert result['best_score'] == 0.95


# ============================================================================
# Download Tasks Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.celery
class TestDownloadTasks:
    """Tests for data download Celery tasks"""
    
    def test_download_sra_dataset_success(self, mock_celery_app, temp_dir):
        """Test SRA dataset download"""
        from src.tasks.download_tasks import download_sra_dataset
        
        accession = 'SRR12345678'
        output_dir = str(temp_dir)
        
        with patch('src.tasks.download_tasks.run_prefetch') as mock_prefetch, \
             patch('src.tasks.download_tasks.run_fasterq_dump') as mock_dump:
            
            mock_prefetch.return_value = True
            mock_dump.return_value = [
                str(temp_dir / 'SRR12345678_1.fastq'),
                str(temp_dir / 'SRR12345678_2.fastq')
            ]
            
            result = download_sra_dataset(accession, output_dir)
            
            assert result['status'] == 'success'
            assert len(result['files']) == 2
            mock_prefetch.assert_called_once_with(accession)
            mock_dump.assert_called_once()
    
    def test_download_sra_dataset_network_error(self, mock_celery_app, temp_dir):
        """Test SRA download with network failure"""
        from src.tasks.download_tasks import download_sra_dataset
        
        accession = 'SRR12345678'
        
        with patch('src.tasks.download_tasks.run_prefetch') as mock_prefetch:
            mock_prefetch.side_effect = ConnectionError("Network unavailable")
            
            with pytest.raises(ConnectionError):
                download_sra_dataset(accession, str(temp_dir))
    
    def test_download_batch_sra_success(self, mock_celery_app, temp_dir):
        """Test batch SRA download"""
        from src.tasks.download_tasks import download_batch_sra
        
        accessions = ['SRR001', 'SRR002', 'SRR003']
        output_dir = str(temp_dir)
        
        with patch('src.tasks.download_tasks.download_sra_dataset.delay') as mock_download:
            mock_download.return_value.id = 'task_123'
            
            result = download_batch_sra(accessions, output_dir)
            
            assert 'task_ids' in result
            assert len(result['task_ids']) == 3
            assert mock_download.call_count == 3
    
    def test_download_reference_database_success(self, mock_celery_app, temp_dir):
        """Test reference database download"""
        from src.tasks.download_tasks import download_reference_database
        
        db_name = 'silva'
        version = '138.1'
        
        with patch('src.tasks.download_tasks.download_file') as mock_download, \
             patch('src.tasks.download_tasks.verify_checksum') as mock_verify:
            
            mock_download.return_value = str(temp_dir / 'silva_138.1.fasta.gz')
            mock_verify.return_value = True
            
            result = download_reference_database(db_name, version, str(temp_dir))
            
            assert result['status'] == 'success'
            assert 'file_path' in result
            mock_download.assert_called_once()
            mock_verify.assert_called_once()


# ============================================================================
# Maintenance Tasks Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.celery
class TestMaintenanceTasks:
    """Tests for maintenance Celery tasks"""
    
    def test_cleanup_old_results_success(self, mock_celery_app, temp_dir):
        """Test cleanup of old result files"""
        from src.tasks.maintenance_tasks import cleanup_old_results
        
        retention_days = 30
        
        # Create mock old files
        old_file = temp_dir / 'old_result.json'
        old_file.write_text('{}')
        
        with patch('src.tasks.maintenance_tasks.find_old_files') as mock_find:
            mock_find.return_value = [str(old_file)]
            
            result = cleanup_old_results(str(temp_dir), retention_days)
            
            assert 'files_deleted' in result
            assert result['files_deleted'] >= 0
    
    def test_backup_database_success(self, mock_celery_app, temp_dir, mock_config):
        """Test database backup task"""
        from src.tasks.maintenance_tasks import backup_database
        
        backup_dir = str(temp_dir / 'backups')
        
        with patch('src.tasks.maintenance_tasks.create_db_backup') as mock_backup:
            backup_path = str(temp_dir / 'backup_20251122.sql.gz')
            mock_backup.return_value = {
                'backup_path': backup_path,
                'size_bytes': 1024 * 1024,
                'duration_seconds': 5.0
            }
            
            result = backup_database(backup_dir)
            
            assert result['backup_path'] == backup_path
            assert result['size_bytes'] > 0
            mock_backup.assert_called_once()
    
    def test_backup_database_failure(self, mock_celery_app, temp_dir):
        """Test backup task failure handling"""
        from src.tasks.maintenance_tasks import backup_database
        
        with patch('src.tasks.maintenance_tasks.create_db_backup') as mock_backup:
            mock_backup.side_effect = Exception("Disk full")
            
            with pytest.raises(Exception) as exc_info:
                backup_database(str(temp_dir))
            
            assert "Disk full" in str(exc_info.value)
    
    def test_monitor_system_health_success(self, mock_celery_app):
        """Test system health monitoring task"""
        from src.tasks.maintenance_tasks import monitor_system_health
        
        with patch('src.tasks.maintenance_tasks.get_system_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'cpu_percent': 45.0,
                'memory_percent': 60.0,
                'disk_percent': 70.0,
                'status': 'healthy'
            }
            
            result = monitor_system_health()
            
            assert result['status'] == 'healthy'
            assert result['cpu_percent'] < 90
            assert result['memory_percent'] < 90
    
    def test_cleanup_temp_files_success(self, mock_celery_app, temp_dir):
        """Test temporary files cleanup"""
        from src.tasks.maintenance_tasks import cleanup_temp_files
        
        temp_patterns = ['*.tmp', '*.temp', '*.cache']
        
        # Create temp files
        (temp_dir / 'test.tmp').write_text('temp')
        (temp_dir / 'test.cache').write_text('cache')
        
        result = cleanup_temp_files(str(temp_dir), temp_patterns)
        
        assert 'files_deleted' in result
        assert result['files_deleted'] >= 0
    
    def test_optimize_database_success(self, mock_celery_app):
        """Test database optimization task"""
        from src.tasks.maintenance_tasks import optimize_database
        
        with patch('src.tasks.maintenance_tasks.run_vacuum') as mock_vacuum, \
             patch('src.tasks.maintenance_tasks.analyze_tables') as mock_analyze:
            
            mock_vacuum.return_value = True
            mock_analyze.return_value = {'tables_analyzed': 10}
            
            result = optimize_database()
            
            assert result['tables_analyzed'] == 10
            mock_vacuum.assert_called_once()
            mock_analyze.assert_called_once()


# ============================================================================
# Task Chain and Workflow Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.celery
@pytest.mark.slow
class TestTaskWorkflows:
    """Tests for complex task chains and workflows"""
    
    def test_download_and_analyze_workflow(self, mock_celery_app, temp_dir):
        """Test workflow: download SRA -> analyze sequences"""
        from src.tasks.download_tasks import download_sra_dataset
        from src.tasks.analysis_tasks import run_analysis
        
        accession = 'SRR12345678'
        
        with patch('src.tasks.download_tasks.run_prefetch'), \
             patch('src.tasks.download_tasks.run_fasterq_dump') as mock_dump, \
             patch('src.tasks.analysis_tasks.perform_analysis') as mock_analyze:
            
            # Mock download
            output_file = str(temp_dir / f'{accession}.fastq')
            mock_dump.return_value = [output_file]
            
            # Mock analysis
            mock_analyze.return_value = {
                'status': 'success',
                'clusters': 25
            }
            
            # Execute workflow
            download_result = download_sra_dataset(accession, str(temp_dir))
            assert download_result['status'] == 'success'
            
            analysis_result = run_analysis(download_result['files'][0], {})
            assert analysis_result['status'] == 'success'
    
    def test_train_and_evaluate_workflow(self, mock_celery_app, temp_dir):
        """Test workflow: train model -> evaluate -> save results"""
        from src.tasks.training_tasks import train_model, evaluate_model
        
        dataset_path = str(temp_dir / 'data.fasta')
        test_path = str(temp_dir / 'test.fasta')
        
        with patch('src.tasks.training_tasks.build_and_train_model') as mock_train, \
             patch('src.tasks.training_tasks.load_and_evaluate') as mock_eval:
            
            # Mock training
            model_path = str(temp_dir / 'model.pt')
            mock_train.return_value = {
                'model_path': model_path,
                'final_accuracy': 0.90
            }
            
            # Mock evaluation
            mock_eval.return_value = {
                'accuracy': 0.92,
                'f1_score': 0.91
            }
            
            # Execute workflow
            train_result = train_model(dataset_path, {'epochs': 10})
            assert 'model_path' in train_result
            
            eval_result = evaluate_model(train_result['model_path'], test_path)
            assert eval_result['accuracy'] > 0.90
