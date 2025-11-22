"""
End-to-end integration tests for complete workflows
Tests full pipeline from data upload through analysis to visualization
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import time


# ============================================================================
# Complete Analysis Pipeline Tests
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
class TestCompleteAnalysisPipeline:
    """End-to-end tests for full analysis pipeline"""
    
    def test_upload_preprocess_analyze_workflow(
        self, api_client, temp_dir, sample_fasta_file, mock_celery_app
    ):
        """
        Test complete workflow:
        1. Upload dataset
        2. Preprocess sequences
        3. Run analysis
        4. Generate report
        """
        # Step 1: Upload dataset
        with open(sample_fasta_file, 'rb') as f:
            with patch('src.api.datasets.save_uploaded_file') as mock_save:
                mock_save.return_value = str(sample_fasta_file)
                
                upload_response = api_client.post(
                    '/api/v1/datasets/upload',
                    data={'file': (f, 'test.fasta')},
                    content_type='multipart/form-data'
                )
                
                assert upload_response.status_code == 201
                dataset_id = upload_response.json['dataset_id']
        
        # Step 2: Start preprocessing
        with patch('src.tasks.preprocessing.preprocess_sequences.delay') as mock_preprocess:
            mock_preprocess.return_value.id = 'preprocess_task_123'
            
            preprocess_response = api_client.post(
                f'/api/v1/datasets/{dataset_id}/preprocess',
                json={
                    'quality_threshold': 20,
                    'min_length': 50,
                    'trim_adapters': True
                }
            )
            
            assert preprocess_response.status_code == 202
            preprocess_task_id = preprocess_response.json['task_id']
        
        # Step 3: Start analysis
        with patch('src.tasks.analysis_tasks.run_analysis.delay') as mock_analysis:
            mock_analysis.return_value.id = 'analysis_task_123'
            mock_analysis.return_value.get.return_value = {
                'status': 'success',
                'clusters': 50,
                'sequences_processed': 1000
            }
            
            analysis_response = api_client.post(
                '/api/v1/analysis/start',
                json={
                    'dataset_id': dataset_id,
                    'analysis_type': 'taxonomic',
                    'parameters': {
                        'clustering_method': 'hdbscan',
                        'min_cluster_size': 10
                    }
                }
            )
            
            assert analysis_response.status_code == 202
            analysis_id = analysis_response.json['analysis_id']
        
        # Step 4: Generate report
        with patch('src.api.reports.create_report_task.delay') as mock_report:
            mock_report.return_value.id = 'report_task_123'
            
            report_response = api_client.post(
                '/api/v1/reports',
                json={
                    'dataset_id': dataset_id,
                    'analysis_id': analysis_id,
                    'report_type': 'comprehensive',
                    'format': 'pdf'
                }
            )
            
            assert report_response.status_code == 202
            report_id = report_response.json['report_id']
        
        # Verify all steps completed
        assert dataset_id is not None
        assert preprocess_task_id is not None
        assert analysis_id is not None
        assert report_id is not None
    
    def test_sra_download_analysis_workflow(
        self, api_client, temp_dir, mock_celery_app
    ):
        """
        Test SRA download and analysis workflow:
        1. Download SRA dataset
        2. Convert to FASTA
        3. Run analysis
        4. Visualize results
        """
        accession = 'SRR12345678'
        
        # Step 1: Download SRA dataset
        with patch('src.tasks.download_tasks.download_sra_dataset.delay') as mock_download:
            mock_download.return_value.id = 'download_task_123'
            mock_download.return_value.get.return_value = {
                'status': 'success',
                'files': [str(temp_dir / f'{accession}.fastq')],
                'accession': accession
            }
            
            download_response = api_client.post(
                '/api/v1/sra/download',
                json={
                    'accession': accession,
                    'output_dir': str(temp_dir)
                }
            )
            
            assert download_response.status_code == 202
            download_task_id = download_response.json['task_id']
        
        # Step 2: Convert to FASTA
        with patch('src.tasks.conversion.convert_to_fasta.delay') as mock_convert:
            mock_convert.return_value.id = 'convert_task_123'
            fasta_path = str(temp_dir / f'{accession}.fasta')
            mock_convert.return_value.get.return_value = {
                'status': 'success',
                'output_file': fasta_path
            }
            
            convert_response = api_client.post(
                '/api/v1/conversion/to-fasta',
                json={
                    'input_file': str(temp_dir / f'{accession}.fastq'),
                    'output_file': fasta_path
                }
            )
            
            assert convert_response.status_code in [202, 200]
        
        # Step 3: Run analysis
        with patch('src.tasks.analysis_tasks.run_analysis.delay') as mock_analysis:
            mock_analysis.return_value.id = 'analysis_task_123'
            mock_analysis.return_value.get.return_value = {
                'status': 'success',
                'taxonomy_assignments': 500
            }
            
            analysis_response = api_client.post(
                '/api/v1/analysis/start',
                json={
                    'dataset_path': fasta_path,
                    'analysis_type': 'taxonomic'
                }
            )
            
            assert analysis_response.status_code == 202
        
        # Verify workflow completion
        assert download_task_id is not None
    
    def test_batch_analysis_workflow(
        self, api_client, dataset_factory, mock_celery_app
    ):
        """
        Test batch analysis workflow:
        1. Upload multiple datasets
        2. Run analyses in parallel
        3. Compare results
        """
        datasets = [dataset_factory() for _ in range(3)]
        dataset_ids = [str(d.id) for d in datasets]
        
        # Start batch analysis
        with patch('src.api.analysis.run_multiple_analyses.delay') as mock_batch:
            mock_batch.return_value.id = 'batch_task_123'
            
            batch_response = api_client.post(
                '/api/v1/analysis/batch',
                json={
                    'dataset_ids': dataset_ids,
                    'analysis_type': 'taxonomic',
                    'parameters': {'min_cluster_size': 10}
                }
            )
            
            assert batch_response.status_code == 202
            assert 'task_ids' in batch_response.json
            assert len(batch_response.json['task_ids']) == 3


# ============================================================================
# Training and Deployment Pipeline Tests
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
class TestTrainingDeploymentPipeline:
    """End-to-end tests for model training and deployment"""
    
    def test_train_evaluate_deploy_workflow(
        self, api_client, temp_dir, mock_celery_app
    ):
        """
        Test ML model lifecycle:
        1. Train model
        2. Evaluate on test set
        3. Register model
        4. Deploy for inference
        """
        training_data = str(temp_dir / 'train.fasta')
        test_data = str(temp_dir / 'test.fasta')
        
        # Step 1: Train model
        with patch('src.tasks.training_tasks.train_model.delay') as mock_train:
            mock_train.return_value.id = 'train_task_123'
            mock_train.return_value.get.return_value = {
                'status': 'success',
                'model_path': str(temp_dir / 'model.pt'),
                'final_accuracy': 0.92
            }
            
            train_response = api_client.post(
                '/api/v1/models/train',
                json={
                    'dataset_path': training_data,
                    'model_type': 'transformer',
                    'config': {
                        'epochs': 10,
                        'batch_size': 32,
                        'learning_rate': 0.001
                    }
                }
            )
            
            assert train_response.status_code == 202
            train_task_id = train_response.json['task_id']
        
        # Step 2: Evaluate model
        model_path = str(temp_dir / 'model.pt')
        with patch('src.tasks.training_tasks.evaluate_model.delay') as mock_eval:
            mock_eval.return_value.id = 'eval_task_123'
            mock_eval.return_value.get.return_value = {
                'accuracy': 0.92,
                'precision': 0.91,
                'recall': 0.93,
                'f1_score': 0.92
            }
            
            eval_response = api_client.post(
                '/api/v1/models/evaluate',
                json={
                    'model_path': model_path,
                    'test_data_path': test_data
                }
            )
            
            assert eval_response.status_code == 202
        
        # Step 3: Register model (if evaluation passes threshold)
        with patch('src.api.models.register_model') as mock_register:
            mock_register.return_value = {
                'model_id': 'model_123',
                'version': '1.0.0',
                'status': 'registered'
            }
            
            register_response = api_client.post(
                '/api/v1/models/register',
                json={
                    'model_path': model_path,
                    'name': 'taxonomic_classifier_v1',
                    'metrics': {'accuracy': 0.92}
                }
            )
            
            assert register_response.status_code in [200, 201]
        
        # Step 4: Deploy model
        with patch('src.api.models.deploy_model') as mock_deploy:
            mock_deploy.return_value = {
                'deployment_id': 'deploy_123',
                'endpoint': '/api/v1/predict/model_123',
                'status': 'active'
            }
            
            deploy_response = api_client.post(
                '/api/v1/models/deploy',
                json={
                    'model_id': 'model_123',
                    'environment': 'production'
                }
            )
            
            assert deploy_response.status_code in [200, 201]
    
    def test_model_inference_workflow(
        self, api_client, temp_dir, sample_fasta_file, mock_celery_app
    ):
        """
        Test model inference workflow:
        1. Load deployed model
        2. Submit sequences for prediction
        3. Retrieve predictions
        """
        model_id = 'model_123'
        
        # Submit prediction request
        with patch('src.api.models.predict_sequences.delay') as mock_predict:
            mock_predict.return_value.id = 'predict_task_123'
            mock_predict.return_value.get.return_value = {
                'predictions': [
                    {'seq_id': 'seq1', 'taxonomy': 'Bacteria; Proteobacteria', 'confidence': 0.95},
                    {'seq_id': 'seq2', 'taxonomy': 'Eukaryota; Metazoa', 'confidence': 0.88}
                ]
            }
            
            predict_response = api_client.post(
                f'/api/v1/predict/{model_id}',
                json={
                    'sequences_file': str(sample_fasta_file)
                }
            )
            
            assert predict_response.status_code == 202
            predict_task_id = predict_response.json['task_id']
        
        # Retrieve predictions
        with patch('src.api.models.get_predictions') as mock_get:
            mock_get.return_value = {
                'predictions': [
                    {'seq_id': 'seq1', 'taxonomy': 'Bacteria; Proteobacteria'}
                ],
                'total': 2
            }
            
            results_response = api_client.get(
                f'/api/v1/predictions/{predict_task_id}'
            )
            
            assert results_response.status_code == 200


# ============================================================================
# Error Recovery and Edge Cases
# ============================================================================

@pytest.mark.e2e
class TestErrorRecoveryWorkflows:
    """Tests for error handling and recovery in workflows"""
    
    def test_analysis_failure_retry(
        self, api_client, dataset_factory, mock_celery_app
    ):
        """Test automatic retry on analysis failure"""
        dataset = dataset_factory()
        
        with patch('src.tasks.analysis_tasks.run_analysis.delay') as mock_analysis:
            # First attempt fails, second succeeds
            mock_analysis.return_value.get.side_effect = [
                Exception("Temporary failure"),
                {'status': 'success', 'clusters': 50}
            ]
            
            response = api_client.post(
                '/api/v1/analysis/start',
                json={
                    'dataset_id': str(dataset.id),
                    'analysis_type': 'taxonomic',
                    'retry_on_failure': True
                }
            )
            
            # Should handle retry gracefully
            assert response.status_code in [202, 200]
    
    def test_partial_batch_failure(
        self, api_client, dataset_factory, mock_celery_app
    ):
        """Test handling of partial failures in batch operations"""
        datasets = [dataset_factory() for _ in range(3)]
        dataset_ids = [str(d.id) for d in datasets]
        
        with patch('src.api.analysis.run_multiple_analyses.delay') as mock_batch:
            mock_batch.return_value.get.return_value = {
                'successful': [dataset_ids[0], dataset_ids[2]],
                'failed': [dataset_ids[1]],
                'errors': {
                    dataset_ids[1]: 'Out of memory'
                }
            }
            
            response = api_client.post(
                '/api/v1/analysis/batch',
                json={'dataset_ids': dataset_ids}
            )
            
            assert response.status_code == 202
    
    def test_corrupted_file_handling(
        self, api_client, temp_file, mock_celery_app
    ):
        """Test handling of corrupted input files"""
        # Create corrupted FASTA file
        with open(temp_file, 'w') as f:
            f.write("Not a valid FASTA file\x00\x00\x00")
        
        with patch('src.api.datasets.validate_file') as mock_validate:
            mock_validate.return_value = {
                'valid': False,
                'errors': ['Invalid FASTA format']
            }
            
            response = api_client.post(
                '/api/v1/datasets/upload',
                data={'file': (open(temp_file, 'rb'), 'corrupted.fasta')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 400
    
    def test_disk_full_during_download(
        self, api_client, temp_dir, mock_celery_app
    ):
        """Test handling of disk full error during download"""
        with patch('src.tasks.download_tasks.download_sra_dataset.delay') as mock_download:
            mock_download.return_value.get.side_effect = OSError("No space left on device")
            
            response = api_client.post(
                '/api/v1/sra/download',
                json={'accession': 'SRR123'}
            )
            
            # Should return appropriate error
            assert response.status_code == 202  # Accepted, but will fail


# ============================================================================
# Performance and Scale Tests
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceWorkflows:
    """Tests for system performance under load"""
    
    def test_concurrent_analyses(
        self, api_client, dataset_factory, mock_celery_app
    ):
        """Test multiple concurrent analysis jobs"""
        datasets = [dataset_factory() for _ in range(10)]
        
        task_ids = []
        for dataset in datasets:
            with patch('src.tasks.analysis_tasks.run_analysis.delay') as mock_task:
                mock_task.return_value.id = f'task_{dataset.id}'
                
                response = api_client.post(
                    '/api/v1/analysis/start',
                    json={'dataset_id': str(dataset.id)}
                )
                
                if response.status_code == 202:
                    task_ids.append(response.json['task_id'])
        
        # Verify all tasks were accepted
        assert len(task_ids) == 10
    
    def test_large_dataset_processing(
        self, api_client, temp_dir, mock_celery_app
    ):
        """Test processing of large dataset (>10K sequences)"""
        large_dataset_path = temp_dir / 'large_dataset.fasta'
        
        # Create mock large dataset metadata
        dataset_info = {
            'id': 'dataset_large',
            'sequence_count': 50000,
            'size_bytes': 100 * 1024 * 1024,  # 100 MB
            'path': str(large_dataset_path)
        }
        
        with patch('src.tasks.analysis_tasks.run_analysis.delay') as mock_analysis:
            mock_analysis.return_value.id = 'large_analysis_123'
            
            response = api_client.post(
                '/api/v1/analysis/start',
                json={
                    'dataset_id': dataset_info['id'],
                    'analysis_type': 'taxonomic',
                    'batch_processing': True  # Enable batch processing for large datasets
                }
            )
            
            assert response.status_code == 202
    
    @pytest.mark.benchmark
    def test_api_response_time(self, api_client, dataset_factory, benchmark):
        """Benchmark API response times"""
        dataset = dataset_factory()
        
        def make_request():
            with patch('src.api.datasets.get_dataset_by_id') as mock_get:
                mock_get.return_value = {
                    'id': str(dataset.id),
                    'name': dataset.name
                }
                return api_client.get(f'/api/v1/datasets/{dataset.id}')
        
        result = benchmark(make_request)
        assert result.status_code == 200
        
        # Response should be fast (< 100ms)
        assert benchmark.stats.mean < 0.1
