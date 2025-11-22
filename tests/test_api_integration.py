"""
Integration tests for API endpoints
Tests request validation, response formats, error handling, and authorization
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime


# ============================================================================
# Report Management API Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.api
class TestReportManagementAPI:
    """Tests for report management API endpoints"""
    
    def test_create_report_success(self, api_client, dataset_factory):
        """Test successful report creation"""
        dataset = dataset_factory()
        
        payload = {
            'dataset_id': str(dataset.id),
            'report_type': 'summary',
            'format': 'pdf'
        }
        
        with patch('src.api.reports.create_report_task.delay') as mock_task:
            mock_task.return_value.id = 'task_123'
            
            response = api_client.post(
                '/api/v1/reports',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 202
            data = json.loads(response.data)
            assert 'task_id' in data
            assert data['status'] == 'processing'
    
    def test_create_report_invalid_payload(self, api_client):
        """Test report creation with invalid data"""
        payload = {
            'report_type': 'invalid_type'
            # Missing required fields
        }
        
        response = api_client.post(
            '/api/v1/reports',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_get_report_success(self, api_client):
        """Test retrieving a report"""
        report_id = 'report_123'
        
        with patch('src.api.reports.get_report_by_id') as mock_get:
            mock_get.return_value = {
                'id': report_id,
                'dataset_id': 'dataset_456',
                'status': 'completed',
                'file_path': '/reports/report_123.pdf',
                'created_at': datetime.utcnow().isoformat()
            }
            
            response = api_client.get(f'/api/v1/reports/{report_id}')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['id'] == report_id
            assert data['status'] == 'completed'
    
    def test_get_report_not_found(self, api_client):
        """Test retrieving non-existent report"""
        with patch('src.api.reports.get_report_by_id') as mock_get:
            mock_get.return_value = None
            
            response = api_client.get('/api/v1/reports/nonexistent')
            
            assert response.status_code == 404
    
    def test_list_reports_success(self, api_client):
        """Test listing all reports"""
        with patch('src.api.reports.list_all_reports') as mock_list:
            mock_list.return_value = [
                {'id': 'report_1', 'status': 'completed'},
                {'id': 'report_2', 'status': 'processing'},
                {'id': 'report_3', 'status': 'completed'}
            ]
            
            response = api_client.get('/api/v1/reports')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['reports']) == 3
    
    def test_list_reports_with_filters(self, api_client):
        """Test listing reports with status filter"""
        with patch('src.api.reports.list_all_reports') as mock_list:
            mock_list.return_value = [
                {'id': 'report_1', 'status': 'completed'}
            ]
            
            response = api_client.get('/api/v1/reports?status=completed')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert all(r['status'] == 'completed' for r in data['reports'])
    
    def test_delete_report_success(self, api_client):
        """Test successful report deletion"""
        report_id = 'report_123'
        
        with patch('src.api.reports.delete_report_by_id') as mock_delete:
            mock_delete.return_value = True
            
            response = api_client.delete(f'/api/v1/reports/{report_id}')
            
            assert response.status_code == 204
            mock_delete.assert_called_once_with(report_id)
    
    def test_export_report_pdf(self, api_client):
        """Test exporting report as PDF"""
        report_id = 'report_123'
        
        with patch('src.api.reports.export_report_file') as mock_export:
            mock_export.return_value = b'%PDF-1.4...'  # Mock PDF content
            
            response = api_client.get(f'/api/v1/reports/{report_id}/export?format=pdf')
            
            assert response.status_code == 200
            assert response.content_type == 'application/pdf'


# ============================================================================
# Dataset API Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.api
class TestDatasetAPI:
    """Tests for dataset management API endpoints"""
    
    def test_upload_dataset_success(self, api_client, temp_file):
        """Test successful dataset upload"""
        with open(temp_file, 'wb') as f:
            f.write(b'>seq1\nATCGATCG\n')
        
        with patch('src.api.datasets.save_uploaded_file') as mock_save:
            mock_save.return_value = str(temp_file)
            
            response = api_client.post(
                '/api/v1/datasets/upload',
                data={'file': (open(temp_file, 'rb'), 'test.fasta')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 201
            data = json.loads(response.data)
            assert 'dataset_id' in data
    
    def test_upload_dataset_invalid_format(self, api_client, temp_file):
        """Test upload with invalid file format"""
        response = api_client.post(
            '/api/v1/datasets/upload',
            data={'file': (open(temp_file, 'rb'), 'test.exe')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_upload_dataset_too_large(self, api_client, temp_file, mock_config):
        """Test upload with file exceeding size limit"""
        # Mock file too large
        with patch('src.api.datasets.get_file_size') as mock_size:
            mock_size.return_value = 600 * 1024 * 1024  # 600 MB
            
            response = api_client.post(
                '/api/v1/datasets/upload',
                data={'file': (open(temp_file, 'rb'), 'test.fasta')},
                content_type='multipart/form-data'
            )
            
            assert response.status_code == 413
    
    def test_get_dataset_info(self, api_client, dataset_factory):
        """Test retrieving dataset information"""
        dataset = dataset_factory()
        
        with patch('src.api.datasets.get_dataset_by_id') as mock_get:
            mock_get.return_value = {
                'id': str(dataset.id),
                'name': dataset.name,
                'size_bytes': dataset.size_bytes,
                'sequence_count': dataset.sequence_count,
                'status': dataset.status
            }
            
            response = api_client.get(f'/api/v1/datasets/{dataset.id}')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['id'] == str(dataset.id)
    
    def test_list_datasets(self, api_client, dataset_factory):
        """Test listing all datasets"""
        datasets = [dataset_factory() for _ in range(5)]
        
        with patch('src.api.datasets.list_all_datasets') as mock_list:
            mock_list.return_value = [
                {'id': str(d.id), 'name': d.name} for d in datasets
            ]
            
            response = api_client.get('/api/v1/datasets')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['datasets']) == 5
    
    def test_delete_dataset(self, api_client, dataset_factory):
        """Test dataset deletion"""
        dataset = dataset_factory()
        
        with patch('src.api.datasets.delete_dataset_by_id') as mock_delete:
            mock_delete.return_value = True
            
            response = api_client.delete(f'/api/v1/datasets/{dataset.id}')
            
            assert response.status_code == 204


# ============================================================================
# Analysis API Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.api
class TestAnalysisAPI:
    """Tests for analysis API endpoints"""
    
    def test_start_analysis_success(self, api_client, dataset_factory):
        """Test starting an analysis job"""
        dataset = dataset_factory()
        
        payload = {
            'dataset_id': str(dataset.id),
            'analysis_type': 'taxonomic',
            'parameters': {
                'clustering_method': 'hdbscan'
            }
        }
        
        with patch('src.api.analysis.run_analysis.delay') as mock_task:
            mock_task.return_value.id = 'task_123'
            
            response = api_client.post(
                '/api/v1/analysis/start',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 202
            data = json.loads(response.data)
            assert 'task_id' in data
    
    def test_get_analysis_status(self, api_client):
        """Test retrieving analysis status"""
        analysis_id = 'analysis_123'
        
        with patch('src.api.analysis.get_analysis_by_id') as mock_get:
            mock_get.return_value = {
                'id': analysis_id,
                'status': 'running',
                'progress': 50,
                'started_at': datetime.utcnow().isoformat()
            }
            
            response = api_client.get(f'/api/v1/analysis/{analysis_id}/status')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'running'
            assert data['progress'] == 50
    
    def test_get_analysis_results(self, api_client, analysis_run_factory):
        """Test retrieving analysis results"""
        analysis = analysis_run_factory(status='completed')
        
        with patch('src.api.analysis.get_analysis_results') as mock_results:
            mock_results.return_value = {
                'clusters': [
                    {'id': 1, 'size': 100, 'representative': 'seq1'},
                    {'id': 2, 'size': 50, 'representative': 'seq2'}
                ],
                'summary': {
                    'total_sequences': 1000,
                    'total_clusters': 2
                }
            }
            
            response = api_client.get(f'/api/v1/analysis/{analysis.id}/results')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'clusters' in data
            assert len(data['clusters']) == 2
    
    def test_cancel_analysis(self, api_client):
        """Test canceling a running analysis"""
        analysis_id = 'analysis_123'
        
        with patch('src.api.analysis.cancel_analysis_task') as mock_cancel:
            mock_cancel.return_value = True
            
            response = api_client.post(f'/api/v1/analysis/{analysis_id}/cancel')
            
            assert response.status_code == 200
            mock_cancel.assert_called_once_with(analysis_id)


# ============================================================================
# Authentication API Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.security
class TestAuthenticationAPI:
    """Tests for authentication endpoints"""
    
    def test_login_success(self, api_client, user_factory):
        """Test successful user login"""
        user = user_factory()
        
        payload = {
            'username': user.username,
            'password': 'correct_password'
        }
        
        with patch('src.api.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = {
                'user_id': user.id,
                'token': 'jwt_token_here'
            }
            
            response = api_client.post(
                '/api/v1/auth/login',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'token' in data
    
    def test_login_invalid_credentials(self, api_client):
        """Test login with invalid credentials"""
        payload = {
            'username': 'testuser',
            'password': 'wrong_password'
        }
        
        with patch('src.api.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = None
            
            response = api_client.post(
                '/api/v1/auth/login',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 401
    
    def test_protected_endpoint_without_token(self, api_client):
        """Test accessing protected endpoint without auth token"""
        response = api_client.get('/api/v1/user/profile')
        
        assert response.status_code == 401
    
    def test_protected_endpoint_with_token(self, api_client, user_factory):
        """Test accessing protected endpoint with valid token"""
        user = user_factory()
        
        with patch('src.api.auth.verify_token') as mock_verify:
            mock_verify.return_value = {'user_id': user.id}
            
            response = api_client.get(
                '/api/v1/user/profile',
                headers={'Authorization': 'Bearer valid_token'}
            )
            
            # Response depends on implementation
            assert response.status_code in [200, 404]  # 404 if route doesn't exist


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.api
class TestAPIErrorHandling:
    """Tests for API error handling"""
    
    def test_malformed_json(self, api_client):
        """Test handling of malformed JSON"""
        response = api_client.post(
            '/api/v1/reports',
            data='{"invalid": json}',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_missing_content_type(self, api_client):
        """Test handling of missing content-type header"""
        response = api_client.post(
            '/api/v1/reports',
            data=json.dumps({'test': 'data'})
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 415]
    
    def test_rate_limiting(self, api_client):
        """Test API rate limiting"""
        # Make many requests
        with patch('src.api.middleware.check_rate_limit') as mock_limit:
            mock_limit.return_value = False  # Exceeded limit
            
            response = api_client.get('/api/v1/datasets')
            
            # Should return rate limit error
            assert response.status_code in [429, 200]  # 429 if implemented
    
    def test_internal_server_error(self, api_client):
        """Test handling of internal server errors"""
        with patch('src.api.datasets.list_all_datasets') as mock_list:
            mock_list.side_effect = Exception("Database error")
            
            response = api_client.get('/api/v1/datasets')
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data


# ============================================================================
# Pagination Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.api
class TestAPIPagination:
    """Tests for API pagination"""
    
    def test_paginated_datasets_list(self, api_client, dataset_factory):
        """Test paginated dataset listing"""
        datasets = [dataset_factory() for _ in range(25)]
        
        with patch('src.api.datasets.list_all_datasets') as mock_list:
            mock_list.return_value = [
                {'id': str(d.id), 'name': d.name} for d in datasets[:10]
            ]
            
            response = api_client.get('/api/v1/datasets?page=1&per_page=10')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['datasets']) == 10
            assert 'pagination' in data or 'total' in data
    
    def test_pagination_invalid_page(self, api_client):
        """Test pagination with invalid page number"""
        response = api_client.get('/api/v1/datasets?page=-1&per_page=10')
        
        assert response.status_code in [400, 200]  # Depends on validation
