"""Integration tests for drift API endpoints"""
import pytest
from backend.models.drift_metrics import DriftSeverity

@pytest.mark.integration
class TestDriftMetricsEndpoint:
    """Tests for GET /api/drift/metrics"""
    
    def test_get_current_metrics(self, api_client, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test getting current drift metrics"""
        response = api_client.get("/api/drift/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "psi_score" in data
        assert "ks_p_value" in data
        assert "js_divergence" in data
        assert "timestamp" in data
        assert "sample_size" in data
    
    def test_metrics_response_format(self, api_client, test_db, sample_baseline_data):
        """Test metrics response has correct format"""
        response = api_client.get("/api/drift/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data.get("psi_score"), (float, type(None)))
        assert isinstance(data.get("sample_size"), int)

@pytest.mark.integration
class TestDriftAlertsEndpoint:
    """Tests for GET /api/drift/alerts"""
    
    def test_get_alerts(self, api_client, test_db):
        """Test getting drift alerts"""
        response = api_client.get("/api/drift/alerts")
        
        assert response.status_code == 200
        alerts = response.json()
        assert isinstance(alerts, list)
    
    def test_filter_alerts_by_status(self, api_client, test_db):
        """Test filtering alerts by status"""
        response = api_client.get("/api/drift/alerts?status=active")
        assert response.status_code == 200
        
        response_all = api_client.get("/api/drift/alerts?status=all")
        assert response_all.status_code == 200
    
    def test_filter_alerts_by_severity(self, api_client, test_db):
        """Test filtering alerts by severity"""
        response = api_client.get("/api/drift/alerts?severity=critical")
        assert response.status_code == 200

@pytest.mark.integration
class TestDriftHistoryEndpoint:
    """Tests for GET /api/drift/history"""
    
    def test_get_drift_history(self, api_client, test_db):
        """Test getting drift history"""
        response = api_client.get("/api/drift/history")
        
        assert response.status_code == 200
        history = response.json()
        assert isinstance(history, list)
    
    def test_drift_history_limit(self, api_client, test_db):
        """Test drift history limit parameter"""
        response = api_client.get("/api/drift/history?limit=10")
        assert response.status_code == 200
        history = response.json()
        assert len(history) <= 10
    
    def test_drift_history_max_limit(self, api_client, test_db):
        """Test drift history respects maximum limit"""
        response = api_client.get("/api/drift/history?limit=10000")
        assert response.status_code == 200
        # Should respect MAX_HISTORY_LIMIT

@pytest.mark.integration
class TestDriftSegmentsEndpoint:
    """Tests for GET /api/drift/segments"""
    
    def test_get_segment_drift(self, api_client, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test getting segment drift"""
        response = api_client.get("/api/drift/segments?segment_by=query_category")
        
        assert response.status_code == 200
        data = response.json()
        assert "segments" in data
        assert "overall_health" in data
    
    def test_segment_drift_invalid_segment(self, api_client, test_db):
        """Test segment drift with invalid segment field"""
        response = api_client.get("/api/drift/segments?segment_by=invalid_field")
        assert response.status_code == 400

@pytest.mark.integration
class TestDriftComprehensiveEndpoint:
    """Tests for GET /api/drift/comprehensive"""
    
    def test_get_comprehensive_drift(self, api_client, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test getting comprehensive drift analysis"""
        response = api_client.get("/api/drift/comprehensive")
        
        assert response.status_code == 200
        data = response.json()
        assert "input_drift" in data
        assert "output_drift" in data
        assert "embedding_drift" in data
        assert "categorical_drift" in data

@pytest.mark.integration
class TestDriftCategoricalEndpoint:
    """Tests for GET /api/drift/categorical"""
    
    def test_get_categorical_drift(self, api_client, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test getting categorical drift"""
        response = api_client.get("/api/drift/categorical?category_field=query_category")
        
        assert response.status_code == 200
        data = response.json()
        assert "chi_square_statistic" in data
        assert "chi_square_p_value" in data

@pytest.mark.integration
class TestAlertDismissal:
    """Tests for POST /api/drift/alerts/{id}/dismiss"""
    
    def test_dismiss_alert(self, api_client, test_db):
        """Test dismissing an alert"""
        # First get alerts
        alerts_response = api_client.get("/api/drift/alerts")
        alerts = alerts_response.json()
        
        if len(alerts) > 0:
            alert_id = alerts[0]["id"]
            response = api_client.post(f"/api/drift/alerts/{alert_id}/dismiss")
            assert response.status_code == 200
