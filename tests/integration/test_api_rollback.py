"""Integration tests for rollback API endpoints"""
import pytest

@pytest.mark.integration
class TestRollbackVersionsEndpoint:
    """Tests for GET /api/rollback/versions"""
    
    def test_get_rollback_versions(self, api_client, test_db, sample_configuration):
        """Test getting rollback versions"""
        response = api_client.get("/api/rollback/versions")
        
        assert response.status_code == 200
        versions = response.json()
        assert isinstance(versions, list)
    
    def test_versions_include_required_fields(self, api_client, test_db, sample_configuration):
        """Test versions include required fields"""
        response = api_client.get("/api/rollback/versions")
        versions = response.json()
        
        if len(versions) > 0:
            version = versions[0]
            assert "id" in version
            assert "version_label" in version
            assert "snapshot_timestamp" in version

@pytest.mark.integration
class TestRollbackExecuteEndpoint:
    """Tests for POST /api/rollback/execute"""
    
    def test_execute_rollback(self, api_client, test_db, sample_configuration):
        """Test executing a rollback"""
        v1 = sample_configuration["v1"]
        
        response = api_client.post(
            "/api/rollback/execute",
            json={"version_id": v1.id, "reason": "Test rollback"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "status" in data
    
    def test_execute_rollback_invalid_version(self, api_client, test_db):
        """Test executing rollback with invalid version ID"""
        response = api_client.post(
            "/api/rollback/execute",
            json={"version_id": 99999, "reason": "Test"}
        )
        assert response.status_code == 404

@pytest.mark.integration
class TestRollbackTriggerStatusEndpoint:
    """Tests for GET /api/rollback/triggers/status"""
    
    def test_get_trigger_status(self, api_client, test_db):
        """Test getting trigger status"""
        response = api_client.get("/api/rollback/triggers/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "triggers" in data
        assert "cooldown_active" in data

@pytest.mark.integration
class TestRollbackAuditLogEndpoint:
    """Tests for GET /api/rollback/audit-log"""
    
    def test_get_audit_log(self, api_client, test_db, sample_configuration):
        """Test getting rollback audit log"""
        response = api_client.get("/api/rollback/audit-log")
        
        assert response.status_code == 200
        audit_log = response.json()
        assert isinstance(audit_log, list)

@pytest.mark.integration
class TestMarkKnownGoodEndpoint:
    """Tests for POST /api/rollback/versions/{id}/mark-known-good"""
    
    def test_mark_version_known_good(self, api_client, test_db, sample_configuration):
        """Test marking version as known-good"""
        v1 = sample_configuration["v1"]
        
        response = api_client.post(f"/api/rollback/versions/{v1.id}/mark-known-good")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version_id" in data

@pytest.mark.integration
class TestRecommendedVersionEndpoint:
    """Tests for GET /api/rollback/recommended-version"""
    
    def test_get_recommended_version(self, api_client, test_db, sample_configuration):
        """Test getting recommended version"""
        response = api_client.get("/api/rollback/recommended-version")
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "version_label" in data
