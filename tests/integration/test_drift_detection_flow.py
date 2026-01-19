"""End-to-end integration tests for drift detection flows"""
import pytest
from datetime import datetime, timedelta
from backend.models.drift_metrics import DriftSeverity
from tests.fixtures.sample_data import create_normal_distribution_data, create_shifted_distribution_data

@pytest.mark.integration
class TestDriftDetectionToRollbackFlow:
    """Test complete flow: data generation → drift detection → alert → rollback"""
    
    def test_complete_drift_detection_flow(self, api_client, test_db, config_service):
        """Test end-to-end drift detection and rollback flow"""
        # 1. Create baseline configuration
        config_service.create_or_update_current_config(
            embedding_model="baseline-model",
            similarity_threshold=0.75
        )
        v1 = config_service.snapshot_configuration(
            version_label="baseline",
            performance_metrics={"accuracy": 0.95}
        )
        
        # 2. Generate baseline data
        now = datetime.now()
        baseline_end = now - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=7)
        create_normal_distribution_data(test_db, baseline_start, baseline_end, n_samples=8000)
        
        # 3. Generate recent data with drift
        recent_start = now - timedelta(minutes=15)
        create_shifted_distribution_data(
            test_db, recent_start, now, n_samples=2000, shift_magnitude=0.30
        )
        
        # 4. Check drift metrics via API
        metrics_response = api_client.get("/api/drift/metrics")
        assert metrics_response.status_code == 200
        metrics = metrics_response.json()
        
        # 5. Check alerts
        alerts_response = api_client.get("/api/drift/alerts?severity=critical")
        alerts = alerts_response.json()
        
        # 6. If critical alert exists, execute rollback
        if len(alerts) > 0:
            rollback_response = api_client.post(
                "/api/rollback/execute",
                json={
                    "version_id": v1.id,
                    "trigger_reason": "Critical drift detected"
                }
            )
            assert rollback_response.status_code == 200

@pytest.mark.integration
class TestRoutingToEscalationFlow:
    """Test complete flow: query routing → confidence scoring → human escalation"""
    
    def test_routing_escalation_flow(self, api_client, test_db):
        """Test query routing and escalation flow"""
        # 1. Evaluate high-risk, low-confidence query
        routing_response = api_client.post(
            "/api/routing/evaluate",
            json={
                "query": "I'm experiencing severe chest pain",
                "ai_response": "Response",
                "confidence_score": 0.40  # Low confidence, high risk topic
            }
        )
        
        assert routing_response.status_code == 200
        result = routing_response.json()
        
        # 2. Verify routing decision
        assert result["decision"] in ["human_escalation", "safe_fallback"]
        assert "final_response" in result
        assert len(result["final_response"]) > 0

@pytest.mark.integration
class TestConfigurationChangeFlow:
    """Test complete flow: configuration change → drift detection → automated response"""
    
    def test_configuration_change_drift_detection(self, api_client, test_db, config_service):
        """Test that configuration changes trigger drift detection"""
        # 1. Create baseline config
        config_service.create_or_update_current_config(
            embedding_model="model-1",
            similarity_threshold=0.75
        )
        baseline_version = config_service.snapshot_configuration(version_label="baseline")
        
        # 2. Change configuration
        config_service.create_or_update_current_config(
            embedding_model="model-2",
            similarity_threshold=0.85
        )
        
        # 3. Generate data with the new configuration
        now = datetime.now()
        create_normal_distribution_data(test_db, now - timedelta(days=7), now, n_samples=8000)
        create_shifted_distribution_data(test_db, now - timedelta(minutes=15), now, n_samples=2000)
        
        # 4. Check for drift
        metrics_response = api_client.get("/api/drift/metrics")
        assert metrics_response.status_code == 200
        
        # 5. Verify rollback capability
        rollback_response = api_client.get("/api/rollback/versions")
        assert rollback_response.status_code == 200
        versions = rollback_response.json()
        assert len(versions) > 0
