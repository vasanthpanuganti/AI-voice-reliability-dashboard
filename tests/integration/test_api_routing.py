"""Integration tests for routing API endpoints"""
import pytest

@pytest.mark.integration
class TestRoutingEvaluateEndpoint:
    """Tests for POST /api/routing/evaluate"""
    
    def test_evaluate_query_routing(self, api_client, test_db):
        """Test evaluating query routing"""
        response = api_client.post(
            "/api/routing/evaluate",
            json={
                "query": "I need to refill my medication",
                "ai_response": "Response text",
                "confidence_score": 0.75
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        assert "adjusted_confidence" in data
        assert "final_response" in data
    
    def test_evaluate_low_confidence_query(self, api_client, test_db):
        """Test evaluating low confidence query"""
        response = api_client.post(
            "/api/routing/evaluate",
            json={
                "query": "Test query",
                "ai_response": "Response",
                "confidence_score": 0.25  # Below threshold
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should route to human or fallback
        assert data["decision"] in ["human_escalation", "safe_fallback"]

@pytest.mark.integration
class TestRoutingThresholdsEndpoint:
    """Tests for GET /api/routing/thresholds"""
    
    def test_get_routing_thresholds(self, api_client, test_db):
        """Test getting routing thresholds"""
        response = api_client.get("/api/routing/thresholds")
        
        assert response.status_code == 200
        data = response.json()
        assert "thresholds" in data
        assert "topic_penalties" in data

@pytest.mark.integration
class TestRoutingTopicsEndpoint:
    """Tests for GET /api/routing/topics"""
    
    def test_get_sensitive_topics(self, api_client, test_db):
        """Test getting sensitive topics"""
        response = api_client.get("/api/routing/topics")
        
        assert response.status_code == 200
        topics = response.json()
        assert isinstance(topics, dict)
        assert len(topics) > 0

@pytest.mark.integration
class TestRoutingStatsEndpoint:
    """Tests for GET /api/routing/stats"""
    
    def test_get_routing_stats(self, api_client, test_db):
        """Test getting routing statistics"""
        response = api_client.get("/api/routing/stats")
        
        assert response.status_code == 200
        stats = response.json()
        assert isinstance(stats, dict)
