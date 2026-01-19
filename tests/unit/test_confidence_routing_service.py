"""Unit tests for confidence_routing_service.py"""
import pytest
from backend.services.confidence_routing_service import (
    ConfidenceRoutingService,
    RouteDecision,
    RiskLevel
)

class TestTopicClassification:
    """Tests for topic classification"""
    
    def test_classify_medication_topic(self, routing_service):
        """Test classification of medication queries"""
        topic, risk, requires_validation = routing_service.classify_topic(
            "I need to refill my medication"
        )
        assert topic == "medication"
        assert risk == RiskLevel.HIGH
        assert requires_validation is True
    
    def test_classify_clinical_symptom_topic(self, routing_service):
        """Test classification of clinical symptom queries"""
        topic, risk, requires_validation = routing_service.classify_topic(
            "I'm experiencing chest pain"
        )
        assert topic == "clinical_symptom"
        assert risk == RiskLevel.CRITICAL
        assert requires_validation is True
    
    def test_classify_general_topic(self, routing_service):
        """Test classification of general queries"""
        topic, risk, requires_validation = routing_service.classify_topic(
            "What are your office hours?"
        )
        assert topic == "general"
        assert risk == RiskLevel.LOW
        assert requires_validation is False

class TestConfidenceAdjustment:
    """Tests for confidence score adjustment"""
    
    def test_adjusted_confidence_high_risk(self, routing_service):
        """Test confidence adjustment for high-risk topics"""
        base_confidence = 0.80
        adjusted = routing_service.compute_adjusted_confidence(
            base_confidence,
            RiskLevel.HIGH,
            validation_passed=True
        )
        assert adjusted < base_confidence  # Should be penalized
        assert adjusted >= 0.0
        assert adjusted <= 1.0
    
    def test_adjusted_confidence_low_risk(self, routing_service):
        """Test confidence adjustment for low-risk topics"""
        base_confidence = 0.80
        adjusted = routing_service.compute_adjusted_confidence(
            base_confidence,
            RiskLevel.LOW,
            validation_passed=True
        )
        assert adjusted == base_confidence  # No penalty for low risk
    
    def test_adjusted_confidence_validation_failed(self, routing_service):
        """Test confidence adjustment when validation fails"""
        base_confidence = 0.80
        adjusted = routing_service.compute_adjusted_confidence(
            base_confidence,
            RiskLevel.MEDIUM,
            validation_passed=False
        )
        assert adjusted < base_confidence  # Should be penalized

class TestRoutingDecisions:
    """Tests for routing decision logic"""
    
    def test_route_high_confidence_low_risk(self, routing_service):
        """Test routing high confidence, low risk queries"""
        result = routing_service.route_query(
            query="What are your office hours?",
            ai_response="Our office hours are Monday-Friday 9am-5pm",
            base_confidence=0.90
        )
        
        assert result["decision"] == RouteDecision.AI_RESPONSE
        assert result["adjusted_confidence"] >= 0.85
    
    def test_route_low_confidence(self, routing_service):
        """Test routing low confidence queries to humans"""
        result = routing_service.route_query(
            query="General question",
            ai_response="Response",
            base_confidence=0.25  # Below reject threshold
        )
        
        assert result["decision"] == RouteDecision.HUMAN_ESCALATION
    
    def test_route_high_risk_medium_confidence(self, routing_service):
        """Test routing high-risk queries with medium confidence"""
        result = routing_service.route_query(
            query="I need to refill my medication",
            ai_response="Response",
            base_confidence=0.75  # Medium confidence, but high risk
        )
        
        # Should route to human or hold for review
        assert result["decision"] in [
            RouteDecision.HUMAN_ESCALATION,
            RouteDecision.HOLD_FOR_REVIEW,
            RouteDecision.SAFE_FALLBACK
        ]

class TestFallbackResponses:
    """Tests for fallback response selection"""
    
    def test_get_fallback_responses(self, routing_service):
        """Test getting fallback responses"""
        # Fallback responses are accessed via FALLBACK_RESPONSES constant
        from backend.services.confidence_routing_service import FALLBACK_RESPONSES
        
        assert FALLBACK_RESPONSES is not None
        assert isinstance(FALLBACK_RESPONSES, dict)
        assert "default" in FALLBACK_RESPONSES
    
    def test_fallback_for_medication(self, routing_service):
        """Test fallback response for medication queries"""
        result = routing_service.route_query(
            query="I need medication information",
            ai_response="Response",
            base_confidence=0.30
        )
        
        assert "final_response" in result
        assert "medication" in result["final_response"].lower() or "pharmacist" in result["final_response"].lower()
