"""
Confidence-Based Routing API Endpoints

Provides API for confidence-based routing with guardrails,
evaluating AI responses before delivery to patients.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict

from backend.database import get_db
from backend.services.confidence_routing_service import (
    ConfidenceRoutingService,
    GuardrailsEngine,
    RouteDecision,
    RiskLevel,
)

router = APIRouter(prefix="/api/routing", tags=["routing"])


class RouteQueryRequest(BaseModel):
    """Request model for routing a query"""
    query: str
    ai_response: str
    confidence_score: float
    additional_context: Optional[Dict] = None


class RouteQueryResponse(BaseModel):
    """Response model for routing decision"""
    decision: str
    final_response: str
    topic: str
    risk_level: str
    base_confidence: float
    adjusted_confidence: float
    requires_validation: bool
    reasoning: str


class GuardrailCheckRequest(BaseModel):
    """Request model for guardrail checks"""
    text: str
    check_type: str = "query"  # "query" or "response"


@router.post("/evaluate", response_model=RouteQueryResponse)
def evaluate_query(
    request: RouteQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Evaluate a query and AI response for routing decision.
    
    This endpoint determines whether to:
    - Deliver the AI response directly
    - Route to a human agent
    - Use a safe fallback response
    - Hold for review
    """
    routing_service = ConfidenceRoutingService(db)
    
    result = routing_service.route_query(
        query=request.query,
        ai_response=request.ai_response,
        base_confidence=request.confidence_score,
        additional_context=request.additional_context
    )
    
    return RouteQueryResponse(
        decision=result["decision"],
        final_response=result["final_response"],
        topic=result["topic"],
        risk_level=result["risk_level"],
        base_confidence=result["base_confidence"],
        adjusted_confidence=result["adjusted_confidence"],
        requires_validation=result["requires_validation"],
        reasoning=result["reasoning"]
    )


@router.post("/guardrails/check")
def check_guardrails(request: GuardrailCheckRequest):
    """
    Check text against guardrails for content safety.
    
    Can check either queries (incoming) or responses (outgoing).
    """
    engine = GuardrailsEngine()
    
    if request.check_type == "query":
        result = engine.check_query(request.text)
    else:
        result = engine.check_response(request.text)
    
    return result


@router.get("/stats")
def get_routing_stats(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """
    Get routing statistics for the monitoring dashboard.
    
    Returns breakdown of routing decisions over specified time period.
    """
    routing_service = ConfidenceRoutingService(db)
    return routing_service.get_routing_stats(hours=hours)


@router.get("/thresholds")
def get_confidence_thresholds(db: Session = Depends(get_db)):
    """
    Get current confidence threshold configuration.
    """
    routing_service = ConfidenceRoutingService(db)
    return {
        "thresholds": routing_service.confidence_thresholds,
        "topic_penalties": {
            level.value: penalty
            for level, penalty in routing_service.topic_confidence_penalties.items()
        }
    }


@router.get("/topics")
def get_sensitive_topics():
    """
    Get list of sensitive topics and their risk levels.
    """
    from backend.services.confidence_routing_service import SENSITIVE_TOPICS
    
    return {
        topic: {
            "risk_level": config["risk_level"].value,
            "requires_validation": config["requires_validation"],
            "pattern_count": len(config["patterns"])
        }
        for topic, config in SENSITIVE_TOPICS.items()
    }


@router.get("/fallbacks")
def get_fallback_responses():
    """
    Get configured fallback responses by category.
    """
    from backend.services.confidence_routing_service import FALLBACK_RESPONSES
    return FALLBACK_RESPONSES
