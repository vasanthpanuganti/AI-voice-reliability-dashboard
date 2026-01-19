"""
Confidence-Based Routing with Guardrails Service

Creates a safety layer that evaluates every AI response before it reaches the patient,
routing low-confidence or high-risk outputs to human agents or safe fallback responses.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re
import numpy as np
from sqlalchemy.orm import Session

from backend.models.query_log import QueryLog
from backend.config import settings


class RouteDecision(str, Enum):
    """Routing decision for a query"""
    AI_RESPONSE = "ai_response"           # Safe to respond with AI
    HUMAN_ESCALATION = "human_escalation" # Route to human agent
    SAFE_FALLBACK = "safe_fallback"       # Use safe fallback response
    HOLD_FOR_REVIEW = "hold_for_review"   # Hold for async review


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Sensitive topic patterns for healthcare
SENSITIVE_TOPICS = {
    "medication": {
        "patterns": [
            r"\b(medication|medicine|drug|prescription|dosage|dose|mg|pill|tablet)\b",
            r"\b(take|taking|took|started|stopped|missed)\b.*\b(medication|medicine|drug)\b",
            r"\b(side effect|adverse|reaction|allergy|allergic)\b",
        ],
        "risk_level": RiskLevel.HIGH,
        "requires_validation": True,
    },
    "clinical_symptom": {
        "patterns": [
            r"\b(pain|ache|hurt|burning|bleeding|swelling|fever)\b",
            r"\b(chest pain|shortness of breath|difficulty breathing|dizzy|faint)\b",
            r"\b(emergency|urgent|serious|severe|critical)\b",
        ],
        "risk_level": RiskLevel.CRITICAL,
        "requires_validation": True,
    },
    "billing_dispute": {
        "patterns": [
            r"\b(wrong|incorrect|error|mistake|overcharged|dispute)\b.*\b(bill|charge|payment)\b",
            r"\b(insurance|coverage|denied|rejected|appeal)\b",
        ],
        "risk_level": RiskLevel.MEDIUM,
        "requires_validation": False,
    },
    "personal_health_info": {
        "patterns": [
            r"\b(my|mine)\b.*\b(diagnosis|condition|disease|illness|treatment)\b",
            r"\b(test result|lab result|blood work|scan|x-ray|mri)\b",
            r"\b(medical record|health record|chart|history)\b",
        ],
        "risk_level": RiskLevel.HIGH,
        "requires_validation": True,
    },
}

# Safe fallback responses by category
FALLBACK_RESPONSES = {
    "medication": "For questions about medications, I'd like to connect you with a pharmacist or your care team who can provide accurate information. Would you like me to transfer you?",
    "clinical_symptom": "I understand you're experiencing symptoms. For your safety, I recommend speaking with a healthcare professional directly. Would you like me to help you schedule an urgent appointment or connect you with our nurse line?",
    "billing_dispute": "I see you have a concern about your bill. Let me connect you with our billing specialist who can review your account in detail. One moment please.",
    "personal_health_info": "To protect your privacy and ensure accuracy, I'll connect you with a member of your care team who can access your records securely. Is that okay?",
    "general_low_confidence": "I want to make sure you get accurate information. Let me connect you with someone who can help you directly.",
    "default": "Let me connect you with someone who can best assist you with this request.",
}


class ConfidenceRoutingService:
    """
    Service for confidence-based routing with guardrails.
    
    Evaluates every AI response before it reaches the patient,
    routing low-confidence or high-risk outputs appropriately.
    """
    
    def __init__(self, db: Session):
        self.db = db
        
        # Confidence thresholds (calibrated for healthcare)
        self.confidence_thresholds = {
            "high_confidence": 0.85,      # Safe for AI response
            "medium_confidence": 0.70,    # May need validation
            "low_confidence": 0.50,       # Likely needs human
            "reject_threshold": 0.30,     # Always escalate
        }
        
        # Topic-specific confidence adjustments
        self.topic_confidence_penalties = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.10,
            RiskLevel.HIGH: 0.20,
            RiskLevel.CRITICAL: 0.35,
        }
    
    def classify_topic(self, query: str) -> Tuple[str, RiskLevel, bool]:
        """
        Classify query into topic categories and assess risk level.
        
        Args:
            query: The patient query text
            
        Returns:
            Tuple of (topic_name, risk_level, requires_validation)
        """
        query_lower = query.lower()
        
        for topic_name, topic_config in SENSITIVE_TOPICS.items():
            for pattern in topic_config["patterns"]:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return (
                        topic_name,
                        topic_config["risk_level"],
                        topic_config["requires_validation"]
                    )
        
        # Default: general query with low risk
        return ("general", RiskLevel.LOW, False)
    
    def compute_adjusted_confidence(
        self,
        base_confidence: float,
        risk_level: RiskLevel,
        validation_passed: bool = True
    ) -> float:
        """
        Compute adjusted confidence score based on topic risk.
        
        Higher risk topics get confidence penalties to trigger more conservative routing.
        
        Args:
            base_confidence: Original confidence score from AI model
            risk_level: Risk level of the topic
            validation_passed: Whether semantic validation passed
            
        Returns:
            Adjusted confidence score
        """
        penalty = self.topic_confidence_penalties.get(risk_level, 0.0)
        adjusted = base_confidence - penalty
        
        # Additional penalty if validation failed
        if not validation_passed:
            adjusted -= 0.15
        
        return max(0.0, min(1.0, adjusted))
    
    def validate_response_semantics(
        self,
        query: str,
        response: str,
        topic: str
    ) -> Dict:
        """
        Validate AI response for semantic correctness.
        
        Checks for:
        - Contradictions
        - Hallucinated facts (dates, times, names that don't exist)
        - Unsafe medical advice
        
        Args:
            query: Original query
            response: AI-generated response
            topic: Classified topic
            
        Returns:
            Validation result with passed/failed and reasons
        """
        issues = []
        passed = True
        
        response_lower = response.lower()
        
        # Check for dangerous medical advice patterns
        dangerous_patterns = [
            (r"\b(stop|discontinue|quit)\b.*\b(medication|medicine|treatment)\b", "stop_medication_advice"),
            (r"\b(increase|double|triple)\b.*\b(dose|dosage)\b", "dosage_change_advice"),
            (r"\b(don't|do not|no need)\b.*\b(doctor|physician|emergency|hospital)\b", "discourage_medical_care"),
            (r"\b(you (have|definitely have|probably have))\b.*\b(disease|condition|cancer|diabetes)\b", "diagnosis_claim"),
        ]
        
        for pattern, issue_type in dangerous_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                issues.append({
                    "type": issue_type,
                    "severity": "high",
                    "description": f"Response may contain unsafe advice: {issue_type}"
                })
                passed = False
        
        # Check for hallucinated specifics (fake dates, times, phone numbers)
        # These patterns detect overly specific claims that should be verified
        specificity_patterns = [
            (r"\b\d{1,2}:\d{2}\s*(am|pm)\b", "specific_time"),
            (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.*\b(at|on)\b", "specific_day"),
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone_number"),
        ]
        
        for pattern, pattern_type in specificity_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                # Flag for validation - these need to be verified against source systems
                issues.append({
                    "type": f"unverified_{pattern_type}",
                    "severity": "medium",
                    "description": f"Response contains specific {pattern_type} that should be verified"
                })
        
        # Check for hedging language (indicates low model confidence)
        hedging_patterns = [
            r"\b(i think|i believe|maybe|perhaps|possibly|might be|could be|not sure)\b",
            r"\b(i'm not certain|i cannot confirm|i don't have access)\b",
        ]
        
        hedging_count = 0
        for pattern in hedging_patterns:
            hedging_count += len(re.findall(pattern, response_lower, re.IGNORECASE))
        
        if hedging_count >= 2:
            issues.append({
                "type": "high_uncertainty",
                "severity": "medium",
                "description": "Response shows high uncertainty language"
            })
        
        return {
            "passed": passed and len([i for i in issues if i["severity"] == "high"]) == 0,
            "issues": issues,
            "issue_count": len(issues),
            "high_severity_count": len([i for i in issues if i["severity"] == "high"]),
        }
    
    def route_query(
        self,
        query: str,
        ai_response: str,
        base_confidence: float,
        additional_context: Optional[Dict] = None
    ) -> Dict:
        """
        Main routing decision function.
        
        Evaluates query and response to determine the safest routing:
        - AI_RESPONSE: Safe to deliver AI response to patient
        - HUMAN_ESCALATION: Route to human agent immediately
        - SAFE_FALLBACK: Deliver safe fallback response
        - HOLD_FOR_REVIEW: Queue for async human review
        
        Args:
            query: Patient query
            ai_response: Generated AI response
            base_confidence: Model confidence score
            additional_context: Optional additional metadata
            
        Returns:
            Routing decision with explanation
        """
        # Step 1: Classify topic and risk
        topic, risk_level, requires_validation = self.classify_topic(query)
        
        # Step 2: Validate response semantics
        validation_result = self.validate_response_semantics(query, ai_response, topic)
        
        # Step 3: Compute adjusted confidence
        adjusted_confidence = self.compute_adjusted_confidence(
            base_confidence,
            risk_level,
            validation_result["passed"]
        )
        
        # Step 4: Make routing decision
        decision = self._make_routing_decision(
            adjusted_confidence,
            risk_level,
            validation_result,
            requires_validation
        )
        
        # Step 5: Get appropriate response
        if decision == RouteDecision.SAFE_FALLBACK:
            final_response = FALLBACK_RESPONSES.get(topic, FALLBACK_RESPONSES["default"])
        elif decision == RouteDecision.AI_RESPONSE:
            final_response = ai_response
        else:
            final_response = FALLBACK_RESPONSES.get(topic, FALLBACK_RESPONSES["default"])
        
        return {
            "decision": decision.value,
            "final_response": final_response,
            "topic": topic,
            "risk_level": risk_level.value,
            "base_confidence": base_confidence,
            "adjusted_confidence": adjusted_confidence,
            "validation": validation_result,
            "requires_validation": requires_validation,
            "reasoning": self._generate_reasoning(
                decision, adjusted_confidence, risk_level, validation_result
            ),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _make_routing_decision(
        self,
        adjusted_confidence: float,
        risk_level: RiskLevel,
        validation_result: Dict,
        requires_validation: bool
    ) -> RouteDecision:
        """Internal routing decision logic"""
        
        # Critical risk always goes to human
        if risk_level == RiskLevel.CRITICAL:
            return RouteDecision.HUMAN_ESCALATION
        
        # Validation failed with high severity issues
        if validation_result["high_severity_count"] > 0:
            return RouteDecision.HUMAN_ESCALATION
        
        # Very low confidence always escalates
        if adjusted_confidence < self.confidence_thresholds["reject_threshold"]:
            return RouteDecision.HUMAN_ESCALATION
        
        # Low confidence with validation requirement
        if adjusted_confidence < self.confidence_thresholds["low_confidence"]:
            if requires_validation:
                return RouteDecision.HUMAN_ESCALATION
            return RouteDecision.SAFE_FALLBACK
        
        # Medium confidence
        if adjusted_confidence < self.confidence_thresholds["medium_confidence"]:
            if risk_level == RiskLevel.HIGH:
                return RouteDecision.SAFE_FALLBACK
            if validation_result["issue_count"] > 0:
                return RouteDecision.HOLD_FOR_REVIEW
            return RouteDecision.AI_RESPONSE
        
        # High confidence
        if adjusted_confidence < self.confidence_thresholds["high_confidence"]:
            if risk_level == RiskLevel.HIGH and validation_result["issue_count"] > 0:
                return RouteDecision.HOLD_FOR_REVIEW
            return RouteDecision.AI_RESPONSE
        
        # Very high confidence - safe for AI response
        return RouteDecision.AI_RESPONSE
    
    def _generate_reasoning(
        self,
        decision: RouteDecision,
        confidence: float,
        risk_level: RiskLevel,
        validation: Dict
    ) -> str:
        """Generate human-readable reasoning for the routing decision"""
        
        reasons = []
        
        if decision == RouteDecision.HUMAN_ESCALATION:
            if risk_level == RiskLevel.CRITICAL:
                reasons.append("Query involves critical health topic requiring human expertise")
            if confidence < self.confidence_thresholds["reject_threshold"]:
                reasons.append(f"Confidence score ({confidence:.2f}) below minimum threshold")
            if validation["high_severity_count"] > 0:
                reasons.append("Response contains potentially unsafe content")
        
        elif decision == RouteDecision.SAFE_FALLBACK:
            if confidence < self.confidence_thresholds["low_confidence"]:
                reasons.append(f"Low confidence ({confidence:.2f}) - using safe fallback")
            if risk_level == RiskLevel.HIGH:
                reasons.append("High-risk topic - using conservative response")
        
        elif decision == RouteDecision.HOLD_FOR_REVIEW:
            reasons.append("Response flagged for human review before delivery")
            if validation["issue_count"] > 0:
                reasons.append(f"Found {validation['issue_count']} items requiring verification")
        
        elif decision == RouteDecision.AI_RESPONSE:
            reasons.append(f"High confidence ({confidence:.2f}) with validated response")
            if risk_level == RiskLevel.LOW:
                reasons.append("Low-risk topic suitable for AI handling")
        
        return "; ".join(reasons) if reasons else "Standard routing applied"
    
    def get_routing_stats(self, hours: int = 24) -> Dict:
        """
        Get routing statistics for monitoring dashboard.
        
        Returns breakdown of routing decisions over time period.
        """
        # This would query actual routing logs in production
        # For MVP, return aggregated stats
        return {
            "period_hours": hours,
            "total_queries": 0,
            "decisions": {
                RouteDecision.AI_RESPONSE.value: 0,
                RouteDecision.HUMAN_ESCALATION.value: 0,
                RouteDecision.SAFE_FALLBACK.value: 0,
                RouteDecision.HOLD_FOR_REVIEW.value: 0,
            },
            "by_risk_level": {
                RiskLevel.LOW.value: 0,
                RiskLevel.MEDIUM.value: 0,
                RiskLevel.HIGH.value: 0,
                RiskLevel.CRITICAL.value: 0,
            },
            "avg_confidence": 0.0,
            "validation_failure_rate": 0.0,
        }


class GuardrailsEngine:
    """
    Guardrails engine for content safety and compliance.
    
    Provides pre-response and post-response checks for:
    - Content safety (no harmful advice)
    - HIPAA compliance indicators
    - Factual accuracy flags
    """
    
    def __init__(self):
        self.safety_rules = self._load_safety_rules()
    
    def _load_safety_rules(self) -> List[Dict]:
        """Load safety rules for healthcare context"""
        return [
            {
                "id": "no_diagnosis",
                "description": "AI should not provide definitive medical diagnoses",
                "pattern": r"\b(you (have|definitely have)|this is (definitely|certainly))\b.*\b(disease|condition|syndrome|disorder)\b",
                "action": "block",
                "severity": "high",
            },
            {
                "id": "no_prescription",
                "description": "AI should not prescribe medications",
                "pattern": r"\b(you should (take|start|try)|i (recommend|prescribe))\b.*\b(mg|medication|medicine|drug)\b",
                "action": "block",
                "severity": "high",
            },
            {
                "id": "emergency_redirect",
                "description": "Redirect emergency situations to 911",
                "pattern": r"\b(heart attack|stroke|can't breathe|severe bleeding|unconscious|overdose)\b",
                "action": "redirect_emergency",
                "severity": "critical",
            },
            {
                "id": "phi_protection",
                "description": "Flag potential PHI disclosure",
                "pattern": r"\b(ssn|social security|date of birth|dob|medical record number|mrn)\b",
                "action": "flag_review",
                "severity": "medium",
            },
        ]
    
    def check_query(self, query: str) -> Dict:
        """Check incoming query against guardrails"""
        triggered_rules = []
        
        for rule in self.safety_rules:
            if re.search(rule["pattern"], query.lower(), re.IGNORECASE):
                triggered_rules.append({
                    "rule_id": rule["id"],
                    "description": rule["description"],
                    "action": rule["action"],
                    "severity": rule["severity"],
                })
        
        # Check for emergency keywords
        emergency_keywords = ["911", "emergency", "dying", "can't breathe", "heart attack", "stroke"]
        is_emergency = any(kw in query.lower() for kw in emergency_keywords)
        
        return {
            "passed": len([r for r in triggered_rules if r["action"] == "block"]) == 0,
            "is_emergency": is_emergency,
            "triggered_rules": triggered_rules,
            "recommended_action": self._get_recommended_action(triggered_rules, is_emergency),
        }
    
    def check_response(self, response: str) -> Dict:
        """Check AI response against guardrails before delivery"""
        triggered_rules = []
        
        for rule in self.safety_rules:
            if re.search(rule["pattern"], response.lower(), re.IGNORECASE):
                triggered_rules.append({
                    "rule_id": rule["id"],
                    "description": rule["description"],
                    "action": rule["action"],
                    "severity": rule["severity"],
                })
        
        blocked = any(r["action"] == "block" for r in triggered_rules)
        
        return {
            "passed": not blocked,
            "triggered_rules": triggered_rules,
            "should_block": blocked,
            "requires_modification": any(r["action"] == "flag_review" for r in triggered_rules),
        }
    
    def _get_recommended_action(self, rules: List[Dict], is_emergency: bool) -> str:
        """Get recommended action based on triggered rules"""
        if is_emergency:
            return "redirect_to_911"
        
        if any(r["action"] == "block" for r in rules):
            return "escalate_to_human"
        
        if any(r["action"] == "redirect_emergency" for r in rules):
            return "redirect_to_911"
        
        if any(r["action"] == "flag_review" for r in rules):
            return "flag_for_review"
        
        return "proceed"
