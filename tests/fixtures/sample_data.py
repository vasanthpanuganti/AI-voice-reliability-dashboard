"""Test data generators for drift detection tests"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from backend.models.query_log import QueryLog

def generate_simple_embedding(query: str, dimension: int = 384) -> list:
    """Generate a simple deterministic embedding based on query text."""
    np.random.seed(hash(query) % (2**32))
    embedding = np.random.randn(dimension).astype(float)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

def generate_query_logs(
    db_session,
    n_samples: int,
    category_distribution: Dict[str, float],
    start_time: datetime,
    end_time: datetime,
    confidence_distribution: tuple = (8, 2),  # Beta distribution parameters
    sample_queries: Dict[str, List[str]] = None
) -> List[QueryLog]:
    """
    Generate query logs with specified distribution.
    
    Args:
        db_session: Database session
        n_samples: Number of queries to generate
        category_distribution: Dict mapping category to percentage (must sum to 1.0)
        start_time: Start timestamp
        end_time: End timestamp
        confidence_distribution: Beta distribution parameters for confidence scores
        sample_queries: Dict mapping category to list of sample query strings
    
    Returns:
        List of created QueryLog objects
    """
    if sample_queries is None:
        sample_queries = {
            "appointment": ["I need to schedule an appointment", "Can I book a visit?", "I want to cancel my appointment"],
            "prescription": ["Can I refill my medication?", "I need a prescription renewal", "Is my medication ready?"],
            "billing": ["What is my account balance?", "Can I set up a payment plan?", "I have a question about my bill"],
            "clinical_symptom": ["I have a headache", "I'm experiencing chest pain", "My blood pressure seems high"],
            "general": ["What are your office hours?", "Where is your clinic located?", "Do you accept walk-ins?"],
        }
    
    queries_created = []
    window_duration = (end_time - start_time).total_seconds()
    
    for category, pct in category_distribution.items():
        n_cat = int(n_samples * pct)
        queries = sample_queries.get(category, [f"Sample {category} query"])
        
        for i in range(n_cat):
            query_text = queries[i % len(queries)]
            timestamp = start_time + timedelta(
                seconds=np.random.uniform(0, window_duration) if window_duration > 0 else 0
            )
            
            confidence = round(np.random.beta(confidence_distribution[0], confidence_distribution[1]), 4)
            
            query_log = QueryLog(
                query=query_text,
                query_category=category,
                embedding=generate_simple_embedding(query_text),
                confidence_score=confidence,
                ai_response=f"Response to {category} query: {query_text}",
                timestamp=timestamp
            )
            db_session.add(query_log)
            queries_created.append(query_log)
    
    db_session.commit()
    return queries_created

def create_normal_distribution_data(db_session, start_time: datetime, end_time: datetime, n_samples: int = 8000):
    """Create baseline data with normal distribution"""
    category_distribution = {
        "appointment": 0.35,
        "prescription": 0.25,
        "billing": 0.20,
        "clinical_symptom": 0.10,
        "general": 0.10,
    }
    return generate_query_logs(
        db_session, n_samples, category_distribution, start_time, end_time,
        confidence_distribution=(8, 2)  # High confidence baseline
    )

def create_shifted_distribution_data(
    db_session, 
    start_time: datetime, 
    end_time: datetime, 
    n_samples: int = 2000,
    shift_magnitude: float = 0.30
):
    """Create recent data with shifted distribution for drift testing"""
    # Shift billing from 20% to 20% + shift_magnitude
    billing_shift = min(shift_magnitude, 0.50)  # Cap at 50%
    
    category_distribution = {
        "appointment": 0.35 - (billing_shift * 0.3),  # Reduce proportionally
        "prescription": 0.25 - (billing_shift * 0.2),
        "billing": 0.20 + billing_shift,
        "clinical_symptom": 0.10,
        "general": 0.10,
    }
    
    # Normalize to ensure sum = 1.0
    total = sum(category_distribution.values())
    category_distribution = {k: v/total for k, v in category_distribution.items()}
    
    return generate_query_logs(
        db_session, n_samples, category_distribution, start_time, end_time,
        confidence_distribution=(5, 4)  # Lower confidence for output drift
    )

def create_low_confidence_queries(db_session, n_samples: int = 100, threshold: float = 0.30):
    """Create queries with confidence scores below threshold for routing tests"""
    queries = []
    now = datetime.now()
    
    for i in range(n_samples):
        confidence = np.random.uniform(0.0, threshold - 0.01)  # Below threshold
        query_log = QueryLog(
            query=f"Low confidence query {i}",
            query_category="general",
            embedding=generate_simple_embedding(f"query_{i}"),
            confidence_score=round(confidence, 4),
            ai_response=f"Response {i}",
            timestamp=now - timedelta(minutes=14, seconds=i)
        )
        db_session.add(query_log)
        queries.append(query_log)
    
    db_session.commit()
    return queries
