"""Pytest configuration and shared fixtures"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import Base, get_db
from backend.api.main import app
from backend.models.query_log import QueryLog
from backend.models.configuration import Configuration, ConfigVersion
from backend.models.drift_metrics import DriftMetric, DriftAlert
from backend.services.drift_detection_service import DriftDetectionService
from backend.services.rollback_service import RollbackService
from backend.services.confidence_routing_service import ConfidenceRoutingService
from backend.services.configuration_service import ConfigurationService

# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine"""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_db(test_engine) -> Generator[Session, None, None]:
    """Create a test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

@pytest.fixture(scope="function")
def api_client(test_db: Session, test_engine) -> TestClient:
    """Create a FastAPI test client with test database override"""
    # Ensure tables are created
    Base.metadata.create_all(bind=test_engine)
    
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def drift_service(test_db: Session) -> DriftDetectionService:
    """Create a DriftDetectionService instance"""
    return DriftDetectionService(test_db)

@pytest.fixture(scope="function")
def rollback_service(test_db: Session) -> RollbackService:
    """Create a RollbackService instance"""
    return RollbackService(test_db)

@pytest.fixture(scope="function")
def routing_service(test_db: Session) -> ConfidenceRoutingService:
    """Create a ConfidenceRoutingService instance"""
    return ConfidenceRoutingService(test_db)

@pytest.fixture(scope="function")
def config_service(test_db: Session) -> ConfigurationService:
    """Create a ConfigurationService instance"""
    return ConfigurationService(test_db)

def generate_simple_embedding(query: str, dimension: int = 384) -> list:
    """Generate a simple deterministic embedding based on query text."""
    np.random.seed(hash(query) % (2**32))
    embedding = np.random.randn(dimension).astype(float)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

@pytest.fixture(scope="function")
def sample_baseline_data(test_db: Session):
    """Create 7 days of baseline query data"""
    queries_created = []
    now = datetime.now()
    start_time = now - timedelta(days=7)
    
    # Normal distribution: appointment 35%, prescription 25%, billing 20%, clinical_symptom 10%, general 10%
    category_distribution = {
        "appointment": 0.35,
        "prescription": 0.25,
        "billing": 0.20,
        "clinical_symptom": 0.10,
        "general": 0.10,
    }
    
    sample_queries = {
        "appointment": ["I need to schedule an appointment", "Can I book a visit?", "I want to cancel my appointment"],
        "prescription": ["Can I refill my medication?", "I need a prescription renewal", "Is my medication ready?"],
        "billing": ["What is my account balance?", "Can I set up a payment plan?", "I have a question about my bill"],
        "clinical_symptom": ["I have a headache", "I'm experiencing chest pain", "My blood pressure seems high"],
        "general": ["What are your office hours?", "Where is your clinic located?", "Do you accept walk-ins?"],
    }
    
    n_samples = 8000
    for category, pct in category_distribution.items():
        n_cat = int(n_samples * pct)
        queries = sample_queries[category]
        
        for i in range(n_cat):
            query_text = queries[i % len(queries)]
            timestamp = start_time + timedelta(
                seconds=np.random.uniform(0, (now - start_time).total_seconds())
            )
            
            query_log = QueryLog(
                query=query_text,
                query_category=category,
                embedding=generate_simple_embedding(query_text),
                confidence_score=round(np.random.beta(8, 2), 4),  # High confidence baseline
                ai_response=f"Response to {category} query: {query_text}",
                timestamp=timestamp
            )
            test_db.add(query_log)
            queries_created.append(query_log)
    
    test_db.commit()
    return queries_created

@pytest.fixture(scope="function")
def sample_recent_data_with_drift(test_db: Session):
    """Create recent queries with 20% shift in distribution (for drift testing)"""
    queries_created = []
    now = datetime.now()
    start_time = now - timedelta(minutes=15)
    
    # Shifted distribution: billing increases from 20% to 50% (30% shift, >20% threshold)
    category_distribution = {
        "appointment": 0.15,  # Down from 35%
        "prescription": 0.15,  # Down from 25%
        "billing": 0.50,  # Up from 20% (30% shift)
        "clinical_symptom": 0.10,
        "general": 0.10,
    }
    
    sample_queries = {
        "appointment": ["I need to schedule an appointment", "Can I book a visit?", "I want to cancel my appointment"],
        "prescription": ["Can I refill my medication?", "I need a prescription renewal", "Is my medication ready?"],
        "billing": ["What is my account balance?", "Can I set up a payment plan?", "I have a question about my bill"],
        "clinical_symptom": ["I have a headache", "I'm experiencing chest pain", "My blood pressure seems high"],
        "general": ["What are your office hours?", "Where is your clinic located?", "Do you accept walk-ins?"],
    }
    
    n_samples = 2000
    for category, pct in category_distribution.items():
        n_cat = int(n_samples * pct)
        queries = sample_queries[category]
        
        for i in range(n_cat):
            query_text = queries[i % len(queries)]
            timestamp = start_time + timedelta(
                seconds=np.random.uniform(0, (now - start_time).total_seconds())
            )
            
            # Lower confidence to simulate output drift
            confidence = round(np.random.beta(5, 4), 4)  # Lower confidence than baseline
            
            query_log = QueryLog(
                query=query_text,
                query_category=category,
                embedding=generate_simple_embedding(query_text),
                confidence_score=confidence,
                ai_response=f"Response to {category} query: {query_text}",
                timestamp=timestamp
            )
            test_db.add(query_log)
            queries_created.append(query_log)
    
    test_db.commit()
    return queries_created

@pytest.fixture(scope="function")
def sample_configuration(test_db: Session, config_service: ConfigurationService):
    """Create sample configuration with version history"""
    config_service.create_or_update_current_config(
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.75,
        confidence_threshold=0.70
    )
    
    v1 = config_service.snapshot_configuration(
        version_label="v1.0_baseline",
        performance_metrics={"accuracy": 0.95, "latency_ms": 120, "error_rate": 0.02}
    )
    config_service.mark_version_as_known_good(v1.id)
    
    v2 = config_service.snapshot_configuration(
        version_label="v1.1_optimized",
        performance_metrics={"accuracy": 0.93, "latency_ms": 100, "error_rate": 0.03}
    )
    
    test_db.commit()
    return {"v1": v1, "v2": v2}
