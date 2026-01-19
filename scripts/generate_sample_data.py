"""
Generate sample data for AI Pipeline Resilience Dashboard demo.
Creates realistic healthcare query data to demonstrate drift detection and rollback.
No external dependencies (Kaggle) required.
"""
import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import SessionLocal, init_db
from backend.models.query_log import QueryLog
from backend.models.drift_metrics import DriftMetric, DriftAlert, MetricType, DriftSeverity
from backend.models.configuration import Configuration, ConfigVersion
from backend.services.configuration_service import ConfigurationService
from backend.config import settings

# Sample healthcare queries by category (realistic for appointment booking system)
SAMPLE_QUERIES = {
    "appointment": [
        "I need to schedule an appointment with Dr. Smith",
        "Can I book a visit for next Monday?",
        "I want to cancel my appointment on Friday",
        "How do I reschedule my checkup?",
        "Is there an available slot this week?",
        "I need to see a specialist urgently",
        "Can I get an appointment today?",
        "What times are available for Dr. Johnson?",
        "I need to change my appointment time",
        "Book me for the earliest available slot",
        "I want to schedule a follow-up visit",
        "Can I get a same-day appointment?",
        "I need to cancel due to illness",
        "Is the doctor available on weekends?",
        "Schedule me for a routine checkup",
    ],
    "prescription": [
        "Can I refill my blood pressure medication?",
        "I need a prescription renewal",
        "Is my medication ready for pickup?",
        "How do I get a refill on my diabetes medicine?",
        "Can you send my prescription to a different pharmacy?",
        "I ran out of my heart medication",
        "When will my prescription be ready?",
        "I need to refill my inhaler",
        "Can I get a 90-day supply of my medication?",
        "Is there a generic version of my prescription?",
        "I need to update my pharmacy information",
        "My prescription expired, can I get a new one?",
        "Can the doctor call in a refill?",
        "I need my allergy medication renewed",
        "How much will my prescription cost?",
    ],
    "billing": [
        "What is my current account balance?",
        "Can I set up a payment plan?",
        "I have a question about my bill",
        "What is my insurance coverage for this visit?",
        "Can I pay my bill online?",
        "I received an incorrect charge",
        "What are my payment options?",
        "Does my insurance cover this procedure?",
        "I need an itemized statement",
        "Can you verify my insurance information?",
        "I need to update my payment method",
        "What is my copay for this visit?",
        "Can I get a receipt for my payment?",
        "Why was I charged this amount?",
        "Is there a discount for paying in full?",
    ],
    "clinical_symptom": [
        "I have a persistent headache",
        "I'm experiencing chest pain",
        "My blood pressure seems high",
        "I have a fever and cough",
        "I'm having trouble sleeping",
        "I feel dizzy when I stand up",
        "I have a rash that won't go away",
        "My joints are aching",
        "I've been having stomach problems",
        "I'm experiencing shortness of breath",
        "I have a sore throat",
        "My vision has been blurry",
        "I've been feeling very tired lately",
        "I have numbness in my hands",
        "I'm experiencing back pain",
    ],
    "general": [
        "What are your office hours?",
        "Where is your clinic located?",
        "Do you accept walk-ins?",
        "What should I bring to my appointment?",
        "How do I access my medical records?",
        "Can I get a copy of my test results?",
        "What is the doctor's phone number?",
        "How do I reach after-hours care?",
        "What vaccinations do you offer?",
        "Do you have parking available?",
        "What forms do I need to fill out?",
        "Can I bring someone with me?",
        "What is your cancellation policy?",
        "Do you offer telehealth visits?",
        "How long is the typical wait time?",
    ],
}

def generate_simple_embedding(query: str, dimension: int = 384) -> list:
    """
    Generate a simple deterministic embedding based on query text.
    Uses hash-based approach for consistency without external models.
    """
    np.random.seed(hash(query) % (2**32))
    embedding = np.random.randn(dimension).astype(float)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

def create_baseline_data(db, n_samples: int = 1000):
    """Create baseline data representing normal operation (Week 1)"""
    print(f"Creating {n_samples} baseline queries...")
    
    # Balanced distribution for baseline (normal operation)
    category_distribution = {
        "appointment": 0.35,
        "prescription": 0.25,
        "billing": 0.20,
        "clinical_symptom": 0.10,
        "general": 0.10,
    }
    
    # Generate timestamps for past 7 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    queries_created = 0
    for category, pct in category_distribution.items():
        n_cat = int(n_samples * pct)
        queries = SAMPLE_QUERIES[category]
        
        for i in range(n_cat):
            query_text = queries[i % len(queries)]
            # Add slight variation
            if i % 3 == 0:
                query_text = query_text.lower()
            elif i % 5 == 0:
                query_text = query_text + " please"
            
            # Random timestamp in baseline period
            timestamp = start_time + timedelta(
                seconds=np.random.uniform(0, (end_time - start_time).total_seconds())
            )
            
            query_log = QueryLog(
                query=query_text,
                query_category=category,
                embedding=generate_simple_embedding(query_text),
                confidence_score=str(round(np.random.beta(8, 2), 4)),  # High confidence (good performance)
                ai_response=f"Response to {category} query: {query_text[:50]}...",
                timestamp=timestamp
            )
            db.add(query_log)
            queries_created += 1
    
    db.commit()
    print(f"Created {queries_created} baseline queries")
    return queries_created

def create_drifted_data(db, n_samples: int = 200):
    """Create drifted data representing distribution shift (current window)"""
    print(f"Creating {n_samples} drifted queries to simulate drift...")
    
    # Shifted distribution (more billing queries - simulates drift)
    shifted_distribution = {
        "appointment": 0.15,  # Decreased from 35%
        "prescription": 0.15,  # Decreased from 25%
        "billing": 0.50,       # Increased from 20% - Major shift!
        "clinical_symptom": 0.10,
        "general": 0.10,
    }
    
    # Generate timestamps for recent window (last 15 minutes)
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=15)
    
    queries_created = 0
    for category, pct in shifted_distribution.items():
        n_cat = int(n_samples * pct)
        queries = SAMPLE_QUERIES[category]
        
        for i in range(n_cat):
            query_text = queries[i % len(queries)]
            
            # Random timestamp in current window
            timestamp = start_time + timedelta(
                seconds=np.random.uniform(0, (end_time - start_time).total_seconds())
            )
            
            # Lower confidence for drifted data (simulates degradation)
            confidence = round(np.random.beta(5, 4), 4)  # Lower confidence
            
            query_log = QueryLog(
                query=query_text,
                query_category=category,
                embedding=generate_simple_embedding(query_text),
                confidence_score=str(confidence),
                ai_response=f"Response to {category} query: {query_text[:50]}...",
                timestamp=timestamp
            )
            db.add(query_log)
            queries_created += 1
    
    db.commit()
    print(f"Created {queries_created} drifted queries")
    return queries_created

def create_sample_drift_metrics(db):
    """Create sample drift metrics to show in dashboard"""
    print("Creating sample drift metrics...")
    
    now = datetime.now()
    
    # Create historical metrics (showing normal -> drift progression)
    metrics_data = [
        # Day 1-6: Normal operation
        {"days_ago": 6, "psi": 0.05, "ks_p": 0.85, "js": 0.03},
        {"days_ago": 5, "psi": 0.06, "ks_p": 0.78, "js": 0.04},
        {"days_ago": 4, "psi": 0.04, "ks_p": 0.82, "js": 0.03},
        {"days_ago": 3, "psi": 0.07, "ks_p": 0.75, "js": 0.05},
        {"days_ago": 2, "psi": 0.08, "ks_p": 0.70, "js": 0.06},
        {"days_ago": 1, "psi": 0.12, "ks_p": 0.45, "js": 0.09},
        # Recent: Drift detected
        {"hours_ago": 6, "psi": 0.18, "ks_p": 0.03, "js": 0.15},
        {"hours_ago": 3, "psi": 0.22, "ks_p": 0.01, "js": 0.18},
        {"hours_ago": 1, "psi": 0.28, "ks_p": 0.005, "js": 0.22},
    ]
    
    for m in metrics_data:
        if "days_ago" in m:
            timestamp = now - timedelta(days=m["days_ago"])
        else:
            timestamp = now - timedelta(hours=m["hours_ago"])
        
        window_start = timestamp - timedelta(minutes=15)
        
        drift_metric = DriftMetric(
            metric_type=MetricType.INPUT_DRIFT,
            psi_score=m["psi"],
            ks_statistic=0.15,
            ks_p_value=m["ks_p"],
            js_divergence=m["js"],
            window_start=window_start,
            window_end=timestamp,
            sample_size=200,
            timestamp=timestamp
        )
        db.add(drift_metric)
    
    db.commit()
    print("Created sample drift metrics history")

def create_sample_alerts(db):
    """Create sample alerts to demonstrate alerting system"""
    print("Creating sample alerts...")
    
    now = datetime.now()
    
    # Get latest drift metric for linking
    latest_metric = db.query(DriftMetric).order_by(DriftMetric.timestamp.desc()).first()
    metric_id = latest_metric.id if latest_metric else None
    
    # Create alerts
    alerts = [
        {
            "metric_type": MetricType.INPUT_DRIFT,
            "metric_name": "psi_score",
            "metric_value": 0.28,
            "threshold_value": 0.25,
            "severity": DriftSeverity.CRITICAL,
            "status": "active",
        },
        {
            "metric_type": MetricType.OUTPUT_DRIFT,
            "metric_name": "ks_p_value",
            "metric_value": 0.005,
            "threshold_value": 0.01,
            "severity": DriftSeverity.CRITICAL,
            "status": "active",
        },
    ]
    
    for alert_data in alerts:
        alert = DriftAlert(
            metric_type=alert_data["metric_type"],
            metric_name=alert_data["metric_name"],
            metric_value=alert_data["metric_value"],
            threshold_value=alert_data["threshold_value"],
            severity=alert_data["severity"],
            status=alert_data["status"],
            drift_metric_id=metric_id,
            created_at=now - timedelta(minutes=5)
        )
        db.add(alert)
    
    db.commit()
    print("Created sample alerts")

def create_configuration_versions(db):
    """Create configuration with version history"""
    print("Creating configuration versions...")
    
    config_service = ConfigurationService(db)
    
    # Create current configuration
    config_service.create_or_update_current_config(
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.75,
        confidence_threshold=0.70
    )
    
    # Create baseline version (known good)
    v1 = config_service.snapshot_configuration(
        version_label="v1.0_baseline",
        performance_metrics={"accuracy": 0.95, "latency_ms": 120, "error_rate": 0.02}
    )
    config_service.mark_version_as_known_good(v1.id)
    
    # Create second version
    v2 = config_service.snapshot_configuration(
        version_label="v1.1_optimized",
        performance_metrics={"accuracy": 0.93, "latency_ms": 100, "error_rate": 0.03}
    )
    
    # Create current version (potentially problematic)
    v3 = config_service.snapshot_configuration(
        version_label="v1.2_current",
        performance_metrics={"accuracy": 0.88, "latency_ms": 95, "error_rate": 0.08}
    )
    
    db.commit()
    print("Created configuration versions")

def clear_existing_data(db):
    """Clear existing data from database"""
    print("Clearing existing data...")
    
    from backend.models.rollback import RollbackEvent
    
    db.query(DriftAlert).delete()
    db.query(DriftMetric).delete()
    db.query(RollbackEvent).delete()
    db.query(ConfigVersion).delete()
    db.query(Configuration).delete()
    db.query(QueryLog).delete()
    db.commit()
    print("Cleared existing data")

def main():
    """Generate all sample data"""
    print("=" * 60)
    print("AI Pipeline Resilience Dashboard - Sample Data Generator")
    print("=" * 60)
    print("\nThis script creates realistic demo data to demonstrate:")
    print("  - Drift detection (PSI, KS Test, JS Divergence)")
    print("  - Alert generation and explanations")
    print("  - Configuration versioning and rollback")
    print("")
    
    # Initialize database
    print("1. Initializing SQLite database...")
    init_db()
    print("   Database initialized")
    
    db = SessionLocal()
    
    try:
        # Clear existing data
        print("\n2. Clearing existing data...")
        clear_existing_data(db)
        
        # Create baseline data (normal operation)
        print("\n3. Creating baseline data (Week 1 - normal operation)...")
        baseline_count = create_baseline_data(db, n_samples=1000)
        
        # Create drifted data (recent window with distribution shift)
        print("\n4. Creating drifted data (simulating 30% distribution shift)...")
        drift_count = create_drifted_data(db, n_samples=200)
        
        # Create configuration versions
        print("\n5. Creating configuration versions...")
        create_configuration_versions(db)
        
        # Create drift metrics history
        print("\n6. Creating drift metrics history...")
        create_sample_drift_metrics(db)
        
        # Create sample alerts
        print("\n7. Creating sample alerts...")
        create_sample_alerts(db)
        
        print("\n" + "=" * 60)
        print("Sample data generation complete!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  - Baseline queries: {baseline_count}")
        print(f"  - Drifted queries: {drift_count}")
        print(f"  - Configuration versions: 3")
        print(f"  - Drift metrics history: 9 records")
        print(f"  - Active alerts: 2")
        print(f"\nDatabase: SQLite (ai_resilience.db)")
        print(f"\nNext steps:")
        print(f"  1. Start API: python run_api.py")
        print(f"  2. Start Dashboard: python run_dashboard.py")
        print(f"  3. Open: http://localhost:8501")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
