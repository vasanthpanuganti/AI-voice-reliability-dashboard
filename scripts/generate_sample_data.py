"""
Generate sample data for AI Pipeline Resilience Dashboard demo.
Creates realistic healthcare query data demonstrating:
- Normal operation phase
- Drift/degradation phase with multiple alerts
- Rollback events
- Recovery phase
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
from backend.models.rollback import RollbackEvent, RollbackTriggerType, RollbackStatus
from backend.services.configuration_service import ConfigurationService

# Sample healthcare queries by category
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
    """Generate a deterministic embedding based on query text."""
    np.random.seed(hash(query) % (2**32))
    embedding = np.random.randn(dimension).astype(float)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def create_queries_for_phase(db, phase_config, phase_name):
    """Create queries for a specific phase of the demo."""
    print(f"  Creating {phase_config['n_samples']} queries for {phase_name}...")
    
    queries_created = 0
    for category, pct in phase_config['distribution'].items():
        n_cat = int(phase_config['n_samples'] * pct)
        queries = SAMPLE_QUERIES[category]
        
        for i in range(n_cat):
            query_text = queries[i % len(queries)]
            if i % 3 == 0:
                query_text = query_text.lower()
            elif i % 5 == 0:
                query_text = query_text + " please"
            
            timestamp = phase_config['start_time'] + timedelta(
                seconds=np.random.uniform(0, (phase_config['end_time'] - phase_config['start_time']).total_seconds())
            )
            
            confidence = round(np.random.beta(
                phase_config['confidence_alpha'], 
                phase_config['confidence_beta']
            ), 4)
            
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
    return queries_created


def create_all_query_data(db):
    """Create query data showing the full lifecycle: normal -> drift -> rollback -> recovery"""
    now = datetime.now()
    
    # Phase 1: Normal operation (7-4 days ago) - Baseline
    phase1 = {
        'n_samples': 4000,
        'start_time': now - timedelta(days=7),
        'end_time': now - timedelta(days=4),
        'distribution': {
            "appointment": 0.35,
            "prescription": 0.25,
            "billing": 0.20,
            "clinical_symptom": 0.10,
            "general": 0.10,
        },
        'confidence_alpha': 8,  # High confidence
        'confidence_beta': 2
    }
    
    # Phase 2: Early drift (4-2 days ago) - Distribution starting to shift
    phase2 = {
        'n_samples': 2000,
        'start_time': now - timedelta(days=4),
        'end_time': now - timedelta(days=2),
        'distribution': {
            "appointment": 0.28,
            "prescription": 0.22,
            "billing": 0.30,  # Starting to increase
            "clinical_symptom": 0.10,
            "general": 0.10,
        },
        'confidence_alpha': 6,  # Slightly lower confidence
        'confidence_beta': 3
    }
    
    # Phase 3: Significant drift (2 days - 6 hours ago) - Major degradation
    phase3 = {
        'n_samples': 2500,
        'start_time': now - timedelta(days=2),
        'end_time': now - timedelta(hours=6),
        'distribution': {
            "appointment": 0.15,
            "prescription": 0.15,
            "billing": 0.50,  # Major shift
            "clinical_symptom": 0.10,
            "general": 0.10,
        },
        'confidence_alpha': 4,  # Lower confidence - degradation
        'confidence_beta': 5
    }
    
    # Phase 4: Post-rollback recovery (last 6 hours) - Returning to normal
    phase4 = {
        'n_samples': 1500,
        'start_time': now - timedelta(hours=6),
        'end_time': now,
        'distribution': {
            "appointment": 0.32,
            "prescription": 0.24,
            "billing": 0.24,  # Normalizing after rollback
            "clinical_symptom": 0.10,
            "general": 0.10,
        },
        'confidence_alpha': 7,  # Confidence recovering
        'confidence_beta': 2
    }
    
    total = 0
    total += create_queries_for_phase(db, phase1, "Phase 1: Normal Operation")
    total += create_queries_for_phase(db, phase2, "Phase 2: Early Drift")
    total += create_queries_for_phase(db, phase3, "Phase 3: Significant Drift")
    total += create_queries_for_phase(db, phase4, "Phase 4: Post-Rollback Recovery")
    
    return total


def create_drift_metrics_timeline(db):
    """Create drift metrics showing the full timeline with degradation and recovery."""
    print("Creating drift metrics timeline...")
    now = datetime.now()
    
    metrics_timeline = [
        # Phase 1: Normal operation (7-4 days ago)
        {"days_ago": 7, "hours_ago": 0, "psi": 0.04, "ks_p": 0.92, "js": 0.02, "sample_size": 500},
        {"days_ago": 6, "hours_ago": 12, "psi": 0.05, "ks_p": 0.88, "js": 0.03, "sample_size": 520},
        {"days_ago": 6, "hours_ago": 0, "psi": 0.04, "ks_p": 0.85, "js": 0.03, "sample_size": 480},
        {"days_ago": 5, "hours_ago": 12, "psi": 0.06, "ks_p": 0.82, "js": 0.04, "sample_size": 510},
        {"days_ago": 5, "hours_ago": 0, "psi": 0.05, "ks_p": 0.80, "js": 0.03, "sample_size": 490},
        {"days_ago": 4, "hours_ago": 12, "psi": 0.07, "ks_p": 0.78, "js": 0.05, "sample_size": 530},
        {"days_ago": 4, "hours_ago": 0, "psi": 0.06, "ks_p": 0.75, "js": 0.04, "sample_size": 500},
        
        # Phase 2: Early drift (4-2 days ago) - Warnings start
        {"days_ago": 3, "hours_ago": 18, "psi": 0.10, "ks_p": 0.65, "js": 0.07, "sample_size": 520},
        {"days_ago": 3, "hours_ago": 12, "psi": 0.13, "ks_p": 0.45, "js": 0.09, "sample_size": 540},
        {"days_ago": 3, "hours_ago": 6, "psi": 0.16, "ks_p": 0.08, "js": 0.11, "sample_size": 550},  # First warning
        {"days_ago": 3, "hours_ago": 0, "psi": 0.18, "ks_p": 0.04, "js": 0.13, "sample_size": 560},  # More warnings
        {"days_ago": 2, "hours_ago": 18, "psi": 0.20, "ks_p": 0.03, "js": 0.15, "sample_size": 580},
        {"days_ago": 2, "hours_ago": 12, "psi": 0.23, "ks_p": 0.015, "js": 0.17, "sample_size": 600},
        
        # Phase 3: Critical drift (2 days - 6 hours ago) - Critical/Emergency alerts
        {"days_ago": 2, "hours_ago": 6, "psi": 0.26, "ks_p": 0.008, "js": 0.21, "sample_size": 620},  # Critical
        {"days_ago": 2, "hours_ago": 0, "psi": 0.30, "ks_p": 0.005, "js": 0.24, "sample_size": 640},
        {"days_ago": 1, "hours_ago": 18, "psi": 0.35, "ks_p": 0.002, "js": 0.27, "sample_size": 660},
        {"days_ago": 1, "hours_ago": 12, "psi": 0.42, "ks_p": 0.0008, "js": 0.32, "sample_size": 680},  # Emergency
        {"days_ago": 1, "hours_ago": 6, "psi": 0.45, "ks_p": 0.0005, "js": 0.35, "sample_size": 700},  # Peak degradation
        
        # Rollback executed here (6 hours ago)
        
        # Phase 4: Recovery after rollback (last 6 hours) - but with some lingering issues
        {"days_ago": 0, "hours_ago": 5, "psi": 0.22, "ks_p": 0.02, "js": 0.18, "sample_size": 400},  # Immediate improvement
        {"days_ago": 0, "hours_ago": 4, "psi": 0.18, "ks_p": 0.04, "js": 0.14, "sample_size": 420},  # Still warning
        {"days_ago": 0, "hours_ago": 3, "psi": 0.16, "ks_p": 0.06, "js": 0.11, "sample_size": 440},  # Warning
        {"days_ago": 0, "hours_ago": 2, "psi": 0.19, "ks_p": 0.035, "js": 0.13, "sample_size": 460},  # Small spike - warning
        {"days_ago": 0, "hours_ago": 1, "psi": 0.17, "ks_p": 0.045, "js": 0.12, "sample_size": 480},  # Warning
        {"days_ago": 0, "hours_ago": 0, "psi": 0.16, "ks_p": 0.048, "js": 0.11, "sample_size": 500},  # Current - warning level
    ]
    
    metric_ids = []
    for m in metrics_timeline:
        timestamp = now - timedelta(days=m["days_ago"], hours=m["hours_ago"])
        window_start = timestamp - timedelta(minutes=15)
        
        drift_metric = DriftMetric(
            metric_type=MetricType.INPUT_DRIFT,
            psi_score=m["psi"],
            ks_statistic=0.15 + (m["psi"] * 0.5),
            ks_p_value=m["ks_p"],
            js_divergence=m["js"],
            window_start=window_start,
            window_end=timestamp,
            sample_size=m["sample_size"],
            timestamp=timestamp
        )
        db.add(drift_metric)
        db.flush()
        metric_ids.append((drift_metric.id, m, timestamp))
    
    db.commit()
    print(f"  Created {len(metrics_timeline)} drift metric records")
    return metric_ids


def create_alerts_timeline(db, metric_ids):
    """Create 15-20 alerts showing the progression of issues and recovery."""
    print("Creating alerts timeline...")
    
    alerts_config = []
    
    for metric_id, m, timestamp in metric_ids:
        # PSI alerts
        if m["psi"] >= 0.40:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.INPUT_DRIFT,
                "metric_name": "psi_score",
                "metric_value": m["psi"],
                "threshold_value": 0.40,
                "severity": DriftSeverity.EMERGENCY,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 5 else "active",
                "timestamp": timestamp
            })
        elif m["psi"] >= 0.25:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.INPUT_DRIFT,
                "metric_name": "psi_score",
                "metric_value": m["psi"],
                "threshold_value": 0.25,
                "severity": DriftSeverity.CRITICAL,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 5 else "active",
                "timestamp": timestamp
            })
        elif m["psi"] >= 0.15:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.INPUT_DRIFT,
                "metric_name": "psi_score",
                "metric_value": m["psi"],
                "threshold_value": 0.15,
                "severity": DriftSeverity.WARNING,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 4 else "active",
                "timestamp": timestamp
            })
        
        # KS p-value alerts (lower is worse)
        if m["ks_p"] <= 0.001:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.OUTPUT_DRIFT,
                "metric_name": "ks_p_value",
                "metric_value": m["ks_p"],
                "threshold_value": 0.001,
                "severity": DriftSeverity.EMERGENCY,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 5 else "active",
                "timestamp": timestamp
            })
        elif m["ks_p"] <= 0.01:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.OUTPUT_DRIFT,
                "metric_name": "ks_p_value",
                "metric_value": m["ks_p"],
                "threshold_value": 0.01,
                "severity": DriftSeverity.CRITICAL,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 5 else "active",
                "timestamp": timestamp
            })
        elif m["ks_p"] <= 0.05:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.OUTPUT_DRIFT,
                "metric_name": "ks_p_value",
                "metric_value": m["ks_p"],
                "threshold_value": 0.05,
                "severity": DriftSeverity.WARNING,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 4 else "active",
                "timestamp": timestamp
            })
        
        # JS divergence alerts
        if m["js"] >= 0.30:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.EMBEDDING_DRIFT,
                "metric_name": "js_divergence",
                "metric_value": m["js"],
                "threshold_value": 0.30,
                "severity": DriftSeverity.EMERGENCY,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 5 else "active",
                "timestamp": timestamp
            })
        elif m["js"] >= 0.20:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.EMBEDDING_DRIFT,
                "metric_name": "js_divergence",
                "metric_value": m["js"],
                "threshold_value": 0.20,
                "severity": DriftSeverity.CRITICAL,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 5 else "active",
                "timestamp": timestamp
            })
        elif m["js"] >= 0.10:
            alerts_config.append({
                "metric_id": metric_id,
                "metric_type": MetricType.EMBEDDING_DRIFT,
                "metric_name": "js_divergence",
                "metric_value": m["js"],
                "threshold_value": 0.10,
                "severity": DriftSeverity.WARNING,
                "status": "resolved" if m["days_ago"] > 0 or m["hours_ago"] > 4 else "active",
                "timestamp": timestamp
            })
    
    for alert_data in alerts_config:
        alert = DriftAlert(
            metric_type=alert_data["metric_type"],
            metric_name=alert_data["metric_name"],
            metric_value=alert_data["metric_value"],
            threshold_value=alert_data["threshold_value"],
            severity=alert_data["severity"],
            status=alert_data["status"],
            drift_metric_id=alert_data["metric_id"],
            created_at=alert_data["timestamp"]
        )
        db.add(alert)
    
    db.commit()
    
    active_count = len([a for a in alerts_config if a["status"] == "active"])
    resolved_count = len([a for a in alerts_config if a["status"] == "resolved"])
    print(f"  Created {len(alerts_config)} alerts ({active_count} active, {resolved_count} resolved)")
    return len(alerts_config)


def create_configuration_versions(db):
    """Create configuration versions showing the problematic version and rollback."""
    print("Creating configuration versions...")
    
    config_service = ConfigurationService(db)
    now = datetime.now()
    
    # Create current configuration
    config_service.create_or_update_current_config(
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.75,
        confidence_threshold=0.70
    )
    
    # v1.0 - Original baseline (known good)
    v1 = config_service.snapshot_configuration(
        version_label="v1.0_baseline",
        performance_metrics={"accuracy": 0.95, "latency_ms": 120, "error_rate": 0.02}
    )
    config_service.mark_version_as_known_good(v1.id)
    
    # v1.1 - Optimized version (known good)
    v2 = config_service.snapshot_configuration(
        version_label="v1.1_optimized",
        performance_metrics={"accuracy": 0.94, "latency_ms": 95, "error_rate": 0.025}
    )
    config_service.mark_version_as_known_good(v2.id)
    
    # v1.2 - Problematic version that caused drift
    v3 = config_service.snapshot_configuration(
        version_label="v1.2_experimental",
        performance_metrics={"accuracy": 0.78, "latency_ms": 85, "error_rate": 0.15}
    )
    
    # v1.1_restored - Restored after rollback (current)
    v4 = config_service.snapshot_configuration(
        version_label="v1.1_restored",
        performance_metrics={"accuracy": 0.93, "latency_ms": 98, "error_rate": 0.03}
    )
    config_service.mark_version_as_known_good(v4.id)
    
    db.commit()
    print("  Created 4 configuration versions")
    return v1.id, v2.id, v3.id, v4.id


def create_rollback_events(db, version_ids):
    """Create rollback events showing the automated recovery."""
    print("Creating rollback events...")
    
    v1_id, v2_id, v3_id, v4_id = version_ids
    now = datetime.now()
    
    # Rollback event triggered by emergency alert
    rollback = RollbackEvent(
        trigger_type=RollbackTriggerType.AUTOMATED,
        trigger_reason="Automated rollback triggered by EMERGENCY drift alert: psi_score exceeded 0.40 threshold",
        restored_version_id=v2_id,  # Rolling back to v1.1_optimized
        previous_version_id=v3_id,  # From v1.2_experimental
        status=RollbackStatus.SUCCESS,
        executed_by="system",
        executed_at=now - timedelta(hours=6),
        components_restored=["embedding_model", "similarity_threshold", "confidence_threshold"],
        components_failed=[],
        verification_metrics={"psi_after": 0.18, "status": "improved"}
    )
    db.add(rollback)
    db.commit()
    print("  Created 1 rollback event")


def clear_existing_data(db):
    """Clear existing data from database."""
    print("Clearing existing data...")
    
    db.query(DriftAlert).delete()
    db.query(DriftMetric).delete()
    db.query(RollbackEvent).delete()
    db.query(ConfigVersion).delete()
    db.query(Configuration).delete()
    db.query(QueryLog).delete()
    db.commit()
    print("  Cleared all existing data")


def main():
    """Generate all sample data demonstrating the full drift-rollback lifecycle."""
    print("=" * 70)
    print("AI Pipeline Resilience Dashboard - Sample Data Generator")
    print("=" * 70)
    print("\nThis script creates demo data showing:")
    print("  1. Normal operation phase (baseline)")
    print("  2. Early drift detection (warnings)")
    print("  3. Critical degradation (emergency alerts)")
    print("  4. Automated rollback execution")
    print("  5. System recovery after rollback")
    print("")
    
    # Initialize database
    print("Step 1: Initializing database...")
    init_db()
    
    db = SessionLocal()
    
    try:
        # Clear existing data
        print("\nStep 2: Clearing existing data...")
        clear_existing_data(db)
        
        # Create query data for all phases
        print("\nStep 3: Creating query data (4 phases)...")
        total_queries = create_all_query_data(db)
        
        # Create configuration versions
        print("\nStep 4: Creating configuration versions...")
        version_ids = create_configuration_versions(db)
        
        # Create drift metrics timeline
        print("\nStep 5: Creating drift metrics timeline...")
        metric_ids = create_drift_metrics_timeline(db)
        
        # Create alerts
        print("\nStep 6: Creating alerts...")
        alert_count = create_alerts_timeline(db, metric_ids)
        
        # Create rollback events
        print("\nStep 7: Creating rollback events...")
        create_rollback_events(db, version_ids)
        
        print("\n" + "=" * 70)
        print("Sample data generation complete!")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  - Total queries: {total_queries:,}")
        print(f"  - Drift metrics: {len(metric_ids)} records")
        print(f"  - Alerts: {alert_count} total")
        print(f"  - Configuration versions: 4")
        print(f"  - Rollback events: 1")
        print(f"\nTimeline:")
        print(f"  - Days 7-4: Normal operation")
        print(f"  - Days 4-2: Drift begins (warnings)")
        print(f"  - Days 2-0.25: Critical degradation (emergency)")
        print(f"  - Last 6 hours: Recovery after rollback")
        print(f"\nNext steps:")
        print(f"  1. Start API: python run_api.py")
        print(f"  2. Open: http://localhost:8000")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
