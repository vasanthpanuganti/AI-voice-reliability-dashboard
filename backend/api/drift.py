"""FastAPI endpoints for drift detection"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime

from backend.database import get_db
from backend.services.drift_detection_service import DriftDetectionService
from backend.models.drift_metrics import DriftAlert, DriftMetric
from backend.models.query_log import QueryLog
from pydantic import BaseModel

router = APIRouter(prefix="/api/drift", tags=["drift"])

MAX_HISTORY_LIMIT = 1000  # Prevent memory exhaustion from unbounded queries

@router.post("/refresh-queries")
def refresh_recent_queries_endpoint(db: Session = Depends(get_db)):
    """
    Refresh recent queries to ensure active window is populated.
    This endpoint automatically generates queries in the last 15 minutes if needed.
    """
    try:
        from scripts.generate_sample_data import refresh_recent_queries
        from datetime import datetime, timedelta
        
        # Check current active queries
        cutoff = datetime.now() - timedelta(minutes=15)
        recent_count = db.query(QueryLog).filter(QueryLog.timestamp >= cutoff).count()
        
        target_count = 1500
        if recent_count < target_count:
            needed = target_count - recent_count
            refresh_recent_queries(db, n_samples=needed + 200)
            return {
                "status": "success",
                "message": f"Generated {needed + 200} new queries",
                "previous_count": recent_count,
                "new_count": recent_count + needed + 200
            }
        else:
            return {
                "status": "success",
                "message": "Sufficient queries already exist",
                "current_count": recent_count
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh queries: {str(e)}")

class DriftMetricsResponse(BaseModel):
    """Response model for drift metrics"""
    psi_score: Optional[float]
    ks_statistic: Optional[float]
    ks_p_value: Optional[float]
    js_divergence: Optional[float]
    wasserstein_distance: Optional[float] = None
    chi_square_statistic: Optional[float] = None
    chi_square_p_value: Optional[float] = None
    timestamp: datetime
    sample_size: int
    total_queries: Optional[int] = None
    baseline_size: Optional[int] = None
    computation_time_ms: Optional[float] = None
    
    model_config = {"from_attributes": True}

class AlertResponse(BaseModel):
    """Response model for alerts"""
    id: int
    metric_type: str
    metric_name: str
    metric_value: float
    threshold_value: float
    severity: str
    status: str
    created_at: datetime
    
    model_config = {"from_attributes": True}

@router.get("/metrics", response_model=DriftMetricsResponse)
def get_current_metrics(db: Session = Depends(get_db)):
    """
    Get current drift metrics
    PRD Requirement DD-007: Real-time dashboard displaying current drift metrics
    
    Returns the most recent computed metrics from database, or computes new ones if needed.
    """
    from datetime import timedelta
    
    # Get total query count for display
    total_queries = db.query(func.count(QueryLog.id)).scalar() or 0
    
    # Try to get most recent metrics from database (within last 5 minutes)
    recent_metric = db.query(DriftMetric).order_by(
        DriftMetric.timestamp.desc()
    ).first()
    
    # If we have recent metrics (within 5 minutes), return them
    if recent_metric and recent_metric.timestamp:
        time_diff = datetime.now() - recent_metric.timestamp.replace(tzinfo=None)
        if time_diff < timedelta(minutes=5):
            # Return cached metrics with total queries
            return DriftMetricsResponse(
                psi_score=recent_metric.psi_score,
                ks_statistic=recent_metric.ks_statistic,
                ks_p_value=recent_metric.ks_p_value,
                js_divergence=recent_metric.js_divergence,
                timestamp=recent_metric.timestamp,
                sample_size=recent_metric.sample_size or 0,
                total_queries=total_queries
            )
    
    # Otherwise, compute new metrics
    try:
        service = DriftDetectionService(db)
        drift_metric = service.compute_drift_metrics()
        alerts = service.check_thresholds_and_alert(drift_metric)
        
        # Check for automated rollback on critical/emergency alerts (PRD RB-004)
        for alert in alerts:
            if alert.severity.value in ["critical", "emergency"]:
                try:
                    from backend.services.rollback_service import RollbackService
                    rollback_service = RollbackService(db)
                    rollback_event = rollback_service.auto_rollback_on_alert(alert.id)
                    if rollback_event:
                        print(f"Automated rollback triggered by {alert.severity.value} alert {alert.id}")
                except Exception as e:
                    print(f"Warning: Auto-rollback failed for alert {alert.id}: {e}")
        
        # Return with total queries
        return DriftMetricsResponse(
            psi_score=drift_metric.psi_score,
            ks_statistic=drift_metric.ks_statistic,
            ks_p_value=drift_metric.ks_p_value,
            js_divergence=drift_metric.js_divergence,
            timestamp=drift_metric.timestamp,
            sample_size=drift_metric.sample_size or 0,
            total_queries=total_queries
        )
    except Exception as e:
        # If computation fails, return the most recent metric we have (even if old)
        if recent_metric:
            return DriftMetricsResponse(
                psi_score=recent_metric.psi_score,
                ks_statistic=recent_metric.ks_statistic,
                ks_p_value=recent_metric.ks_p_value,
                js_divergence=recent_metric.js_divergence,
                timestamp=recent_metric.timestamp,
                sample_size=recent_metric.sample_size or 0,
                total_queries=total_queries
            )
        # If no metrics at all, return default
        return DriftMetricsResponse(
            psi_score=0.0,
            ks_statistic=None,
            ks_p_value=1.0,
            js_divergence=0.0,
            timestamp=datetime.now(),
            sample_size=0,
            total_queries=total_queries
        )

@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    status: Optional[str] = "active",
    severity: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get drift alerts, optionally filtered by status and severity.
    Use status=all to get all alerts regardless of status.
    """
    query = db.query(DriftAlert)
    
    # Normalize and filter by status (skip filter if "all" is requested)
    if status:
        status_lower = status.lower()
        if status_lower != "all":
            query = query.filter(DriftAlert.status == status_lower)
    if severity:
        query = query.filter(DriftAlert.severity == severity.lower())
    
    alerts = query.order_by(DriftAlert.created_at.desc()).limit(100).all()
    return alerts

@router.get("/history")
def get_drift_history(limit: int = 100, db: Session = Depends(get_db)):
    """Get historical drift metrics"""
    safe_limit = min(max(1, limit), MAX_HISTORY_LIMIT)
    metrics = db.query(DriftMetric).order_by(
        DriftMetric.timestamp.desc()
    ).limit(safe_limit).all()
    
    return [
        {
            "id": m.id,
            "timestamp": m.timestamp,
            "psi_score": m.psi_score,
            "ks_p_value": m.ks_p_value,
            "js_divergence": m.js_divergence,
            "sample_size": m.sample_size
        }
        for m in metrics
    ]

@router.get("/alerts/{alert_id}/diagnostics")
def get_alert_diagnostics(alert_id: int, db: Session = Depends(get_db)):
    """
    Get detailed diagnostics for an alert explaining why it occurred.
    PRD Requirement: Users should understand what drift means and what's going wrong.
    """
    alert = db.query(DriftAlert).filter(DriftAlert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Get associated drift metric
    drift_metric = None
    if alert.drift_metric_id:
        drift_metric = db.query(DriftMetric).filter(DriftMetric.id == alert.drift_metric_id).first()
    
    if not drift_metric:
        raise HTTPException(status_code=404, detail="Drift metric not found for this alert")
    
    # Generate explanation
    service = DriftDetectionService(db)
    explanation = service.generate_alert_explanation(alert, drift_metric)
    
    return {
        "alert_id": alert.id,
        "alert": {
            "id": alert.id,
            "metric_type": alert.metric_type.value,
            "metric_name": alert.metric_name,
            "metric_value": alert.metric_value,
            "threshold_value": alert.threshold_value,
            "severity": alert.severity.value,
            "status": alert.status,
            "created_at": alert.created_at.isoformat()
        },
        "diagnostics": explanation
    }

@router.post("/alerts/{alert_id}/dismiss")
def dismiss_alert(alert_id: int, db: Session = Depends(get_db)):
    """Dismiss an alert"""
    alert = db.query(DriftAlert).filter(DriftAlert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.status = "dismissed"
    alert.dismissed_at = datetime.now()
    
    db.commit()
    
    return {"message": "Alert dismissed", "alert_id": alert_id}


@router.get("/segments")
def get_segment_drift(
    segment_by: str = "query_category",
    db: Session = Depends(get_db)
):
    """
    Get drift metrics broken down by segment (query category, department, etc.).
    
    PRD Requirement: Segment-level monitoring to catch drift affecting specific
    patient populations or query types even if aggregate metrics look fine.
    
    Args:
        segment_by: Field to segment by (query_category, department, patient_population)
    """
    valid_segments = ["query_category", "department", "patient_population"]
    if segment_by not in valid_segments:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid segment_by. Must be one of: {valid_segments}"
        )
    
    service = DriftDetectionService(db)
    return service.detect_segment_drift(segment_field=segment_by)


@router.get("/summary")
def get_drift_summary(db: Session = Depends(get_db)):
    """
    Get comprehensive drift summary for the dashboard.
    
    Returns:
        Summary with aggregate metrics, segment health, alerts, and recommendations
    """
    service = DriftDetectionService(db)
    return service.get_drift_summary()


@router.post("/check-triggers")
def check_automated_triggers(db: Session = Depends(get_db)):
    """
    Manually trigger a check of automated rollback triggers.
    
    Returns whether any triggers were activated and rollback executed.
    """
    from backend.services.rollback_service import RollbackService
    
    rollback_service = RollbackService(db)
    result = rollback_service.check_automated_triggers()
    
    if result:
        return {
            "triggered": True,
            "rollback_id": result.id,
            "trigger_reason": result.trigger_reason,
            "status": result.status.value if result.status else None
        }
    
    return {
        "triggered": False,
        "message": "No automated triggers activated"
    }


@router.get("/categorical")
def get_categorical_drift(
    category_field: str = "query_category",
    db: Session = Depends(get_db)
):
    """
    Get categorical drift analysis using Chi-Square test.
    
    OPTIMIZATION: Uses SQL aggregation for efficient category counting
    rather than loading all records into memory.
    
    Chi-Square test detects when the distribution of categories has changed
    significantly between baseline and current periods.
    
    Args:
        category_field: Field to analyze (query_category, department, patient_population)
    """
    valid_fields = ["query_category", "department", "patient_population"]
    if category_field not in valid_fields:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid category_field. Must be one of: {valid_fields}"
        )
    
    service = DriftDetectionService(db)
    return service.detect_categorical_drift(category_field=category_field)


@router.get("/comprehensive")
def get_comprehensive_drift_analysis(db: Session = Depends(get_db)):
    """
    Get comprehensive drift analysis including all metric types.
    
    Returns detailed results for:
    - Input drift (PSI)
    - Output drift (KS test)
    - Embedding drift (Wasserstein, JS divergence)
    - Categorical drift (Chi-Square)
    
    OPTIMIZATION: Computes all metrics in parallel for faster response.
    """
    service = DriftDetectionService(db)
    return service.compute_drift_metrics_comprehensive()


@router.get("/baseline-stats")
def get_baseline_statistics(db: Session = Depends(get_db)):
    """
    Get pre-computed baseline statistics.
    
    Returns cached histogram data, category counts, and summary statistics
    used for incremental drift computation.
    
    OPTIMIZATION: This endpoint exposes the baseline cache for debugging
    and verification purposes.
    """
    service = DriftDetectionService(db)
    stats = service.get_baseline_statistics()
    
    # Convert numpy arrays to lists for JSON serialization
    return {
        "query_length": stats.get("query_length", {}),
        "confidence_score": stats.get("confidence_score", {}),
        "category_counts": stats.get("category_counts", {}),
        "cache_ttl_hours": 24,
        "last_computed": service._baseline_stats_time.isoformat() if service._baseline_stats_time else None
    }


@router.post("/invalidate-cache")
def invalidate_baseline_cache(db: Session = Depends(get_db)):
    """
    Invalidate the baseline statistics cache.
    
    Call this when:
    - New baseline data is added
    - Baseline period configuration changes
    - After major data corrections
    """
    service = DriftDetectionService(db)
    service.invalidate_baseline_cache()
    
    return {
        "message": "Baseline cache invalidated",
        "note": "Next drift computation will recompute baseline statistics"
    }