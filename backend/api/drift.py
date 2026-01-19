"""FastAPI endpoints for drift detection"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from backend.database import get_db
from backend.services.drift_detection_service import DriftDetectionService
from backend.models.drift_metrics import DriftAlert, DriftMetric
from pydantic import BaseModel

router = APIRouter(prefix="/api/drift", tags=["drift"])

class DriftMetricsResponse(BaseModel):
    """Response model for drift metrics"""
    psi_score: Optional[float]
    ks_statistic: Optional[float]
    ks_p_value: Optional[float]
    js_divergence: Optional[float]
    timestamp: datetime
    sample_size: int
    
    class Config:
        from_attributes = True

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
    
    class Config:
        from_attributes = True

@router.get("/metrics", response_model=DriftMetricsResponse)
def get_current_metrics(db: Session = Depends(get_db)):
    """
    Get current drift metrics
    PRD Requirement DD-007: Real-time dashboard displaying current drift metrics
    
    Returns the most recent computed metrics from database, or computes new ones if needed.
    """
    from datetime import timedelta
    
    # Try to get most recent metrics from database (within last 5 minutes)
    recent_metric = db.query(DriftMetric).order_by(
        DriftMetric.timestamp.desc()
    ).first()
    
    # If we have recent metrics (within 5 minutes), return them
    if recent_metric and recent_metric.timestamp:
        time_diff = datetime.now() - recent_metric.timestamp.replace(tzinfo=None)
        if time_diff < timedelta(minutes=5):
            # Return cached metrics
            return DriftMetricsResponse(
                psi_score=recent_metric.psi_score,
                ks_statistic=recent_metric.ks_statistic,
                ks_p_value=recent_metric.ks_p_value,
                js_divergence=recent_metric.js_divergence,
                timestamp=recent_metric.timestamp,
                sample_size=recent_metric.sample_size or 0
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
        
        return drift_metric
    except Exception as e:
        # If computation fails, return the most recent metric we have (even if old)
        if recent_metric:
            return DriftMetricsResponse(
                psi_score=recent_metric.psi_score,
                ks_statistic=recent_metric.ks_statistic,
                ks_p_value=recent_metric.ks_p_value,
                js_divergence=recent_metric.js_divergence,
                timestamp=recent_metric.timestamp,
                sample_size=recent_metric.sample_size or 0
            )
        # If no metrics at all, return default
        return DriftMetricsResponse(
            psi_score=0.0,
            ks_statistic=None,
            ks_p_value=1.0,
            js_divergence=0.0,
            timestamp=datetime.now(),
            sample_size=0
        )

@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    status: Optional[str] = "active",
    severity: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get drift alerts, optionally filtered by status and severity"""
    query = db.query(DriftAlert)
    
    if status:
        query = query.filter(DriftAlert.status == status)
    if severity:
        query = query.filter(DriftAlert.severity == severity)
    
    alerts = query.order_by(DriftAlert.created_at.desc()).limit(100).all()
    return alerts

@router.get("/history")
def get_drift_history(limit: int = 100, db: Session = Depends(get_db)):
    """Get historical drift metrics"""
    metrics = db.query(DriftMetric).order_by(
        DriftMetric.timestamp.desc()
    ).limit(limit).all()
    
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
