"""Drift metrics and alerts models"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Enum, ForeignKey
from sqlalchemy.sql import func
from backend.database import Base
import enum

class DriftSeverity(str, enum.Enum):
    """Alert severity levels"""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(str, enum.Enum):
    """Types of drift metrics"""
    INPUT_DRIFT = "input_drift"
    OUTPUT_DRIFT = "output_drift"
    EMBEDDING_DRIFT = "embedding_drift"

class DriftMetric(Base):
    """Stores computed drift metrics over time"""
    __tablename__ = "drift_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Metric type
    metric_type = Column(Enum(MetricType), nullable=False, index=True)
    
    # PSI metrics
    psi_score = Column(Float)
    
    # KS test metrics
    ks_statistic = Column(Float)
    ks_p_value = Column(Float)
    
    # Jensen-Shannon divergence
    js_divergence = Column(Float)

    # Wasserstein distance (Earth Mover's Distance for embeddings)
    wasserstein_distance = Column(Float)

    # Window information
    window_start = Column(DateTime(timezone=True))
    window_end = Column(DateTime(timezone=True))
    sample_size = Column(Integer)

class DriftAlert(Base):
    """Alerts generated when drift thresholds are breached"""
    __tablename__ = "drift_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Alert details
    metric_type = Column(Enum(MetricType), nullable=False, index=True)
    metric_name = Column(String(100))  # e.g., "psi_score", "ks_p_value"
    metric_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    severity = Column(Enum(DriftSeverity), nullable=False, index=True)
    
    # Status
    status = Column(String(20), default="active", index=True)  # active, dismissed, resolved
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    dismissed_at = Column(DateTime(timezone=True))
    dismissed_by = Column(String(100))
    
    # Link to drift metric
    drift_metric_id = Column(Integer, ForeignKey("drift_metrics.id"), nullable=True)
