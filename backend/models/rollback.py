"""Rollback events model"""
from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, JSON, Text
from sqlalchemy.sql import func
from backend.database import Base
import enum

class RollbackTriggerType(str, enum.Enum):
    """Types of rollback triggers"""
    AUTOMATED = "automated"
    MANUAL = "manual"

class RollbackStatus(str, enum.Enum):
    """Rollback execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"

class RollbackEvent(Base):
    """Audit log of all rollback events"""
    __tablename__ = "rollback_events"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Trigger information
    trigger_type = Column(Enum(RollbackTriggerType), nullable=False, index=True)
    trigger_reason = Column(Text)  # e.g., "Critical drift alert", "Manual intervention"
    alert_id = Column(Integer, ForeignKey("drift_alerts.id"), nullable=True)  # If triggered by alert
    
    # Version information
    previous_version_id = Column(Integer, ForeignKey("config_versions.id"), nullable=True)
    restored_version_id = Column(Integer, ForeignKey("config_versions.id"), nullable=False)
    
    # Previous configuration (snapshot at rollback time)
    previous_config = Column(JSON)  # Full config snapshot
    
    # Execution details
    executed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    executed_by = Column(String(100))  # User ID or "system"
    status = Column(Enum(RollbackStatus), default=RollbackStatus.PENDING, index=True)
    
    # Rollback result
    components_restored = Column(JSON)  # List of component names restored
    components_failed = Column(JSON)  # List of components that failed to restore
    error_message = Column(Text)
    
    # Verification metrics after rollback
    verification_metrics = Column(JSON)  # Drift metrics after rollback
