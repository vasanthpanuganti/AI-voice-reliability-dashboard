"""Rollback Service - Handles configuration rollbacks with automated triggers"""
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import and_

from backend.models.configuration import ConfigVersion
from backend.models.drift_metrics import DriftAlert, DriftMetric, DriftSeverity
from backend.models.rollback import RollbackEvent, RollbackTriggerType, RollbackStatus
from backend.services.configuration_service import ConfigurationService


# Automated rollback trigger configuration
ROLLBACK_TRIGGERS = {
    "emergency_alert": {
        "enabled": True,
        "description": "Trigger rollback on EMERGENCY severity alerts",
        "cooldown_minutes": 30,  # Don't trigger again within this window
    },
    "sustained_critical": {
        "enabled": True,
        "description": "Trigger rollback after sustained CRITICAL alerts",
        "threshold_count": 3,  # Number of critical alerts
        "window_minutes": 15,  # Within this time window
        "cooldown_minutes": 60,
    },
    "error_rate_spike": {
        "enabled": True,
        "description": "Trigger rollback on error rate exceeding threshold",
        "threshold_percentage": 15,  # 15% error rate
        "cooldown_minutes": 30,
    },
    "confidence_collapse": {
        "enabled": True,
        "description": "Trigger rollback when average confidence drops significantly",
        "threshold_drop": 0.25,  # 25% drop from baseline
        "cooldown_minutes": 30,
    },
}


class RollbackService:
    """Service for executing configuration rollbacks"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = ConfigurationService(db)
    
    def execute_rollback(
        self,
        version_id: int,
        trigger_type: RollbackTriggerType = RollbackTriggerType.MANUAL,
        trigger_reason: Optional[str] = None,
        executed_by: Optional[str] = None,
        alert_id: Optional[int] = None
    ) -> RollbackEvent:
        """
        Execute atomic rollback to a specific configuration version.
        
        Args:
            version_id: Configuration version ID to rollback to
            trigger_type: Whether rollback is automated or manual
            trigger_reason: Reason for rollback
            executed_by: User ID or "system"
            alert_id: Alert ID that triggered this rollback (if applicable)
            
        Returns:
            RollbackEvent with execution details
        """
        # Get current config before rollback
        current_config = self.config_service.get_current_config()
        previous_version_id = None
        previous_config_snapshot = None
        
        if current_config:
            # Find most recent version matching current config
            versions = self.config_service.get_all_versions(limit=1)
            if versions:
                previous_version_id = versions[0].id
        
        # Get snapshot of current state
        if current_config:
            previous_config_snapshot = {
                "embedding_model": current_config.embedding_model,
                "prompt_template": current_config.prompt_template,
                "similarity_threshold": current_config.similarity_threshold,
                "confidence_threshold": current_config.confidence_threshold
            }
        
        # Create rollback event (start as pending)
        rollback_event = RollbackEvent(
            trigger_type=trigger_type,
            trigger_reason=trigger_reason or "Manual rollback",
            alert_id=alert_id,
            previous_version_id=previous_version_id,
            restored_version_id=version_id,
            previous_config=previous_config_snapshot,
            executed_by=executed_by or "system",
            status=RollbackStatus.IN_PROGRESS
        )
        
        self.db.add(rollback_event)
        self.db.flush()
        
        try:
            # Execute rollback (atomic operation)
            restored_config = self.config_service.restore_configuration_from_version(version_id)
            
            # Determine which components were restored
            components_restored = []
            if previous_config_snapshot:
                if previous_config_snapshot.get("embedding_model") != restored_config.embedding_model:
                    components_restored.append("embedding_model")
                if previous_config_snapshot.get("prompt_template") != restored_config.prompt_template:
                    components_restored.append("prompt_template")
                if previous_config_snapshot.get("similarity_threshold") != restored_config.similarity_threshold:
                    components_restored.append("similarity_threshold")
                if previous_config_snapshot.get("confidence_threshold") != restored_config.confidence_threshold:
                    components_restored.append("confidence_threshold")
            else:
                components_restored = ["all_components"]
            
            # Update rollback event with success
            rollback_event.status = RollbackStatus.SUCCESS
            rollback_event.components_restored = components_restored
            rollback_event.components_failed = []
            
            # Verification metrics (can be populated after rollback completes)
            rollback_event.verification_metrics = {
                "restored_at": datetime.now().isoformat(),
                "restored_embedding_model": restored_config.embedding_model,
                "components_count": len(components_restored)
            }
            
        except Exception as e:
            # Rollback failed
            rollback_event.status = RollbackStatus.FAILED
            rollback_event.error_message = str(e)
            rollback_event.components_restored = []
            rollback_event.components_failed = ["all_components"]
            
            self.db.rollback()
            raise
        
        finally:
            self.db.commit()
            self.db.refresh(rollback_event)
        
        return rollback_event
    
    def auto_rollback_on_alert(self, alert_id: int) -> Optional[RollbackEvent]:
        """
        Automatically rollback when critical/emergency alert is triggered.
        
        Args:
            alert_id: Alert ID that triggered the rollback
            
        Returns:
            RollbackEvent if rollback executed, None if skipped
        """
        alert = self.db.query(DriftAlert).filter(DriftAlert.id == alert_id).first()
        
        if not alert:
            return None

        # Only auto-rollback on Critical or Emergency alerts
        severity_value = alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity)
        if severity_value not in ["critical", "emergency"]:
            return None

        # Find best known-good version
        best_version = self.config_service.get_best_version_by_metrics()

        if not best_version:
            # No known-good version available
            return None

        # Execute automatic rollback
        return self.execute_rollback(
            version_id=best_version.id,
            trigger_type=RollbackTriggerType.AUTOMATED,
            trigger_reason=f"Automated rollback triggered by {severity_value} drift alert: {alert.metric_name}",
            executed_by="system",
            alert_id=alert_id
        )
    
    def get_rollback_history(self, limit: int = 100) -> list:
        """Get rollback event history"""
        from backend.models.rollback import RollbackEvent
        return self.db.query(RollbackEvent).order_by(
            RollbackEvent.executed_at.desc()
        ).limit(limit).all()
    
    def get_rollback_by_id(self, rollback_id: int) -> Optional[RollbackEvent]:
        """Get specific rollback event by ID"""
        return self.db.query(RollbackEvent).filter(RollbackEvent.id == rollback_id).first()
    
    def check_automated_triggers(self) -> Optional[RollbackEvent]:
        """
        Check all automated rollback triggers and execute if conditions met.
        
        This method should be called periodically (e.g., every drift detection cycle)
        to evaluate whether automatic rollback is warranted.
        
        Returns:
            RollbackEvent if rollback triggered, None otherwise
        """
        # Check if we're in cooldown from recent rollback
        if self._is_in_cooldown():
            return None
        
        # Check each trigger type
        if ROLLBACK_TRIGGERS["emergency_alert"]["enabled"]:
            result = self._check_emergency_trigger()
            if result:
                return result
        
        if ROLLBACK_TRIGGERS["sustained_critical"]["enabled"]:
            result = self._check_sustained_critical_trigger()
            if result:
                return result
        
        if ROLLBACK_TRIGGERS["confidence_collapse"]["enabled"]:
            result = self._check_confidence_collapse_trigger()
            if result:
                return result
        
        return None
    
    def _is_in_cooldown(self, minutes: int = 30) -> bool:
        """Check if we recently performed a rollback (cooldown period)"""
        recent_rollback = self.db.query(RollbackEvent).filter(
            and_(
                RollbackEvent.status == RollbackStatus.SUCCESS,
                RollbackEvent.executed_at >= datetime.now() - timedelta(minutes=minutes)
            )
        ).first()
        return recent_rollback is not None
    
    def _check_emergency_trigger(self) -> Optional[RollbackEvent]:
        """Check for EMERGENCY alerts that should trigger immediate rollback"""
        config = ROLLBACK_TRIGGERS["emergency_alert"]
        
        # Find active emergency alerts
        emergency_alert = self.db.query(DriftAlert).filter(
            and_(
                DriftAlert.severity == DriftSeverity.EMERGENCY,
                DriftAlert.status == "active",
                DriftAlert.created_at >= datetime.now() - timedelta(minutes=config["cooldown_minutes"])
            )
        ).order_by(DriftAlert.created_at.desc()).first()
        
        if not emergency_alert:
            return None
        
        # Find best known-good version
        best_version = self.config_service.get_best_version_by_metrics()
        if not best_version:
            return None
        
        # Execute automatic rollback
        return self.execute_rollback(
            version_id=best_version.id,
            trigger_type=RollbackTriggerType.AUTOMATED,
            trigger_reason=f"EMERGENCY alert triggered auto-rollback: {emergency_alert.metric_name} = {emergency_alert.metric_value:.4f} exceeded threshold {emergency_alert.threshold_value:.4f}",
            executed_by="system",
            alert_id=emergency_alert.id
        )
    
    def _check_sustained_critical_trigger(self) -> Optional[RollbackEvent]:
        """Check for sustained CRITICAL alerts that warrant rollback"""
        config = ROLLBACK_TRIGGERS["sustained_critical"]
        
        # Count critical alerts in window
        critical_alerts = self.db.query(DriftAlert).filter(
            and_(
                DriftAlert.severity == DriftSeverity.CRITICAL,
                DriftAlert.status == "active",
                DriftAlert.created_at >= datetime.now() - timedelta(minutes=config["window_minutes"])
            )
        ).all()
        
        if len(critical_alerts) < config["threshold_count"]:
            return None
        
        # Find best known-good version
        best_version = self.config_service.get_best_version_by_metrics()
        if not best_version:
            return None
        
        # Execute automatic rollback
        return self.execute_rollback(
            version_id=best_version.id,
            trigger_type=RollbackTriggerType.AUTOMATED,
            trigger_reason=f"Sustained CRITICAL drift: {len(critical_alerts)} critical alerts in {config['window_minutes']} minutes",
            executed_by="system"
        )
    
    def _check_confidence_collapse_trigger(self) -> Optional[RollbackEvent]:
        """Check for significant confidence score collapse"""
        config = ROLLBACK_TRIGGERS["confidence_collapse"]
        
        # Get baseline average confidence (from first week)
        from backend.models.query_log import QueryLog
        
        # Get earliest query to determine baseline period
        earliest = self.db.query(QueryLog).order_by(QueryLog.timestamp.asc()).first()
        if not earliest:
            return None
        
        baseline_end = earliest.timestamp + timedelta(days=7)
        
        # Calculate baseline average confidence
        baseline_queries = self.db.query(QueryLog).filter(
            QueryLog.timestamp < baseline_end
        ).all()
        
        if len(baseline_queries) < 100:
            return None
        
        baseline_confidences = [
            float(q.confidence_score) for q in baseline_queries 
            if q.confidence_score and self._is_valid_confidence(q.confidence_score)
        ]
        
        if not baseline_confidences:
            return None
        
        baseline_avg = sum(baseline_confidences) / len(baseline_confidences)
        
        # Calculate current average confidence (last 15 minutes)
        current_queries = self.db.query(QueryLog).filter(
            QueryLog.timestamp >= datetime.now() - timedelta(minutes=15)
        ).all()
        
        if len(current_queries) < 10:
            return None
        
        current_confidences = [
            float(q.confidence_score) for q in current_queries 
            if q.confidence_score and self._is_valid_confidence(q.confidence_score)
        ]
        
        if not current_confidences:
            return None
        
        current_avg = sum(current_confidences) / len(current_confidences)
        
        # Check for significant drop
        drop = baseline_avg - current_avg
        if drop < config["threshold_drop"]:
            return None
        
        # Find best known-good version
        best_version = self.config_service.get_best_version_by_metrics()
        if not best_version:
            return None
        
        # Execute automatic rollback
        return self.execute_rollback(
            version_id=best_version.id,
            trigger_type=RollbackTriggerType.AUTOMATED,
            trigger_reason=f"Confidence collapse detected: dropped from {baseline_avg:.2f} to {current_avg:.2f} ({drop:.0%} decrease)",
            executed_by="system"
        )
    
    def _is_valid_confidence(self, value: str) -> bool:
        """Check if confidence score string is valid"""
        try:
            f = float(value)
            return 0.0 <= f <= 1.0
        except (ValueError, TypeError):
            return False
    
    def get_trigger_status(self) -> Dict:
        """Get current status of all automated triggers"""
        in_cooldown = self._is_in_cooldown()
        
        # Get recent alerts count
        recent_emergency = self.db.query(DriftAlert).filter(
            and_(
                DriftAlert.severity == DriftSeverity.EMERGENCY,
                DriftAlert.status == "active",
                DriftAlert.created_at >= datetime.now() - timedelta(hours=1)
            )
        ).count()
        
        recent_critical = self.db.query(DriftAlert).filter(
            and_(
                DriftAlert.severity == DriftSeverity.CRITICAL,
                DriftAlert.status == "active",
                DriftAlert.created_at >= datetime.now() - timedelta(hours=1)
            )
        ).count()
        
        return {
            "in_cooldown": in_cooldown,
            "triggers": {
                name: {
                    "enabled": config["enabled"],
                    "description": config["description"],
                }
                for name, config in ROLLBACK_TRIGGERS.items()
            },
            "current_state": {
                "emergency_alerts_1h": recent_emergency,
                "critical_alerts_1h": recent_critical,
            },
            "last_rollback": self._get_last_rollback_time(),
        }
    
    def _get_last_rollback_time(self) -> Optional[str]:
        """Get timestamp of last successful rollback"""
        last = self.db.query(RollbackEvent).filter(
            RollbackEvent.status == RollbackStatus.SUCCESS
        ).order_by(RollbackEvent.executed_at.desc()).first()
        
        return last.executed_at.isoformat() if last else None
    
    def get_rollback_audit_log(self, limit: int = 50) -> List[Dict]:
        """
        Get detailed audit log of all rollback events.
        
        For compliance and post-incident analysis.
        """
        events = self.db.query(RollbackEvent).order_by(
            RollbackEvent.executed_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": e.id,
                "timestamp": e.executed_at.isoformat() if e.executed_at else None,
                "trigger_type": e.trigger_type.value if e.trigger_type else None,
                "trigger_reason": e.trigger_reason,
                "status": e.status.value if e.status else None,
                "executed_by": e.executed_by,
                "previous_version_id": e.previous_version_id,
                "restored_version_id": e.restored_version_id,
                "components_restored": e.components_restored,
                "components_failed": e.components_failed,
                "error_message": e.error_message,
                "verification_metrics": e.verification_metrics,
            }
            for e in events
        ]