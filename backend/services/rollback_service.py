"""Rollback Service - Handles configuration rollbacks"""
from datetime import datetime
from typing import Optional, Dict
from sqlalchemy.orm import Session

from backend.models.configuration import ConfigVersion
from backend.models.drift_metrics import DriftAlert
from backend.models.rollback import RollbackEvent, RollbackTriggerType, RollbackStatus
from backend.services.configuration_service import ConfigurationService

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
        if alert.severity not in ["critical", "emergency"]:
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
            trigger_reason=f"Automated rollback triggered by {alert.severity} drift alert: {alert.metric_name}",
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
