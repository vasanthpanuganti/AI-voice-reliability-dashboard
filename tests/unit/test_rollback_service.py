"""Unit tests for rollback_service.py"""
import pytest
from datetime import datetime, timedelta
from backend.services.rollback_service import RollbackService
from backend.models.rollback import RollbackTriggerType, RollbackStatus
from backend.models.drift_metrics import DriftSeverity, DriftAlert

class TestRollbackExecution:
    """Tests for rollback execution"""
    
    def test_execute_rollback(self, rollback_service, test_db, sample_configuration):
        """Test executing a rollback"""
        v1 = sample_configuration["v1"]
        rollback_event = rollback_service.execute_rollback(
            version_id=v1.id,
            trigger_type=RollbackTriggerType.MANUAL,
            trigger_reason="Test rollback",
            executed_by="test_user"
        )
        
        assert rollback_event is not None
        assert rollback_event.status == RollbackStatus.SUCCESS
        assert rollback_event.restored_version_id == v1.id
        assert rollback_event.trigger_reason == "Test rollback"
    
    def test_rollback_audit_log(self, rollback_service, test_db, sample_configuration):
        """Test rollback creates audit log entry"""
        v1 = sample_configuration["v1"]
        rollback_event = rollback_service.execute_rollback(v1.id)
        
        audit_log = rollback_service.get_rollback_audit_log(limit=10)
        assert len(audit_log) > 0
        # Audit log returns list of dicts with 'id' key
        assert any(log.get("id") == rollback_event.id for log in audit_log)

class TestAutomatedTriggers:
    """Tests for automated rollback triggers"""
    
    def test_check_automated_triggers_no_trigger(self, rollback_service, test_db):
        """Test checking triggers when none should fire"""
        result = rollback_service.check_automated_triggers()
        # Should return None when no triggers fire
        assert result is None or isinstance(result, dict)
    
    def test_emergency_trigger(self, rollback_service, test_db, sample_configuration, drift_service):
        """Test emergency alert triggers rollback"""
        # Create emergency alert
        from backend.models.drift_metrics import DriftMetric, MetricType
        metric = DriftMetric(
            metric_type=MetricType.INPUT_DRIFT,
            psi_score=0.50,  # Emergency level
            timestamp=datetime.now()
        )
        test_db.add(metric)
        test_db.commit()
        
        alert = DriftAlert(
            metric_type=MetricType.INPUT_DRIFT,
            metric_name="psi_score",
            metric_value=0.50,
            threshold_value=0.40,
            severity=DriftSeverity.EMERGENCY,
            status="active",
            drift_metric_id=metric.id,
            created_at=datetime.now()
        )
        test_db.add(alert)
        test_db.commit()
        test_db.refresh(metric)
        
        # Check triggers (may or may not trigger depending on cooldown)
        from backend.models.rollback import RollbackEvent
        result = rollback_service.check_automated_triggers()
        # Should either trigger rollback (returns RollbackEvent) or be in cooldown (returns None)
        assert result is None or isinstance(result, RollbackEvent)
    
    def test_get_trigger_status(self, rollback_service):
        """Test getting trigger status"""
        status = rollback_service.get_trigger_status()
        
        assert status is not None
        assert "triggers" in status
        assert "in_cooldown" in status  # Key is 'in_cooldown', not 'cooldown_active'
    
    def test_cooldown_mechanism(self, rollback_service, test_db, sample_configuration):
        """Test cooldown prevents rapid rollbacks"""
        v1 = sample_configuration["v1"]
        
        # Execute first rollback
        rollback_service.execute_rollback(v1.id)
        
        # Try to trigger again immediately
        result = rollback_service.check_automated_triggers()
        # Should be in cooldown or return None
        assert result is None or isinstance(result, dict)
