"""FastAPI endpoints for rollback operations"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from backend.database import get_db
from backend.services.rollback_service import RollbackService
from backend.services.configuration_service import ConfigurationService
from backend.models.rollback import RollbackTriggerType

router = APIRouter(prefix="/api/rollback", tags=["rollback"])

class RollbackRequest(BaseModel):
    """Request model for rollback"""
    version_id: int
    reason: str = "Manual rollback"

class RollbackResponse(BaseModel):
    """Response model for rollback"""
    id: int
    trigger_type: str
    status: str
    restored_version_id: int
    executed_at: str
    
    model_config = {"from_attributes": True}

@router.get("/versions")
def get_config_versions(db: Session = Depends(get_db)):
    """Get all configuration versions"""
    service = ConfigurationService(db)
    versions = service.get_all_versions()
    
    return [
        {
            "id": v.id,
            "embedding_model": v.embedding_model,
            "similarity_threshold": v.similarity_threshold,
            "confidence_threshold": v.confidence_threshold,
            "version_label": v.version_label,
            "snapshot_timestamp": v.snapshot_timestamp.isoformat(),
            "is_known_good": v.is_known_good,
            "performance_metrics": v.performance_metrics
        }
        for v in versions
    ]

@router.get("/versions/{version_id}")
def get_version(version_id: int, db: Session = Depends(get_db)):
    """Get specific configuration version"""
    service = ConfigurationService(db)
    version = service.get_version_by_id(version_id)
    
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return {
        "id": version.id,
        "embedding_model": version.embedding_model,
        "prompt_template": version.prompt_template,
        "similarity_threshold": version.similarity_threshold,
        "confidence_threshold": version.confidence_threshold,
        "version_label": version.version_label,
        "snapshot_timestamp": version.snapshot_timestamp.isoformat(),
        "is_known_good": version.is_known_good,
        "performance_metrics": version.performance_metrics
    }

@router.post("/execute", response_model=RollbackResponse)
def execute_rollback(request: RollbackRequest, db: Session = Depends(get_db)):
    """Execute rollback to a specific configuration version"""
    service = RollbackService(db)
    
    try:
        rollback_event = service.execute_rollback(
            version_id=request.version_id,
            trigger_type=RollbackTriggerType.MANUAL,
            trigger_reason=request.reason,
            executed_by="user"  # In real app, get from auth
        )
        
        return rollback_event
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")

@router.get("/history")
def get_rollback_history(limit: int = 100, db: Session = Depends(get_db)):
    """Get rollback event history"""
    service = RollbackService(db)
    events = service.get_rollback_history(limit=limit)
    
    return [
        {
            "id": e.id,
            "trigger_type": e.trigger_type.value if e.trigger_type else None,
            "trigger_reason": e.trigger_reason,
            "status": e.status.value if e.status else None,
            "previous_version_id": e.previous_version_id,
            "restored_version_id": e.restored_version_id,
            "executed_at": e.executed_at.isoformat() if e.executed_at else None,
            "executed_by": e.executed_by,
            "components_restored": e.components_restored
        }
        for e in events
    ]

@router.get("/current-config")
def get_current_config(db: Session = Depends(get_db)):
    """Get current active configuration"""
    service = ConfigurationService(db)
    config = service.get_current_config()
    
    if not config:
        raise HTTPException(status_code=404, detail="No configuration found")
    
    # Get version label if current config is linked to a version
    version_label = None
    if config.current_version_id:
        version = service.get_version_by_id(config.current_version_id)
        if version:
            version_label = version.version_label
    
    return {
        "id": config.id,
        "embedding_model": config.embedding_model,
        "prompt_template": config.prompt_template,
        "similarity_threshold": config.similarity_threshold,
        "confidence_threshold": config.confidence_threshold,
        "current_version_id": config.current_version_id,
        "version_label": version_label,
        "created_at": config.created_at.isoformat() if config.created_at else None,
        "updated_at": config.updated_at.isoformat() if config.updated_at else None
    }

@router.post("/snapshot")
def create_snapshot(
    version_label: str = None,
    db: Session = Depends(get_db)
):
    """Create a snapshot of current configuration"""
    from pydantic import BaseModel
    
    class SnapshotRequest(BaseModel):
        version_label: str = None
    
    service = ConfigurationService(db)
    version = service.snapshot_configuration(version_label=version_label)
    
    return {
        "id": version.id,
        "version_label": version.version_label,
        "snapshot_timestamp": version.snapshot_timestamp.isoformat()
    }


@router.get("/triggers/status")
def get_trigger_status(db: Session = Depends(get_db)):
    """
    Get status of all automated rollback triggers.
    
    Returns trigger configuration, current state, and whether system is in cooldown.
    """
    service = RollbackService(db)
    return service.get_trigger_status()


@router.get("/audit-log")
def get_audit_log(limit: int = 50, db: Session = Depends(get_db)):
    """
    Get detailed audit log of all rollback events.
    
    For compliance and post-incident analysis.
    """
    service = RollbackService(db)
    return service.get_rollback_audit_log(limit=limit)


@router.post("/versions/{version_id}/mark-known-good")
def mark_version_known_good(version_id: int, db: Session = Depends(get_db)):
    """
    Mark a configuration version as known-good.
    
    Known-good versions are preferred targets for automated rollbacks.
    """
    service = ConfigurationService(db)
    version = service.mark_version_as_known_good(version_id)
    
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return {
        "message": "Version marked as known-good",
        "version_id": version.id,
        "version_label": version.version_label
    }


@router.get("/recommended-version")
def get_recommended_version(db: Session = Depends(get_db)):
    """
    Get the recommended version for rollback.
    
    Returns the best known-good version based on performance metrics.
    """
    service = ConfigurationService(db)
    version = service.get_best_version_by_metrics()
    
    if not version:
        raise HTTPException(status_code=404, detail="No known-good versions available")
    
    return {
        "id": version.id,
        "version_label": version.version_label,
        "performance_metrics": version.performance_metrics,
        "is_known_good": version.is_known_good,
        "snapshot_timestamp": version.snapshot_timestamp.isoformat()
    }
