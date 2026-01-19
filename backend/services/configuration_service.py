"""Configuration Service - Manages configuration versioning"""
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import and_

from backend.models.configuration import Configuration, ConfigVersion
from backend.config import settings

class ConfigurationService:
    """Service for managing configuration versions"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_current_config(self) -> Optional[Configuration]:
        """Get current active configuration"""
        return self.db.query(Configuration).filter(Configuration.is_current == True).first()
    
    def create_or_update_current_config(
        self,
        embedding_model: str,
        prompt_template: Optional[str] = None,
        similarity_threshold: float = 0.75,
        confidence_threshold: float = 0.70
    ) -> Configuration:
        """Create or update current configuration"""
        current = self.get_current_config()
        
        if current:
            # Update existing
            current.embedding_model = embedding_model
            current.prompt_template = prompt_template
            current.similarity_threshold = similarity_threshold
            current.confidence_threshold = confidence_threshold
            current.updated_at = datetime.now()
        else:
            # Create new
            current = Configuration(
                embedding_model=embedding_model,
                prompt_template=prompt_template,
                similarity_threshold=similarity_threshold,
                confidence_threshold=confidence_threshold,
                is_current=True
            )
            self.db.add(current)
        
        self.db.commit()
        self.db.refresh(current)
        
        return current
    
    def snapshot_configuration(
        self,
        version_label: Optional[str] = None,
        performance_metrics: Optional[Dict] = None
    ) -> ConfigVersion:
        """
        Create a snapshot of current configuration.
        
        Args:
            version_label: Optional label for this version (e.g., "v1.0", "baseline_week1")
            performance_metrics: Optional performance metrics to link to this version
            
        Returns:
            ConfigVersion object
        """
        current = self.get_current_config()
        
        if not current:
            # Create default config if none exists
            current = self.create_or_update_current_config(
                embedding_model=settings.EMBEDDING_MODEL
            )
        
        # Create version snapshot
        config_version = ConfigVersion(
            configuration_id=current.id,
            embedding_model=current.embedding_model,
            prompt_template=current.prompt_template,
            similarity_threshold=current.similarity_threshold,
            confidence_threshold=current.confidence_threshold,
            version_label=version_label or f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            performance_metrics=performance_metrics or {}
        )
        
        self.db.add(config_version)
        self.db.flush()  # Flush to get the ID
        
        # Link current config to this snapshot if it doesn't have a version yet
        if not current.current_version_id:
            current.current_version_id = config_version.id
            current.updated_at = datetime.now()
        
        self.db.commit()
        self.db.refresh(config_version)
        
        return config_version
    
    def get_version_by_id(self, version_id: int) -> Optional[ConfigVersion]:
        """Get configuration version by ID"""
        return self.db.query(ConfigVersion).filter(ConfigVersion.id == version_id).first()
    
    def get_all_versions(self, limit: int = 100) -> List[ConfigVersion]:
        """Get all configuration versions, ordered by most recent"""
        return self.db.query(ConfigVersion).order_by(
            ConfigVersion.snapshot_timestamp.desc()
        ).limit(limit).all()
    
    def get_known_good_versions(self) -> List[ConfigVersion]:
        """Get versions marked as known-good (stable)"""
        return self.db.query(ConfigVersion).filter(
            ConfigVersion.is_known_good == True
        ).order_by(ConfigVersion.snapshot_timestamp.desc()).all()
    
    def get_best_version_by_metrics(self) -> Optional[ConfigVersion]:
        """Get version with best performance metrics"""
        known_good = self.get_known_good_versions()
        
        if not known_good:
            # If no known-good versions, return most recent
            all_versions = self.get_all_versions(limit=10)
            return all_versions[0] if all_versions else None
        
        # Find version with best metrics (e.g., highest accuracy, lowest error rate)
        best_version = None
        best_score = -1
        
        for version in known_good:
            metrics = version.performance_metrics or {}
            
            # Simple scoring: prefer higher accuracy, lower error rate
            accuracy = metrics.get("accuracy", 0)
            error_rate = metrics.get("error_rate", 1)
            score = accuracy * (1 - error_rate)
            
            if score > best_score:
                best_score = score
                best_version = version
        
        return best_version or known_good[0]  # Fallback to first known-good
    
    def mark_version_as_known_good(self, version_id: int) -> ConfigVersion:
        """Mark a configuration version as known-good (stable)"""
        version = self.get_version_by_id(version_id)
        if version:
            version.is_known_good = True
            self.db.commit()
            self.db.refresh(version)
        return version
    
    def restore_configuration_from_version(self, version_id: int) -> Configuration:
        """
        Restore current configuration from a version snapshot.
        
        Args:
            version_id: ID of version to restore
            
        Returns:
            Updated current Configuration
        """
        version = self.get_version_by_id(version_id)
        if not version:
            raise ValueError(f"Configuration version {version_id} not found")
        
        # Update current configuration
        current = self.get_current_config()
        if not current:
            current = Configuration(is_current=True)
            self.db.add(current)
        
        current.embedding_model = version.embedding_model
        current.prompt_template = version.prompt_template
        current.similarity_threshold = version.similarity_threshold
        current.confidence_threshold = version.confidence_threshold
        current.current_version_id = version_id  # Link current config to the version it's based on
        current.updated_at = datetime.now()
        
        self.db.commit()
        self.db.refresh(current)
        
        return current
    
    def link_metrics_to_version(self, version_id: int, metrics: Dict) -> ConfigVersion:
        """Link performance metrics to a configuration version"""
        version = self.get_version_by_id(version_id)
        if version:
            # Merge with existing metrics
            existing = version.performance_metrics or {}
            existing.update(metrics)
            version.performance_metrics = existing
            self.db.commit()
            self.db.refresh(version)
        return version
