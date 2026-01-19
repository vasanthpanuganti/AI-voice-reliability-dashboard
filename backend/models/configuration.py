"""Configuration and versioning models"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base

class Configuration(Base):
    """Current active configuration"""
    __tablename__ = "configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    embedding_model = Column(String(200), nullable=False)
    prompt_template = Column(Text)
    similarity_threshold = Column(Float, default=0.75)
    confidence_threshold = Column(Float, default=0.70)
    is_current = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    versions = relationship("ConfigVersion", back_populates="configuration")

class ConfigVersion(Base):
    """Versioned snapshots of configurations"""
    __tablename__ = "config_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    configuration_id = Column(Integer, ForeignKey("configurations.id"))
    
    # Snapshot data
    embedding_model = Column(String(200), nullable=False)
    prompt_template = Column(Text)
    similarity_threshold = Column(Float)
    confidence_threshold = Column(Float)
    
    # Performance metrics linked to this version
    performance_metrics = Column(JSON)  # {accuracy, latency, confidence_scores, etc.}
    
    # Version metadata
    snapshot_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    version_label = Column(String(100))  # e.g., "v1.0", "baseline_week1"
    is_known_good = Column(Boolean, default=False, index=True)  # Mark stable versions
    
    # Relationships
    configuration = relationship("Configuration", back_populates="versions")
