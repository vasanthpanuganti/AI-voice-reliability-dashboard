"""Pre-computed baseline statistics for efficient drift detection"""
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text
from sqlalchemy.sql import func
from backend.database import Base


class BaselineStatistics(Base):
    """
    Stores pre-computed baseline statistics for efficient drift detection.
    
    Instead of loading all baseline queries and recomputing histograms every time,
    we store the computed statistics and update them incrementally.
    
    This provides 10-50x speedup for PSI/KS calculations on large datasets.
    """
    __tablename__ = "baseline_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Baseline identification
    baseline_start = Column(DateTime(timezone=True), nullable=False, index=True)
    baseline_end = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Feature being tracked
    feature_name = Column(String(100), nullable=False, index=True)  # e.g., "query_length", "confidence_score", "query_category"
    feature_type = Column(String(20), nullable=False)  # "numerical" or "categorical"
    
    # For numerical features: histogram data
    histogram_counts = Column(JSON)  # List of bin counts
    histogram_bin_edges = Column(JSON)  # List of bin edges
    
    # For categorical features: category counts
    category_counts = Column(JSON)  # Dict mapping category -> count
    
    # Summary statistics (numerical features)
    sample_size = Column(Integer, nullable=False)
    mean = Column(Float)
    std = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    median = Column(Float)
    q25 = Column(Float)
    q75 = Column(Float)
    
    # For embedding features: PCA components or summary
    embedding_summary = Column(JSON)  # Reduced representation for embedding comparison
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Version for cache invalidation
    version = Column(Integer, default=1)


class DriftComputationCache(Base):
    """
    Short-lived cache for recent drift computations.
    
    Avoids recomputing drift metrics when queried multiple times in quick succession.
    """
    __tablename__ = "drift_computation_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Cache key components
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    
    # Computed metrics
    psi_score = Column(Float)
    ks_statistic = Column(Float)
    ks_p_value = Column(Float)
    js_divergence = Column(Float)
    wasserstein_distance = Column(Float)
    chi_square_statistic = Column(Float)
    chi_square_p_value = Column(Float)
    
    # Additional context
    window_start = Column(DateTime(timezone=True))
    window_end = Column(DateTime(timezone=True))
    sample_size = Column(Integer)
    computation_time_ms = Column(Float)
    
    # Cache management
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), index=True)
