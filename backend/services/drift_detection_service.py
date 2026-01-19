"""Drift Detection Service - Core service for detecting model drift

Optimizations applied:
- Pre-computed baseline statistics with histogram caching
- Efficient database queries with projections (fetch only needed columns)
- Categorical drift detection with Chi-Square test
- Intelligent subsampling for large windows
- Parallel computation with ThreadPoolExecutor
- Consistent missing value handling
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, func as sql_func
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import time

from backend.models.query_log import QueryLog
from backend.models.drift_metrics import DriftMetric, DriftAlert, DriftSeverity, MetricType
from backend.utils.drift_statistics import (
    calculate_psi, compute_ks_test, compute_jensen_shannon, compute_wasserstein_distance,
    compute_chi_square, compute_psi_incremental, create_histogram_cache, stratified_subsample
)
from backend.config import settings

# Configuration for optimizations
MAX_SAMPLE_SIZE = 10000  # Maximum samples for drift computation
MIN_SAMPLE_SIZE = 100    # Minimum samples required
CHI_SQUARE_MIN_FREQ = 5  # Minimum expected frequency for chi-square


def _safe_future_result(future, default, timeout=30):
    """Safely retrieve result from a Future with timeout and fallback."""
    try:
        return future.result(timeout=timeout)
    except Exception:
        return default


class DriftDetectionService:
    """Service for detecting drift in AI pipeline"""

    def __init__(self, db: Session):
        self.db = db
        self.window_minutes = settings.DRIFT_WINDOW_MINUTES

        # Baseline caching to reduce database load
        self._baseline_cache = None
        self._baseline_cache_time = None
        self._cache_ttl = timedelta(hours=24)  # Baseline period rarely changes

        # Thresholds
        self.psi_thresholds = {
            DriftSeverity.WARNING: settings.PSI_WARNING_THRESHOLD,
            DriftSeverity.CRITICAL: settings.PSI_CRITICAL_THRESHOLD,
            DriftSeverity.EMERGENCY: settings.PSI_EMERGENCY_THRESHOLD
        }

        self.ks_thresholds = {
            DriftSeverity.WARNING: settings.KS_WARNING_THRESHOLD,
            DriftSeverity.CRITICAL: settings.KS_CRITICAL_THRESHOLD,
            DriftSeverity.EMERGENCY: settings.KS_EMERGENCY_THRESHOLD
        }

        self.js_thresholds = {
            DriftSeverity.WARNING: settings.JS_WARNING_THRESHOLD,
            DriftSeverity.CRITICAL: settings.JS_CRITICAL_THRESHOLD,
            DriftSeverity.EMERGENCY: settings.JS_EMERGENCY_THRESHOLD
        }

        self.wasserstein_thresholds = {
            DriftSeverity.WARNING: settings.WASSERSTEIN_WARNING_THRESHOLD,
            DriftSeverity.CRITICAL: settings.WASSERSTEIN_CRITICAL_THRESHOLD,
            DriftSeverity.EMERGENCY: settings.WASSERSTEIN_EMERGENCY_THRESHOLD
        }

        # Chi-Square thresholds for categorical drift (p-value based, lower = more severe)
        self.chi_square_thresholds = {
            DriftSeverity.WARNING: 0.05,
            DriftSeverity.CRITICAL: 0.01,
            DriftSeverity.EMERGENCY: 0.001
        }

        # Pre-computed baseline statistics cache (histogram-based)
        self._baseline_stats_cache: Dict[str, Any] = {}
        self._baseline_stats_time: Optional[datetime] = None
    
    def get_baseline_period(self) -> Tuple[datetime, datetime]:
        """Get baseline period (first week of data)"""
        # For MVP, use first 7 days as baseline
        earliest_query = self.db.query(QueryLog).order_by(QueryLog.timestamp.asc()).first()
        if not earliest_query:
            # Default to 7 days ago if no data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            return start_time, end_time
        
        baseline_start = earliest_query.timestamp
        baseline_end = baseline_start + timedelta(days=7)
        return baseline_start, baseline_end
    
    def get_current_window(self, window_minutes: Optional[int] = None) -> Tuple[datetime, datetime]:
        """Get current rolling window"""
        if window_minutes is None:
            window_minutes = self.window_minutes
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        return start_time, end_time
    
    def get_baseline_queries(self, use_cache: bool = True) -> List[QueryLog]:
        """
        Get baseline queries for comparison with caching.

        Args:
            use_cache: Whether to use cached baseline (default True)

        Returns:
            List of baseline QueryLog objects
        """
        now = datetime.now()

        # Use cache if valid and requested
        if (use_cache and
            self._baseline_cache is not None and
            self._baseline_cache_time is not None and
            now - self._baseline_cache_time < self._cache_ttl):
            return self._baseline_cache

        # Fetch from database
        baseline_start, baseline_end = self.get_baseline_period()

        queries = self.db.query(QueryLog).filter(
            and_(
                QueryLog.timestamp >= baseline_start,
                QueryLog.timestamp < baseline_end
            )
        ).all()

        # Update cache
        if use_cache:
            self._baseline_cache = queries
            self._baseline_cache_time = now

        return queries
    
    def get_window_queries(self, start_time: datetime, end_time: datetime) -> List[QueryLog]:
        """Get queries in specified time window"""
        queries = self.db.query(QueryLog).filter(
            and_(
                QueryLog.timestamp >= start_time,
                QueryLog.timestamp < end_time
            )
        ).all()

        return queries

    def invalidate_baseline_cache(self):
        """
        Invalidate the baseline cache.
        Call this when baseline period changes or new data affects the baseline.
        """
        self._baseline_cache = None
        self._baseline_cache_time = None
        self._baseline_stats_cache = {}
        self._baseline_stats_time = None

    # ========================================================================
    # OPTIMIZED QUERY METHODS - Use projections to fetch only needed columns
    # ========================================================================

    def get_baseline_features_optimized(self, features: List[str]) -> Dict[str, List]:
        """
        Fetch only specific features from baseline queries (optimized).
        
        Uses SQL projections to avoid loading full QueryLog objects.
        10-50x faster for large datasets.
        
        Args:
            features: List of column names to fetch (e.g., ['query', 'confidence_score'])
            
        Returns:
            Dictionary mapping feature name to list of values
        """
        baseline_start, baseline_end = self.get_baseline_period()
        
        # Build column list dynamically
        columns = [getattr(QueryLog, f) for f in features if hasattr(QueryLog, f)]
        if not columns:
            return {}
        
        results = self.db.query(*columns).filter(
            and_(
                QueryLog.timestamp >= baseline_start,
                QueryLog.timestamp < baseline_end
            )
        ).all()
        
        # Convert to dict of lists
        feature_data = {f: [] for f in features}
        for row in results:
            for i, f in enumerate(features):
                if i < len(row):
                    feature_data[f].append(row[i])
        
        return feature_data

    def get_window_features_optimized(self, start_time: datetime, end_time: datetime, 
                                      features: List[str]) -> Dict[str, List]:
        """
        Fetch only specific features from window queries (optimized).
        
        Args:
            start_time: Window start
            end_time: Window end
            features: List of column names to fetch
            
        Returns:
            Dictionary mapping feature name to list of values
        """
        columns = [getattr(QueryLog, f) for f in features if hasattr(QueryLog, f)]
        if not columns:
            return {}
        
        results = self.db.query(*columns).filter(
            and_(
                QueryLog.timestamp >= start_time,
                QueryLog.timestamp < end_time
            )
        ).all()
        
        feature_data = {f: [] for f in features}
        for row in results:
            for i, f in enumerate(features):
                if i < len(row):
                    feature_data[f].append(row[i])
        
        return feature_data

    def get_category_counts_optimized(self, start_time: datetime, end_time: datetime,
                                      category_field: str = "query_category") -> Dict[str, int]:
        """
        Get category counts using SQL aggregation (much faster than loading all records).
        
        Args:
            start_time: Period start
            end_time: Period end
            category_field: Field to count by
            
        Returns:
            Dictionary mapping category to count
        """
        column = getattr(QueryLog, category_field, None)
        if column is None:
            return {}
        
        results = self.db.query(
            column, sql_func.count(QueryLog.id)
        ).filter(
            and_(
                QueryLog.timestamp >= start_time,
                QueryLog.timestamp < end_time
            )
        ).group_by(column).all()
        
        return {cat or "unknown": count for cat, count in results}

    # ========================================================================
    # PRE-COMPUTED BASELINE STATISTICS
    # ========================================================================

    def get_baseline_statistics(self) -> Dict[str, Any]:
        """
        Get or compute pre-computed baseline statistics.
        
        Caches histogram data, category counts, and summary statistics
        for efficient incremental drift computation.
        
        Returns:
            Dictionary with pre-computed statistics for all features
        """
        now = datetime.now()
        
        # Check cache validity
        if (self._baseline_stats_cache and 
            self._baseline_stats_time and
            now - self._baseline_stats_time < self._cache_ttl):
            return self._baseline_stats_cache
        
        baseline_start, baseline_end = self.get_baseline_period()
        
        # Compute statistics for numerical features
        stats = {}
        
        # 1. Query length histogram
        query_data = self.get_baseline_features_optimized(['query'])
        if query_data.get('query'):
            query_lengths = np.array([len(q) for q in query_data['query'] if q])
            stats['query_length'] = create_histogram_cache(query_lengths)
        
        # 2. Confidence score histogram
        conf_data = self.get_baseline_features_optimized(['confidence_score'])
        if conf_data.get('confidence_score'):
            confidences = []
            for c in conf_data['confidence_score']:
                if c:
                    try:
                        val = float(c)
                        if 0 <= val <= 1:
                            confidences.append(val)
                    except (ValueError, TypeError):
                        pass
            if confidences:
                stats['confidence_score'] = create_histogram_cache(np.array(confidences))
        
        # 3. Category counts (for chi-square)
        stats['category_counts'] = self.get_category_counts_optimized(
            baseline_start, baseline_end, 'query_category'
        )
        
        # 4. Embedding statistics (summarized)
        # For embeddings, we'll still need to compute on-demand due to size
        
        # Cache the results
        self._baseline_stats_cache = stats
        self._baseline_stats_time = now
        
        return stats

    # ========================================================================
    # CATEGORICAL DRIFT DETECTION
    # ========================================================================

    def detect_categorical_drift(self, category_field: str = "query_category") -> Dict:
        """
        Detect drift in categorical features using Chi-Square test.
        
        Useful for detecting shifts in query types, departments, etc.
        
        Args:
            category_field: Field to analyze
            
        Returns:
            Dictionary with chi-square results and category details
        """
        baseline_start, baseline_end = self.get_baseline_period()
        window_start, window_end = self.get_current_window()
        
        # Get category counts using optimized SQL aggregation
        baseline_counts = self.get_category_counts_optimized(
            baseline_start, baseline_end, category_field
        )
        window_counts = self.get_category_counts_optimized(
            window_start, window_end, category_field
        )
        
        if not baseline_counts or not window_counts:
            return {
                "chi_square_statistic": 0.0,
                "chi_square_p_value": 1.0,
                "drift_detected": False,
                "category_details": {},
                "baseline_size": sum(baseline_counts.values()) if baseline_counts else 0,
                "window_size": sum(window_counts.values()) if window_counts else 0
            }
        
        # Compute chi-square test
        chi2, p_value, category_details = compute_chi_square(
            baseline_counts, window_counts, min_expected_freq=CHI_SQUARE_MIN_FREQ
        )
        
        # Check threshold
        severity = self._check_chi_square_threshold(p_value)
        drift_detected = severity is not None
        
        return {
            "chi_square_statistic": chi2,
            "chi_square_p_value": p_value,
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None,
            "category_details": category_details,
            "baseline_size": sum(baseline_counts.values()),
            "window_size": sum(window_counts.values())
        }

    def _check_chi_square_threshold(self, p_value: float) -> Optional[DriftSeverity]:
        return self._check_threshold(p_value, self.chi_square_thresholds, lower_is_worse=True)

    # ========================================================================
    # SUBSAMPLING FOR LARGE DATASETS
    # ========================================================================

    def _subsample_if_needed(self, queries: List[QueryLog], 
                             max_samples: int = MAX_SAMPLE_SIZE) -> List[QueryLog]:
        """
        Intelligently subsample large query sets while preserving distribution.
        
        Args:
            queries: List of QueryLog objects
            max_samples: Maximum samples to return
            
        Returns:
            Subsampled list (or original if small enough)
        """
        if len(queries) <= max_samples:
            return queries
        
        # Extract categories for stratified sampling
        categories = [q.query_category or "unknown" for q in queries]
        
        # Use stratified subsampling
        sampled_queries, _ = stratified_subsample(
            queries, categories, 
            max_samples=max_samples, 
            min_per_category=MIN_SAMPLE_SIZE // 10
        )
        
        return sampled_queries

    # ========================================================================
    # MISSING VALUE HANDLING
    # ========================================================================

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float with missing value handling"""
        if value is None:
            return default
        try:
            result = float(value)
            return result if not np.isnan(result) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_string_length(value: Any, default: int = 0) -> int:
        """Safely get string length with missing value handling"""
        if value is None:
            return default
        try:
            return len(str(value))
        except Exception:
            return default

    def detect_input_drift(self, baseline_queries: List[QueryLog], window_queries: List[QueryLog]) -> Dict:
        """
        Detect input drift (query distribution changes) using PSI.

        OPTIMIZED: Uses subsampling for large datasets and safe value extraction.

        Args:
            baseline_queries: Pre-fetched baseline queries
            window_queries: Pre-fetched window queries

        Returns:
            Dictionary with PSI score and drift detection result
        """
        if len(baseline_queries) < MIN_SAMPLE_SIZE // 10 or len(window_queries) < 5:
            return {
                "psi_score": 0.0,
                "sample_size": len(window_queries),
                "baseline_size": len(baseline_queries),
                "drift_detected": False
            }

        # OPTIMIZATION: Subsample if dataset is too large
        baseline_sample = self._subsample_if_needed(baseline_queries)
        window_sample = self._subsample_if_needed(window_queries, max_samples=MAX_SAMPLE_SIZE // 10)

        # Extract query features with safe handling of missing values
        baseline_lengths = np.array([
            self._safe_string_length(q.query) for q in baseline_sample
        ])
        window_lengths = np.array([
            self._safe_string_length(q.query) for q in window_sample
        ])

        # Calculate PSI
        psi_score = calculate_psi(baseline_lengths, window_lengths)

        # Check threshold
        severity = self._check_psi_threshold(psi_score)
        drift_detected = severity is not None

        return {
            "psi_score": psi_score,
            "sample_size": len(window_queries),
            "baseline_size": len(baseline_queries),
            "sampled_baseline_size": len(baseline_sample),
            "sampled_window_size": len(window_sample),
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None
        }

    def detect_input_drift_incremental(self) -> Dict:
        """
        Detect input drift using pre-computed baseline histogram (10-50x faster).
        
        Uses cached baseline statistics for incremental computation.
        
        Returns:
            Dictionary with PSI score and drift detection result
        """
        # Get pre-computed baseline statistics
        baseline_stats = self.get_baseline_statistics()
        query_length_stats = baseline_stats.get('query_length')
        
        if not query_length_stats or not query_length_stats.get('counts'):
            # Fall back to standard computation
            baseline_queries = self.get_baseline_queries(use_cache=True)
            window_start, window_end = self.get_current_window()
            window_queries = self.get_window_queries(window_start, window_end)
            return self.detect_input_drift(baseline_queries, window_queries)
        
        # Get current window query lengths
        window_start, window_end = self.get_current_window()
        window_data = self.get_window_features_optimized(window_start, window_end, ['query'])
        
        if not window_data.get('query') or len(window_data['query']) < 5:
            return {
                "psi_score": 0.0,
                "sample_size": len(window_data.get('query', [])),
                "baseline_size": query_length_stats.get('sample_size', 0),
                "drift_detected": False
            }
        
        window_lengths = np.array([
            self._safe_string_length(q) for q in window_data['query']
        ])
        
        # Use incremental PSI computation
        baseline_hist = np.array(query_length_stats['counts'])
        baseline_edges = np.array(query_length_stats['bin_edges'])
        
        psi_score = compute_psi_incremental(baseline_hist, baseline_edges, window_lengths)
        
        severity = self._check_psi_threshold(psi_score)
        drift_detected = severity is not None
        
        return {
            "psi_score": psi_score,
            "sample_size": len(window_lengths),
            "baseline_size": query_length_stats.get('sample_size', 0),
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None,
            "computation_mode": "incremental"
        }
    
    def detect_output_drift(self, baseline_queries: List[QueryLog], window_queries: List[QueryLog]) -> Dict:
        """
        Detect output drift (response quality changes) using KS test.

        OPTIMIZED: Uses subsampling for large datasets and safe value extraction.

        Args:
            baseline_queries: Pre-fetched baseline queries
            window_queries: Pre-fetched window queries

        Returns:
            Dictionary with KS test results
        """
        if len(baseline_queries) < MIN_SAMPLE_SIZE // 10 or len(window_queries) < 5:
            return {
                "ks_statistic": 0.0,
                "ks_p_value": 1.0,
                "sample_size": len(window_queries),
                "drift_detected": False
            }

        # OPTIMIZATION: Subsample if dataset is too large
        baseline_sample = self._subsample_if_needed(baseline_queries)
        window_sample = self._subsample_if_needed(window_queries, max_samples=MAX_SAMPLE_SIZE // 10)

        # Use confidence scores with safe extraction and missing value handling
        baseline_values = np.array([
            self._safe_float(q.confidence_score, default=self._safe_string_length(q.ai_response) / 1000.0)
            for q in baseline_sample
        ])

        window_values = np.array([
            self._safe_float(q.confidence_score, default=self._safe_string_length(q.ai_response) / 1000.0)
            for q in window_sample
        ])

        # Perform KS test
        ks_statistic, ks_p_value = compute_ks_test(window_values, baseline_values)

        # Lower p-value means more significant difference
        severity = self._check_ks_threshold(ks_p_value)
        drift_detected = severity is not None

        return {
            "ks_statistic": ks_statistic,
            "ks_p_value": ks_p_value,
            "sample_size": len(window_queries),
            "baseline_size": len(baseline_queries),
            "sampled_baseline_size": len(baseline_sample),
            "sampled_window_size": len(window_sample),
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None
        }

    def detect_output_drift_incremental(self) -> Dict:
        """
        Detect output drift using pre-computed baseline statistics (faster).
        
        Uses cached baseline confidence score distribution.
        
        Returns:
            Dictionary with KS test results
        """
        baseline_stats = self.get_baseline_statistics()
        conf_stats = baseline_stats.get('confidence_score')
        
        if not conf_stats or not conf_stats.get('counts'):
            # Fall back to standard computation
            baseline_queries = self.get_baseline_queries(use_cache=True)
            window_start, window_end = self.get_current_window()
            window_queries = self.get_window_queries(window_start, window_end)
            return self.detect_output_drift(baseline_queries, window_queries)
        
        # Get current window confidence scores
        window_start, window_end = self.get_current_window()
        window_data = self.get_window_features_optimized(
            window_start, window_end, ['confidence_score']
        )
        
        if not window_data.get('confidence_score'):
            return {
                "ks_statistic": 0.0,
                "ks_p_value": 1.0,
                "sample_size": 0,
                "drift_detected": False
            }
        
        window_values = np.array([
            self._safe_float(c) for c in window_data['confidence_score']
            if c is not None
        ])
        
        if len(window_values) < 5:
            return {
                "ks_statistic": 0.0,
                "ks_p_value": 1.0,
                "sample_size": len(window_values),
                "drift_detected": False
            }
        
        # Reconstruct baseline values from histogram for KS test
        # This is an approximation but much faster than loading all baseline data
        baseline_hist = np.array(conf_stats['counts'])
        baseline_edges = np.array(conf_stats['bin_edges'])
        
        # Generate synthetic baseline samples from histogram
        baseline_synthetic = []
        for i, count in enumerate(baseline_hist):
            if i < len(baseline_edges) - 1:
                bin_center = (baseline_edges[i] + baseline_edges[i+1]) / 2
                baseline_synthetic.extend([bin_center] * count)
        
        baseline_values = np.array(baseline_synthetic) if baseline_synthetic else np.array([0.5])
        
        ks_statistic, ks_p_value = compute_ks_test(window_values, baseline_values)
        severity = self._check_ks_threshold(ks_p_value)
        drift_detected = severity is not None
        
        return {
            "ks_statistic": ks_statistic,
            "ks_p_value": ks_p_value,
            "sample_size": len(window_values),
            "baseline_size": conf_stats.get('sample_size', 0),
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None,
            "computation_mode": "incremental"
        }
    
    def detect_embedding_drift(self, baseline_queries: List[QueryLog], window_queries: List[QueryLog]) -> Dict:
        """
        Detect embedding space drift using both Wasserstein distance and Jensen-Shannon divergence.

        Wasserstein distance is more accurate for high-dimensional embeddings as it preserves
        geometric structure and uses PCA to analyze all dimensions effectively.

        Args:
            baseline_queries: Pre-fetched baseline queries
            window_queries: Pre-fetched window queries

        Returns:
            Dictionary with drift results (Wasserstein distance and JS divergence)
        """
        if len(baseline_queries) < 10 or len(window_queries) < 5:
            return {
                "js_divergence": 0.0,
                "wasserstein_distance": 0.0,
                "sample_size": len(window_queries),
                "drift_detected": False
            }

        # Extract embeddings (vectorized with list comprehension)
        baseline_embeddings = np.array([
            q.embedding for q in baseline_queries if q.embedding
        ])

        window_embeddings = np.array([
            q.embedding for q in window_queries if q.embedding
        ])

        if len(baseline_embeddings) < 5 or len(window_embeddings) < 5:
            return {
                "js_divergence": 0.0,
                "wasserstein_distance": 0.0,
                "sample_size": len(window_embeddings),
                "drift_detected": False
            }

        # Calculate Wasserstein distance (primary metric for embeddings)
        wasserstein_dist = compute_wasserstein_distance(window_embeddings, baseline_embeddings)

        # Calculate JS divergence (secondary metric for comparison)
        js_divergence = compute_jensen_shannon(window_embeddings, baseline_embeddings)

        # Check thresholds for both metrics
        wasserstein_severity = self._check_wasserstein_threshold(wasserstein_dist)
        js_severity = self._check_js_threshold(js_divergence)

        # Use the more severe of the two
        severity = None
        if wasserstein_severity == DriftSeverity.EMERGENCY or js_severity == DriftSeverity.EMERGENCY:
            severity = DriftSeverity.EMERGENCY
        elif wasserstein_severity == DriftSeverity.CRITICAL or js_severity == DriftSeverity.CRITICAL:
            severity = DriftSeverity.CRITICAL
        elif wasserstein_severity == DriftSeverity.WARNING or js_severity == DriftSeverity.WARNING:
            severity = DriftSeverity.WARNING

        drift_detected = severity is not None

        return {
            "js_divergence": js_divergence,
            "wasserstein_distance": wasserstein_dist,
            "sample_size": len(window_embeddings),
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None
        }
    
    def _check_threshold(self, value: float, thresholds: Dict[DriftSeverity, float],
                         lower_is_worse: bool = False) -> Optional[DriftSeverity]:
        """Generic threshold checker for all drift metrics."""
        for level in [DriftSeverity.EMERGENCY, DriftSeverity.CRITICAL, DriftSeverity.WARNING]:
            threshold = thresholds[level]
            if (lower_is_worse and value <= threshold) or (not lower_is_worse and value >= threshold):
                return level
        return None

    def _check_psi_threshold(self, psi_score: float) -> Optional[DriftSeverity]:
        return self._check_threshold(psi_score, self.psi_thresholds)

    def _check_ks_threshold(self, p_value: float) -> Optional[DriftSeverity]:
        return self._check_threshold(p_value, self.ks_thresholds, lower_is_worse=True)

    def _check_js_threshold(self, js_divergence: float) -> Optional[DriftSeverity]:
        return self._check_threshold(js_divergence, self.js_thresholds)

    def _check_wasserstein_threshold(self, wasserstein_dist: float) -> Optional[DriftSeverity]:
        return self._check_threshold(wasserstein_dist, self.wasserstein_thresholds)
    
    def compute_drift_metrics(self, use_incremental: bool = False) -> DriftMetric:
        """
        Compute all drift metrics for current window and save to database.

        OPTIMIZED: 
        - Fetches baseline once and computes drift metrics in parallel
        - Supports incremental computation using pre-computed baseline stats
        - Includes categorical drift detection

        Args:
            use_incremental: If True, use pre-computed baseline stats (faster)

        Returns:
            DriftMetric object with computed values
        """
        start_time = time.time()
        window_start, window_end = self.get_current_window()

        # Default fallback values for failed drift computations
        default_input = {"psi_score": 0.0, "drift_detected": False, "sample_size": 0}
        default_output = {"ks_statistic": 0.0, "ks_p_value": 1.0, "drift_detected": False}
        default_embedding = {"js_divergence": 0.0, "wasserstein_distance": 0.0, "drift_detected": False}
        default_categorical = {"chi_square_statistic": 0.0, "chi_square_p_value": 1.0, "drift_detected": False}

        if use_incremental:
            # INCREMENTAL MODE: Use pre-computed baseline statistics
            with ThreadPoolExecutor(max_workers=3) as executor:
                input_future = executor.submit(self.detect_input_drift_incremental)
                output_future = executor.submit(self.detect_output_drift_incremental)
                categorical_future = executor.submit(self.detect_categorical_drift)

                input_drift = _safe_future_result(input_future, default_input)
                output_drift = _safe_future_result(output_future, default_output)
                categorical_drift = _safe_future_result(categorical_future, default_categorical)

            # Still need embeddings from full data for now
            baseline_queries = self.get_baseline_queries(use_cache=True)
            window_queries = self.get_window_queries(window_start, window_end)
            embedding_drift = self.detect_embedding_drift(baseline_queries, window_queries)
        else:
            # STANDARD MODE: Fetch baseline and window data once, reuse for all drift types
            baseline_queries = self.get_baseline_queries(use_cache=True)
            window_queries = self.get_window_queries(window_start, window_end)

            # Compute all drift metrics in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                input_future = executor.submit(self.detect_input_drift, baseline_queries, window_queries)
                output_future = executor.submit(self.detect_output_drift, baseline_queries, window_queries)
                embedding_future = executor.submit(self.detect_embedding_drift, baseline_queries, window_queries)
                categorical_future = executor.submit(self.detect_categorical_drift)

                input_drift = _safe_future_result(input_future, default_input)
                output_drift = _safe_future_result(output_future, default_output)
                embedding_drift = _safe_future_result(embedding_future, default_embedding)
                categorical_drift = _safe_future_result(categorical_future, default_categorical)

        computation_time_ms = (time.time() - start_time) * 1000

        # Create and save drift metric record
        drift_metric = DriftMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.INPUT_DRIFT,  # Primary type
            psi_score=input_drift.get("psi_score"),
            ks_statistic=output_drift.get("ks_statistic"),
            ks_p_value=output_drift.get("ks_p_value"),
            js_divergence=embedding_drift.get("js_divergence"),
            wasserstein_distance=embedding_drift.get("wasserstein_distance"),
            window_start=window_start,
            window_end=window_end,
            sample_size=input_drift.get("sample_size", 0)
        )

        self.db.add(drift_metric)
        self.db.commit()
        self.db.refresh(drift_metric)

        # Add metadata about computation for debugging/monitoring
        drift_metric._categorical_drift = categorical_drift
        drift_metric._computation_time_ms = computation_time_ms
        drift_metric._computation_mode = "incremental" if use_incremental else "standard"

        return drift_metric

    def compute_drift_metrics_comprehensive(self) -> Dict:
        """
        Compute comprehensive drift analysis including all metric types.
        
        Returns detailed results without saving to database (for API responses).
        
        Returns:
            Dictionary with all drift metrics and analysis
        """
        start_time = time.time()
        window_start, window_end = self.get_current_window()

        # Fetch data once
        baseline_queries = self.get_baseline_queries(use_cache=True)
        window_queries = self.get_window_queries(window_start, window_end)

        # Default fallback values
        default_input = {"psi_score": 0.0, "drift_detected": False, "sample_size": 0, "severity": None}
        default_output = {"ks_statistic": 0.0, "ks_p_value": 1.0, "drift_detected": False, "severity": None}
        default_embedding = {"js_divergence": 0.0, "wasserstein_distance": 0.0, "drift_detected": False, "severity": None}
        default_categorical = {"chi_square_statistic": 0.0, "chi_square_p_value": 1.0, "drift_detected": False, "severity": None}

        # Compute all metrics in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            input_future = executor.submit(self.detect_input_drift, baseline_queries, window_queries)
            output_future = executor.submit(self.detect_output_drift, baseline_queries, window_queries)
            embedding_future = executor.submit(self.detect_embedding_drift, baseline_queries, window_queries)
            categorical_future = executor.submit(self.detect_categorical_drift)

            input_drift = _safe_future_result(input_future, default_input)
            output_drift = _safe_future_result(output_future, default_output)
            embedding_drift = _safe_future_result(embedding_future, default_embedding)
            categorical_drift = _safe_future_result(categorical_future, default_categorical)

        computation_time_ms = (time.time() - start_time) * 1000

        # Determine overall severity
        severities = [
            input_drift.get("severity"),
            output_drift.get("severity"),
            embedding_drift.get("severity"),
            categorical_drift.get("severity")
        ]

        severity_order = {"emergency": 3, "critical": 2, "warning": 1, None: 0}
        max_severity = max(severities, key=lambda s: severity_order.get(s, 0))

        return {
            "timestamp": datetime.now().isoformat(),
            "window": {
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "minutes": self.window_minutes
            },
            "sample_sizes": {
                "baseline": len(baseline_queries),
                "window": len(window_queries)
            },
            "input_drift": input_drift,
            "output_drift": output_drift,
            "embedding_drift": embedding_drift,
            "categorical_drift": categorical_drift,
            "overall_severity": max_severity,
            "drift_detected": any([
                input_drift.get("drift_detected"),
                output_drift.get("drift_detected"),
                embedding_drift.get("drift_detected"),
                categorical_drift.get("drift_detected")
            ]),
            "computation_time_ms": round(computation_time_ms, 2)
        }
    
    def generate_alert_explanation(self, alert: DriftAlert, drift_metric: DriftMetric) -> Dict:
        """
        Generate human-readable explanation for why an alert occurred.
        PRD Requirement: Users should understand what drift means and what's going wrong.
        
        Returns:
            Dictionary with explanation, baseline comparison, and recommendations
        """
        baseline_start, baseline_end = self.get_baseline_period()
        
        explanation = {}
        
        if alert.metric_name == "psi_score":
            # Input drift explanation
            baseline_queries = self.get_baseline_queries()
            window_queries = self.get_window_queries(drift_metric.window_start, drift_metric.window_end)
            
            # Analyze query categories to see what's shifted
            baseline_categories = {}
            window_categories = {}
            for q in baseline_queries:
                cat = q.query_category or "general"
                baseline_categories[cat] = baseline_categories.get(cat, 0) + 1
            for q in window_queries:
                cat = q.query_category or "general"
                window_categories[cat] = window_categories.get(cat, 0) + 1
            
            # Find category shifts
            category_shifts = []
            baseline_len = len(baseline_queries) if baseline_queries else 0
            window_len = len(window_queries) if window_queries else 0
            for cat in set(list(baseline_categories.keys()) + list(window_categories.keys())):
                baseline_pct = (baseline_categories.get(cat, 0) / baseline_len * 100) if baseline_len > 0 else 0
                window_pct = (window_categories.get(cat, 0) / window_len * 100) if window_len > 0 else 0
                shift = window_pct - baseline_pct
                if abs(shift) > 5:  # Significant shift
                    category_shifts.append({
                        "category": cat,
                        "baseline_percentage": round(baseline_pct, 1),
                        "current_percentage": round(window_pct, 1),
                        "shift": round(shift, 1)
                    })
            
            explanation = {
                "metric_type": "input_drift",
                "what_it_means": f"PSI score of {alert.metric_value:.4f} indicates a significant shift in the distribution of input queries compared to baseline period.",
                "baseline_period": f"{baseline_start.strftime('%Y-%m-%d')} to {baseline_end.strftime('%Y-%m-%d')}",
                "current_period": f"{drift_metric.window_start.strftime('%Y-%m-%d %H:%M')} to {drift_metric.window_end.strftime('%Y-%m-%d %H:%M')}",
                "baseline_sample_size": len(baseline_queries),
                "current_sample_size": len(window_queries),
                "category_shifts": category_shifts,
                "severity_interpretation": {
                    "warning": "Input patterns are changing but may not affect performance yet.",
                    "critical": "Significant input shift detected. Model performance may be degrading.",
                    "emergency": "Major input distribution change. Immediate action recommended."
                }.get(alert.severity.value, ""),
                "recommendations": [
                    "Review recent query categories to understand the shift",
                    "Check if upstream systems have changed their behavior",
                    "Monitor output quality metrics for degradation",
                    "Consider retraining model if shift persists" if alert.severity.value in ["critical", "emergency"] else None
                ],
                "recommended_action": "investigate" if alert.severity.value == "warning" else "monitor_closely" if alert.severity.value == "critical" else "immediate_action"
            }
            explanation["recommendations"] = [r for r in explanation["recommendations"] if r is not None]
        
        elif alert.metric_name == "ks_p_value":
            # Output drift explanation
            explanation = {
                "metric_type": "output_drift",
                "what_it_means": f"KS test p-value of {alert.metric_value:.4f} indicates a significant change in output distribution (confidence scores or response quality).",
                "baseline_period": f"{baseline_start.strftime('%Y-%m-%d')} to {baseline_end.strftime('%Y-%m-%d')}",
                "current_period": f"{drift_metric.window_start.strftime('%Y-%m-%d %H:%M')} to {drift_metric.window_end.strftime('%Y-%m-%d %H:%M')}",
                "interpretation": "Lower p-value means the current output distribution is significantly different from baseline.",
                "severity_interpretation": {
                    "warning": "Output quality may be changing. Monitor response quality.",
                    "critical": "Significant output drift detected. Response quality likely degraded.",
                    "emergency": "Major output distribution change. Patient safety may be at risk."
                }.get(alert.severity.value, ""),
                "recommendations": [
                    "Review recent AI responses for quality issues",
                    "Check if model configuration has changed",
                    "Verify confidence score calibration",
                    "Consider rolling back to previous configuration" if alert.severity.value in ["critical", "emergency"] else None,
                    "Escalate to human review for affected queries" if alert.severity.value == "emergency" else None
                ],
                "recommended_action": "investigate" if alert.severity.value == "warning" else "rollback_consideration" if alert.severity.value == "critical" else "immediate_rollback"
            }
            explanation["recommendations"] = [r for r in explanation["recommendations"] if r is not None]
        
        elif alert.metric_name == "js_divergence":
            # Embedding drift explanation (JS divergence)
            explanation = {
                "metric_type": "embedding_drift",
                "what_it_means": f"JS divergence of {alert.metric_value:.4f} indicates the embedding space has shifted significantly.",
                "baseline_period": f"{baseline_start.strftime('%Y-%m-%d')} to {baseline_end.strftime('%Y-%m-%d')}",
                "current_period": f"{drift_metric.window_start.strftime('%Y-%m-%d %H:%M')} to {drift_metric.window_end.strftime('%Y-%m-%d %H:%M')}",
                "interpretation": "Embedding drift means the model's internal representations have changed, which affects similarity search.",
                "severity_interpretation": {
                    "warning": "Embedding space is shifting. Similarity search may become less accurate.",
                    "critical": "Significant embedding drift. Retrieval quality likely degraded.",
                    "emergency": "Major embedding space change. System may retrieve wrong information."
                }.get(alert.severity.value, ""),
                "recommendations": [
                    "Check if embedding model version has changed",
                    "Review similarity search results for accuracy",
                    "Validate that retrieved contexts are still relevant",
                    "Consider rolling back embedding model" if alert.severity.value in ["critical", "emergency"] else None,
                    "Update similarity thresholds if needed" if alert.severity.value == "warning" else None
                ],
                "recommended_action": "investigate" if alert.severity.value == "warning" else "rollback_consideration" if alert.severity.value == "critical" else "immediate_rollback"
            }
            explanation["recommendations"] = [r for r in explanation["recommendations"] if r is not None]

        elif alert.metric_name == "wasserstein_distance":
            # Embedding drift explanation (Wasserstein distance - more accurate)
            explanation = {
                "metric_type": "embedding_drift",
                "what_it_means": f"Wasserstein distance of {alert.metric_value:.4f} indicates semantic drift in the embedding space (Earth Mover's Distance).",
                "baseline_period": f"{baseline_start.strftime('%Y-%m-%d')} to {baseline_end.strftime('%Y-%m-%d')}",
                "current_period": f"{drift_metric.window_start.strftime('%Y-%m-%d %H:%M')} to {drift_metric.window_end.strftime('%Y-%m-%d %H:%M')}",
                "interpretation": "Wasserstein distance measures the minimum 'cost' to transform baseline embeddings into current embeddings. Higher values indicate significant semantic shift in how the model represents queries, affecting retrieval accuracy.",
                "metric_advantage": "Wasserstein distance is more accurate than JS divergence for high-dimensional embeddings as it preserves geometric structure.",
                "severity_interpretation": {
                    "warning": "Moderate semantic shift detected. Similarity search accuracy may be declining.",
                    "critical": "Significant semantic drift. Embedding-based retrieval likely degraded.",
                    "emergency": "Major semantic space transformation. Critical risk of incorrect information retrieval."
                }.get(alert.severity.value, ""),
                "recommendations": [
                    "Verify embedding model version and configuration",
                    "Review similarity search results for semantic accuracy",
                    "Check if retrieved documents still match query intent",
                    "Analyze query distribution changes that may affect embeddings",
                    "Consider rolling back embedding model or retraining" if alert.severity.value in ["critical", "emergency"] else None,
                    "Adjust similarity thresholds based on new distribution" if alert.severity.value == "warning" else None
                ],
                "recommended_action": "investigate" if alert.severity.value == "warning" else "rollback_consideration" if alert.severity.value == "critical" else "immediate_rollback"
            }
            explanation["recommendations"] = [r for r in explanation["recommendations"] if r is not None]

        return explanation
    
    def detect_segment_drift(self, segment_field: str = "query_category") -> Dict:
        """
        Detect drift at segment level (by query category, department, etc.).
        
        PRD Requirement: Segment-level monitoring to catch drift affecting specific 
        patient populations or query types even if aggregate metrics look fine.
        
        Args:
            segment_field: Field to segment by (query_category, department, patient_population)
            
        Returns:
            Dictionary with per-segment drift metrics
        """
        baseline_start, baseline_end = self.get_baseline_period()
        window_start, window_end = self.get_current_window()
        
        # Get baseline queries grouped by segment
        baseline_queries = self.get_baseline_queries(use_cache=True)
        window_queries = self.get_window_queries(window_start, window_end)
        
        # Group by segment
        baseline_segments = {}
        window_segments = {}
        
        for q in baseline_queries:
            segment = getattr(q, segment_field, None) or "unknown"
            if segment not in baseline_segments:
                baseline_segments[segment] = []
            baseline_segments[segment].append(q)
        
        for q in window_queries:
            segment = getattr(q, segment_field, None) or "unknown"
            if segment not in window_segments:
                window_segments[segment] = []
            window_segments[segment].append(q)
        
        # Compute drift per segment
        segment_results = {}
        all_segments = set(list(baseline_segments.keys()) + list(window_segments.keys()))
        
        alerts_to_create = []
        
        for segment in all_segments:
            baseline_seg = baseline_segments.get(segment, [])
            window_seg = window_segments.get(segment, [])
            
            # Skip if insufficient data
            if len(baseline_seg) < 10 or len(window_seg) < 5:
                segment_results[segment] = {
                    "status": "insufficient_data",
                    "baseline_count": len(baseline_seg),
                    "window_count": len(window_seg),
                }
                continue
            
            # Compute drift metrics for this segment
            input_drift = self.detect_input_drift(baseline_seg, window_seg)
            output_drift = self.detect_output_drift(baseline_seg, window_seg)
            
            # Calculate distribution shift
            baseline_total = len(baseline_queries) if baseline_queries else 0
            window_total = len(window_queries) if window_queries else 0
            baseline_pct = (len(baseline_seg) / baseline_total * 100) if baseline_total > 0 else 0
            window_pct = (len(window_seg) / window_total * 100) if window_total > 0 else 0
            distribution_shift = window_pct - baseline_pct
            
            # Determine segment health
            segment_health = "healthy"
            if input_drift.get("drift_detected") or output_drift.get("drift_detected"):
                segment_health = "degraded"
                severity = input_drift.get("severity") or output_drift.get("severity")
                if severity in ["critical", "emergency"]:
                    segment_health = "critical"
            
            segment_results[segment] = {
                "status": segment_health,
                "baseline_count": len(baseline_seg),
                "window_count": len(window_seg),
                "baseline_percentage": round(baseline_pct, 2),
                "window_percentage": round(window_pct, 2),
                "distribution_shift": round(distribution_shift, 2),
                "input_drift": input_drift,
                "output_drift": output_drift,
            }
            
            # Generate segment-specific alert if drift detected
            if segment_health in ["degraded", "critical"]:
                severity_enum = DriftSeverity.WARNING
                if segment_health == "critical":
                    severity_enum = DriftSeverity.CRITICAL
                
                alerts_to_create.append({
                    "segment": segment,
                    "severity": severity_enum,
                    "psi": input_drift.get("psi_score", 0),
                    "shift": distribution_shift,
                })
        
        # Calculate overall segment health
        healthy_segments = sum(1 for s in segment_results.values() if s.get("status") == "healthy")
        degraded_segments = sum(1 for s in segment_results.values() if s.get("status") == "degraded")
        critical_segments = sum(1 for s in segment_results.values() if s.get("status") == "critical")
        
        return {
            "segment_field": segment_field,
            "total_segments": len(segment_results),
            "healthy_segments": healthy_segments,
            "degraded_segments": degraded_segments,
            "critical_segments": critical_segments,
            "overall_health": "critical" if critical_segments > 0 else "degraded" if degraded_segments > 0 else "healthy",
            "segments": segment_results,
            "alerts_generated": len(alerts_to_create),
            "window": {
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
            },
            "baseline": {
                "start": baseline_start.isoformat(),
                "end": baseline_end.isoformat(),
            },
        }
    
    def get_drift_summary(self) -> Dict:
        """
        Get comprehensive drift summary for dashboard.
        
        Returns:
            Summary with aggregate metrics, segment health, and recommendations
        """
        # Get latest drift metric
        latest_metric = self.db.query(DriftMetric).order_by(
            DriftMetric.timestamp.desc()
        ).first()
        
        # Get active alerts
        active_alerts = self.db.query(DriftAlert).filter(
            DriftAlert.status == "active"
        ).all()
        
        # Get segment drift
        segment_drift = self.detect_segment_drift()
        
        # Compute overall system health
        if any(a.severity == DriftSeverity.EMERGENCY for a in active_alerts):
            system_health = "emergency"
            recommendation = "Immediate action required. Consider automated rollback."
        elif any(a.severity == DriftSeverity.CRITICAL for a in active_alerts):
            system_health = "critical"
            recommendation = "Critical drift detected. Review and consider rollback."
        elif any(a.severity == DriftSeverity.WARNING for a in active_alerts):
            system_health = "warning"
            recommendation = "Monitor closely. Investigate root cause of drift."
        else:
            system_health = "healthy"
            recommendation = "System operating within normal parameters."
        
        return {
            "system_health": system_health,
            "recommendation": recommendation,
            "active_alerts": len(active_alerts),
            "alerts_by_severity": {
                "emergency": sum(1 for a in active_alerts if a.severity == DriftSeverity.EMERGENCY),
                "critical": sum(1 for a in active_alerts if a.severity == DriftSeverity.CRITICAL),
                "warning": sum(1 for a in active_alerts if a.severity == DriftSeverity.WARNING),
            },
            "latest_metrics": {
                "psi_score": latest_metric.psi_score if latest_metric else None,
                "ks_p_value": latest_metric.ks_p_value if latest_metric else None,
                "js_divergence": latest_metric.js_divergence if latest_metric else None,
                "timestamp": latest_metric.timestamp.isoformat() if latest_metric else None,
            },
            "segment_health": {
                "overall": segment_drift.get("overall_health"),
                "healthy": segment_drift.get("healthy_segments"),
                "degraded": segment_drift.get("degraded_segments"),
                "critical": segment_drift.get("critical_segments"),
            },
        }

    def check_thresholds_and_alert(self, drift_metric: DriftMetric) -> List[DriftAlert]:
        """
        Check computed drift metrics against thresholds and generate alerts.
        PRD Requirement DD-006: Configurable alert thresholds with three severity levels.
        """
        alerts = []

        # Metric configs: (attr, metric_type, metric_name, thresholds, threshold_checker)
        metric_configs = [
            ('psi_score', MetricType.INPUT_DRIFT, 'psi_score', self.psi_thresholds, self._check_psi_threshold),
            ('ks_p_value', MetricType.OUTPUT_DRIFT, 'ks_p_value', self.ks_thresholds, self._check_ks_threshold),
            ('js_divergence', MetricType.EMBEDDING_DRIFT, 'js_divergence', self.js_thresholds, self._check_js_threshold),
            ('wasserstein_distance', MetricType.EMBEDDING_DRIFT, 'wasserstein_distance', self.wasserstein_thresholds, self._check_wasserstein_threshold),
        ]

        for attr, metric_type, metric_name, thresholds, check_fn in metric_configs:
            value = getattr(drift_metric, attr, None)
            if value is not None:
                severity = check_fn(value)
                if severity:
                    alerts.append(DriftAlert(
                        metric_type=metric_type,
                        metric_name=metric_name,
                        metric_value=value,
                        threshold_value=thresholds[severity],
                        severity=severity,
                        drift_metric_id=drift_metric.id
                    ))

        for alert in alerts:
            self.db.add(alert)
        self.db.commit()

        return alerts
