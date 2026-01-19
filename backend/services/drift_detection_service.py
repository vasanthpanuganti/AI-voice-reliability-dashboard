"""Drift Detection Service - Core service for detecting model drift"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_
from concurrent.futures import ThreadPoolExecutor

from backend.models.query_log import QueryLog
from backend.models.drift_metrics import DriftMetric, DriftAlert, DriftSeverity, MetricType
from backend.utils.drift_statistics import calculate_psi, compute_ks_test, compute_jensen_shannon, compute_wasserstein_distance
from backend.config import settings

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
    
    def detect_input_drift(self, baseline_queries: List[QueryLog], window_queries: List[QueryLog]) -> Dict:
        """
        Detect input drift (query distribution changes) using PSI.

        Args:
            baseline_queries: Pre-fetched baseline queries
            window_queries: Pre-fetched window queries

        Returns:
            Dictionary with PSI score and drift detection result
        """
        if len(baseline_queries) < 10 or len(window_queries) < 5:
            return {
                "psi_score": 0.0,
                "sample_size": len(window_queries),
                "baseline_size": len(baseline_queries),
                "drift_detected": False
            }

        # Extract query features for comparison (vectorized)
        baseline_lengths = np.array([len(q.query) for q in baseline_queries])
        window_lengths = np.array([len(q.query) for q in window_queries])

        # Calculate PSI
        psi_score = calculate_psi(baseline_lengths, window_lengths)

        # Check threshold
        severity = self._check_psi_threshold(psi_score)
        drift_detected = severity is not None

        return {
            "psi_score": psi_score,
            "sample_size": len(window_queries),
            "baseline_size": len(baseline_queries),
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None
        }
    
    def detect_output_drift(self, baseline_queries: List[QueryLog], window_queries: List[QueryLog]) -> Dict:
        """
        Detect output drift (response quality changes) using KS test.

        Args:
            baseline_queries: Pre-fetched baseline queries
            window_queries: Pre-fetched window queries

        Returns:
            Dictionary with KS test results
        """
        if len(baseline_queries) < 10 or len(window_queries) < 5:
            return {
                "ks_statistic": 0.0,
                "ks_p_value": 1.0,
                "sample_size": len(window_queries),
                "drift_detected": False
            }

        # Use confidence scores if available, otherwise use response length (vectorized)
        baseline_values = np.array([
            float(q.confidence_score) if q.confidence_score else len(q.ai_response or "")
            for q in baseline_queries
        ])

        window_values = np.array([
            float(q.confidence_score) if q.confidence_score else len(q.ai_response or "")
            for q in window_queries
        ])

        # Perform KS test
        ks_statistic, ks_p_value = compute_ks_test(window_values, baseline_values)

        # Lower p-value means more significant difference
        # Check threshold (we want to detect when p-value is below threshold)
        severity = self._check_ks_threshold(ks_p_value)
        drift_detected = severity is not None

        return {
            "ks_statistic": ks_statistic,
            "ks_p_value": ks_p_value,
            "sample_size": len(window_queries),
            "drift_detected": drift_detected,
            "severity": severity.value if severity else None
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
    
    def _check_psi_threshold(self, psi_score: float) -> Optional[DriftSeverity]:
        """Check PSI score against thresholds"""
        if psi_score >= self.psi_thresholds[DriftSeverity.EMERGENCY]:
            return DriftSeverity.EMERGENCY
        elif psi_score >= self.psi_thresholds[DriftSeverity.CRITICAL]:
            return DriftSeverity.CRITICAL
        elif psi_score >= self.psi_thresholds[DriftSeverity.WARNING]:
            return DriftSeverity.WARNING
        return None
    
    def _check_ks_threshold(self, p_value: float) -> Optional[DriftSeverity]:
        """Check KS test p-value against thresholds (lower is worse)"""
        if p_value <= self.ks_thresholds[DriftSeverity.EMERGENCY]:
            return DriftSeverity.EMERGENCY
        elif p_value <= self.ks_thresholds[DriftSeverity.CRITICAL]:
            return DriftSeverity.CRITICAL
        elif p_value <= self.ks_thresholds[DriftSeverity.WARNING]:
            return DriftSeverity.WARNING
        return None
    
    def _check_js_threshold(self, js_divergence: float) -> Optional[DriftSeverity]:
        """Check JS divergence against thresholds"""
        if js_divergence >= self.js_thresholds[DriftSeverity.EMERGENCY]:
            return DriftSeverity.EMERGENCY
        elif js_divergence >= self.js_thresholds[DriftSeverity.CRITICAL]:
            return DriftSeverity.CRITICAL
        elif js_divergence >= self.js_thresholds[DriftSeverity.WARNING]:
            return DriftSeverity.WARNING
        return None

    def _check_wasserstein_threshold(self, wasserstein_dist: float) -> Optional[DriftSeverity]:
        """Check Wasserstein distance against thresholds"""
        if wasserstein_dist >= self.wasserstein_thresholds[DriftSeverity.EMERGENCY]:
            return DriftSeverity.EMERGENCY
        elif wasserstein_dist >= self.wasserstein_thresholds[DriftSeverity.CRITICAL]:
            return DriftSeverity.CRITICAL
        elif wasserstein_dist >= self.wasserstein_thresholds[DriftSeverity.WARNING]:
            return DriftSeverity.WARNING
        return None
    
    def compute_drift_metrics(self) -> DriftMetric:
        """
        Compute all drift metrics for current window and save to database.

        OPTIMIZED: Fetches baseline once and computes drift metrics in parallel.

        Returns:
            DriftMetric object with computed values
        """
        window_start, window_end = self.get_current_window()

        # OPTIMIZATION: Fetch baseline and window data once, reuse for all drift types
        baseline_queries = self.get_baseline_queries(use_cache=True)
        window_queries = self.get_window_queries(window_start, window_end)

        # OPTIMIZATION: Compute all three drift metrics in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            input_future = executor.submit(self.detect_input_drift, baseline_queries, window_queries)
            output_future = executor.submit(self.detect_output_drift, baseline_queries, window_queries)
            embedding_future = executor.submit(self.detect_embedding_drift, baseline_queries, window_queries)

            input_drift = input_future.result()
            output_drift = output_future.result()
            embedding_drift = embedding_future.result()

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

        return drift_metric
    
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
            for cat in set(list(baseline_categories.keys()) + list(window_categories.keys())):
                baseline_pct = (baseline_categories.get(cat, 0) / len(baseline_queries) * 100) if baseline_queries else 0
                window_pct = (window_categories.get(cat, 0) / len(window_queries) * 100) if window_queries else 0
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
    
    def check_thresholds_and_alert(self, drift_metric: DriftMetric) -> List[DriftAlert]:
        """
        Check computed drift metrics against thresholds and generate alerts.
        PRD Requirement DD-006: Configurable alert thresholds with three severity levels.
        
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check PSI (input drift)
        if drift_metric.psi_score is not None:
            severity = self._check_psi_threshold(drift_metric.psi_score)
            if severity:
                alert = DriftAlert(
                    metric_type=MetricType.INPUT_DRIFT,
                    metric_name="psi_score",
                    metric_value=drift_metric.psi_score,
                    threshold_value=self.psi_thresholds[severity],
                    severity=severity,
                    drift_metric_id=drift_metric.id
                )
                alerts.append(alert)
        
        # Check KS test (output drift)
        if drift_metric.ks_p_value is not None:
            severity = self._check_ks_threshold(drift_metric.ks_p_value)
            if severity:
                alert = DriftAlert(
                    metric_type=MetricType.OUTPUT_DRIFT,
                    metric_name="ks_p_value",
                    metric_value=drift_metric.ks_p_value,
                    threshold_value=self.ks_thresholds[severity],
                    severity=severity,
                    drift_metric_id=drift_metric.id
                )
                alerts.append(alert)
        
        # Check JS divergence (embedding drift)
        if drift_metric.js_divergence is not None:
            severity = self._check_js_threshold(drift_metric.js_divergence)
            if severity:
                alert = DriftAlert(
                    metric_type=MetricType.EMBEDDING_DRIFT,
                    metric_name="js_divergence",
                    metric_value=drift_metric.js_divergence,
                    threshold_value=self.js_thresholds[severity],
                    severity=severity,
                    drift_metric_id=drift_metric.id
                )
                alerts.append(alert)

        # Check Wasserstein distance (embedding drift - primary metric)
        if drift_metric.wasserstein_distance is not None:
            severity = self._check_wasserstein_threshold(drift_metric.wasserstein_distance)
            if severity:
                alert = DriftAlert(
                    metric_type=MetricType.EMBEDDING_DRIFT,
                    metric_name="wasserstein_distance",
                    metric_value=drift_metric.wasserstein_distance,
                    threshold_value=self.wasserstein_thresholds[severity],
                    severity=severity,
                    drift_metric_id=drift_metric.id
                )
                alerts.append(alert)

        # Save alerts to database
        for alert in alerts:
            self.db.add(alert)

        self.db.commit()

        # Note: Automated rollback on critical/emergency alerts is handled in the API endpoint
        # to avoid circular imports. See backend/api/drift.py get_current_metrics endpoint

        return alerts
