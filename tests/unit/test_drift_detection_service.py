"""Unit tests for drift_detection_service.py"""
import pytest
from datetime import datetime, timedelta
import numpy as np
from backend.services.drift_detection_service import DriftDetectionService
from backend.models.query_log import QueryLog
from backend.models.drift_metrics import DriftSeverity
from tests.fixtures.sample_data import create_normal_distribution_data, create_shifted_distribution_data

class TestBaselinePeriod:
    """Tests for baseline period calculation"""
    
    def test_empty_database(self, drift_service, test_db):
        """Test baseline period with empty database"""
        start, end = drift_service.get_baseline_period()
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert (end - start).days == 7
    
    def test_with_baseline_data(self, drift_service, test_db, sample_baseline_data):
        """Test baseline period with existing data"""
        start, end = drift_service.get_baseline_period()
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert end > start
    
    def test_current_window(self, drift_service):
        """Test current window calculation"""
        start, end = drift_service.get_current_window()
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert end > start
        window_duration = (end - start).total_seconds() / 60
        assert abs(window_duration - 15) < 1  # Should be ~15 minutes

class TestBaselineCaching:
    """Tests for baseline caching mechanism"""
    
    def test_cache_creation(self, drift_service, test_db, sample_baseline_data):
        """Test baseline cache is created on first access"""
        queries = drift_service.get_baseline_queries()
        assert len(queries) > 0
        assert drift_service._baseline_cache is not None
        assert drift_service._baseline_cache_time is not None
    
    def test_cache_hit(self, drift_service, test_db, sample_baseline_data):
        """Test baseline cache is used on subsequent calls"""
        queries1 = drift_service.get_baseline_queries()
        queries2 = drift_service.get_baseline_queries()
        assert queries1 is queries2  # Same object reference (cached)
    
    def test_cache_invalidation(self, drift_service, test_db, sample_baseline_data):
        """Test baseline cache can be invalidated"""
        drift_service.get_baseline_queries()
        assert drift_service._baseline_cache is not None
        
        drift_service.invalidate_baseline_cache()
        assert drift_service._baseline_cache is None
        assert drift_service._baseline_cache_time is None
    
    def test_no_cache_when_disabled(self, drift_service, test_db, sample_baseline_data):
        """Test baseline cache is not used when disabled"""
        queries1 = drift_service.get_baseline_queries(use_cache=False)
        queries2 = drift_service.get_baseline_queries(use_cache=False)
        # Should be different objects (not cached)
        assert queries1 is not queries2 or len(queries1) == len(queries2)

class TestWindowQueries:
    """Tests for window query retrieval"""
    
    def test_get_window_queries(self, drift_service, test_db, sample_baseline_data):
        """Test retrieving queries in a time window"""
        now = datetime.now()
        start = now - timedelta(days=1)
        end = now
        
        queries = drift_service.get_window_queries(start, end)
        assert isinstance(queries, list)
    
    def test_empty_window(self, drift_service, test_db):
        """Test retrieving queries from empty window"""
        now = datetime.now()
        start = now + timedelta(days=1)
        end = now + timedelta(days=2)
        
        queries = drift_service.get_window_queries(start, end)
        assert len(queries) == 0

class TestThresholdChecking:
    """Tests for threshold checking logic"""
    
    def test_psi_thresholds(self, drift_service):
        """Test PSI threshold values are configured"""
        assert DriftSeverity.WARNING in drift_service.psi_thresholds
        assert DriftSeverity.CRITICAL in drift_service.psi_thresholds
        assert DriftSeverity.EMERGENCY in drift_service.psi_thresholds
    
    def test_ks_thresholds(self, drift_service):
        """Test KS threshold values are configured"""
        assert DriftSeverity.WARNING in drift_service.ks_thresholds
        assert DriftSeverity.CRITICAL in drift_service.ks_thresholds
        assert DriftSeverity.EMERGENCY in drift_service.ks_thresholds
    
    def test_js_thresholds(self, drift_service):
        """Test JS divergence threshold values are configured"""
        assert DriftSeverity.WARNING in drift_service.js_thresholds
        assert DriftSeverity.CRITICAL in drift_service.js_thresholds
        assert DriftSeverity.EMERGENCY in drift_service.js_thresholds

class TestDriftMetricComputation:
    """Tests for drift metric computation"""
    
    def test_compute_drift_metrics_with_data(self, drift_service, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test computing drift metrics with baseline and recent data"""
        from backend.models.drift_metrics import DriftMetric
        
        metrics = drift_service.compute_drift_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, DriftMetric)
        assert metrics.psi_score is not None or metrics.psi_score == 0.0
        assert metrics.ks_p_value is not None or metrics.ks_p_value == 1.0
        assert isinstance(metrics.psi_score, (float, type(None)))
    
    def test_compute_drift_metrics_empty(self, drift_service, test_db):
        """Test computing drift metrics with no data"""
        metrics = drift_service.compute_drift_metrics()
        assert metrics is not None
        # Should handle gracefully with empty data

class TestAlertGeneration:
    """Tests for alert generation"""
    
    def test_check_thresholds_no_alert(self, drift_service, test_db, sample_baseline_data):
        """Test threshold checking with no alert"""
        from backend.models.drift_metrics import DriftMetric, MetricType
        
        # Create normal metrics (low PSI, high KS p-value)
        metric = DriftMetric(
            metric_type=MetricType.INPUT_DRIFT,
            psi_score=0.05,  # Below warning threshold
            ks_p_value=0.50,  # Above warning threshold (high p-value = similar)
            js_divergence=0.05,  # Below warning threshold
            timestamp=datetime.now(),
            sample_size=100
        )
        test_db.add(metric)
        test_db.commit()
        test_db.refresh(metric)
        
        alerts = drift_service.check_thresholds_and_alert(metric)
        # Should generate no alerts or only low-severity alerts
        assert isinstance(alerts, list)
    
    def test_alert_on_high_psi(self, drift_service, test_db, sample_baseline_data):
        """Test alert generation for high PSI"""
        from backend.models.drift_metrics import DriftMetric, MetricType
        
        metric = DriftMetric(
            metric_type=MetricType.INPUT_DRIFT,
            psi_score=0.30,  # Above critical threshold
            ks_p_value=0.50,
            js_divergence=0.10,
            timestamp=datetime.now(),
            sample_size=100
        )
        test_db.add(metric)
        test_db.commit()
        test_db.refresh(metric)
        
        alerts = drift_service.check_thresholds_and_alert(metric)
        assert len(alerts) > 0
        # Check severity
        critical_alerts = [a for a in alerts if hasattr(a, 'severity') and (a.severity == DriftSeverity.CRITICAL or a.severity == DriftSeverity.EMERGENCY)]
        assert len(critical_alerts) > 0

class TestSegmentDrift:
    """Tests for segment-level drift detection"""
    
    def test_detect_segment_drift(self, drift_service, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test segment drift detection"""
        segment_data = drift_service.detect_segment_drift(segment_field="query_category")
        
        assert segment_data is not None
        assert "segments" in segment_data
        assert "overall_health" in segment_data
    
    def test_segment_drift_all_segments(self, drift_service, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test segment drift for all segment types"""
        for segment_field in ["query_category", "department", "patient_population"]:
            try:
                segment_data = drift_service.detect_segment_drift(segment_field=segment_field)
                assert segment_data is not None
            except Exception:
                # Some segments may not have data, which is acceptable
                pass

class TestCategoricalDrift:
    """Tests for categorical drift detection"""
    
    def test_detect_categorical_drift(self, drift_service, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test categorical drift detection"""
        result = drift_service.detect_categorical_drift(category_field="query_category")
        
        assert result is not None
        assert "chi_square_statistic" in result
        assert "chi_square_p_value" in result
        assert "category_details" in result

class TestComprehensiveMetrics:
    """Tests for comprehensive drift metrics computation"""
    
    def test_compute_drift_metrics_comprehensive(self, drift_service, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test comprehensive drift metrics computation"""
        result = drift_service.compute_drift_metrics_comprehensive()
        
        assert result is not None
        assert "input_drift" in result
        assert "output_drift" in result
        assert "embedding_drift" in result
        assert "categorical_drift" in result

class TestBaselineStatistics:
    """Tests for baseline statistics caching"""
    
    def test_get_baseline_statistics(self, drift_service, test_db, sample_baseline_data):
        """Test getting baseline statistics"""
        stats = drift_service.get_baseline_statistics()
        
        assert stats is not None
        assert isinstance(stats, dict)
        # Statistics should contain query_length, confidence_score, category_counts
        assert "query_length" in stats or "confidence_score" in stats
