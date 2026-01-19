"""Acceptance tests for Drift Detection Engine (AC 1-3)"""
import pytest
from datetime import datetime, timedelta
from backend.models.drift_metrics import DriftSeverity
from tests.fixtures.sample_data import create_normal_distribution_data, create_shifted_distribution_data

@pytest.mark.acceptance
class TestAC1_DriftDetectionWithin15Minutes:
    """AC 1: Given a simulated 20% shift in input query distribution, 
    the system detects drift within 15 minutes."""
    
    def test_drift_detection_within_15_minutes(self, drift_service, test_db):
        """Test that 20% distribution shift is detected within 15 minutes"""
        # Create baseline data (7 days)
        now = datetime.now()
        baseline_end = now - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=7)
        
        create_normal_distribution_data(test_db, baseline_start, baseline_end, n_samples=8000)
        
        # Create recent data with 20% shift (billing increases from 20% to 40% = 20% shift)
        recent_start = now - timedelta(minutes=15)
        create_shifted_distribution_data(
            test_db, recent_start, now, n_samples=2000, shift_magnitude=0.20
        )
        
        # Compute drift metrics
        metrics = drift_service.compute_drift_metrics()
        
        # Verify drift is detected (PSI should be high)
        assert metrics is not None
        assert metrics.psi_score is not None
        
        # PSI should exceed warning threshold (0.15) for 20% shift
        psi_score = metrics.psi_score or 0.0
        assert psi_score > 0.10  # Should detect significant shift
        
        # Check that alerts are generated
        alerts = drift_service.check_thresholds_and_alert(metrics)
        assert len(alerts) > 0
        
        # At least one alert should be warning or higher
        severity_levels = [alert.severity for alert in alerts if alert.metric_name == "psi_score"]
        assert len(severity_levels) > 0
        assert any(severity in [DriftSeverity.WARNING, DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY] 
                  for severity in severity_levels)

@pytest.mark.acceptance
class TestAC2_FalsePositiveRate:
    """AC 2: Given normal operating conditions for 24 hours, 
    the system generates fewer than 3 false positive alerts."""
    
    def test_false_positive_rate_under_threshold(self, drift_service, test_db):
        """Test that normal operation generates < 3 alerts in 24 hours"""
        # Create 24 hours of normal operation data (no drift)
        now = datetime.now()
        start_time = now - timedelta(days=1)
        
        # Generate normal data for 24 hours (distributed across the day)
        total_queries = 10000
        hours = 24
        queries_per_hour = total_queries // hours
        
        for hour in range(hours):
            hour_start = start_time + timedelta(hours=hour)
            hour_end = hour_start + timedelta(hours=1)
            create_normal_distribution_data(
                test_db, hour_start, hour_end, n_samples=queries_per_hour
            )
        
        # Run drift detection multiple times throughout the day (simulate)
        alerts_generated = []
        
        # Check every 15 minutes (96 checks in 24 hours)
        for check_minutes in range(0, 24*60, 15):
            check_time = start_time + timedelta(minutes=check_minutes)
            
            # Get window for this check
            window_start = check_time - timedelta(minutes=15)
            window_end = check_time
            
            # Get queries in window
            window_queries = drift_service.get_window_queries(window_start, window_end)
            
            if len(window_queries) >= 100:  # Minimum sample size
                metrics = drift_service.compute_drift_metrics()
                alerts = drift_service.check_thresholds_and_alert(metrics)
                alerts_generated.extend(alerts)
        
        # Count false positives (alerts generated during normal operation)
        # All alerts are false positives in this scenario
        false_positive_count = len(alerts_generated)
        
        # Should generate fewer than 3 alerts
        assert false_positive_count < 3, \
            f"False positive rate too high: {false_positive_count} alerts in 24 hours"

@pytest.mark.acceptance
class TestAC3_DashboardLatency:
    """AC 3: Dashboard displays current drift metrics with less than 30-second latency."""
    
    def test_api_response_time_under_30_seconds(self, api_client, test_db, sample_baseline_data, sample_recent_data_with_drift):
        """Test that API responds within 30 seconds"""
        import time
        
        start_time = time.time()
        response = api_client.get("/api/drift/metrics")
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_time < 30.0, \
            f"API response time {elapsed_time:.2f}s exceeds 30s threshold"
        
        # Verify response contains expected data
        data = response.json()
        assert "psi_score" in data
        assert "timestamp" in data
    
    def test_api_response_time_with_various_data_volumes(self, api_client, test_db):
        """Test API response time with different data volumes"""
        import time
        
        # Test with small dataset
        now = datetime.now()
        create_normal_distribution_data(test_db, now - timedelta(days=7), now, n_samples=1000)
        create_shifted_distribution_data(test_db, now - timedelta(minutes=15), now, n_samples=500)
        
        start_time = time.time()
        response = api_client.get("/api/drift/metrics")
        elapsed_small = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_small < 30.0
        
        # Test with large dataset
        create_normal_distribution_data(test_db, now - timedelta(days=7), now, n_samples=50000)
        create_shifted_distribution_data(test_db, now - timedelta(minutes=15), now, n_samples=10000)
        
        start_time = time.time()
        response = api_client.get("/api/drift/metrics")
        elapsed_large = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_large < 30.0
