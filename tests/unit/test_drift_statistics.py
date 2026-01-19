"""Unit tests for drift_statistics.py"""
import pytest
import numpy as np
from backend.utils.drift_statistics import (
    calculate_psi,
    compute_ks_test,
    compute_jensen_shannon,
    compute_wasserstein_distance,
    compute_chi_square,
    compute_psi_incremental,
    create_histogram_cache,
    stratified_subsample,
    compute_distribution_features
)

class TestCalculatePSI:
    """Tests for calculate_psi function"""
    
    def test_empty_arrays(self):
        """Test PSI with empty arrays"""
        assert calculate_psi(np.array([]), np.array([])) == 0.0
        assert calculate_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert calculate_psi(np.array([1, 2, 3]), np.array([])) == 0.0
    
    def test_identical_distributions(self):
        """Test PSI with identical distributions should be low"""
        expected = np.array([1, 2, 3, 4, 5] * 20)
        actual = np.array([1, 2, 3, 4, 5] * 20)
        psi = calculate_psi(expected, actual)
        assert psi >= 0.0
        assert psi < 0.01  # Should be very low for identical distributions
    
    def test_single_value(self):
        """Test PSI with single value arrays"""
        expected = np.array([5.0])
        actual = np.array([5.0])
        psi = calculate_psi(expected, actual)
        assert psi == 0.0  # Same value should return 0
    
    def test_single_value_different(self):
        """Test PSI with single different values"""
        expected = np.array([5.0])
        actual = np.array([10.0])
        # Should handle gracefully
        psi = calculate_psi(expected, actual)
        assert psi >= 0.0
    
    def test_constant_arrays(self):
        """Test PSI with constant arrays"""
        expected = np.ones(100) * 5.0
        actual = np.ones(100) * 5.0
        psi = calculate_psi(expected, actual)
        assert psi == 0.0
    
    def test_shifted_distribution(self):
        """Test PSI detects distribution shift"""
        # Normal distribution
        expected = np.random.normal(0, 1, 1000)
        # Shifted distribution
        actual = np.random.normal(2, 1, 1000)
        psi = calculate_psi(expected, actual)
        assert psi > 0.0
        assert psi < 10.0  # Should be reasonable
    
    def test_different_scales(self):
        """Test PSI with different scales"""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 3, 1000)  # Different variance
        psi = calculate_psi(expected, actual)
        assert psi > 0.0
    
    def test_custom_bins(self):
        """Test PSI with custom number of bins"""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(1, 1, 1000)
        psi_10 = calculate_psi(expected, actual, bins=10)
        psi_20 = calculate_psi(expected, actual, bins=20)
        # Should be similar but not identical
        assert abs(psi_10 - psi_20) < 1.0

class TestComputeKSTest:
    """Tests for compute_ks_test function"""
    
    def test_empty_arrays(self):
        """Test KS test with empty arrays"""
        stat, p_value = compute_ks_test(np.array([]), np.array([]))
        assert stat == 0.0
        assert p_value == 1.0
    
    def test_identical_distributions(self):
        """Test KS test with identical distributions should have high p-value"""
        baseline = np.random.normal(0, 1, 1000)
        current = baseline.copy()
        stat, p_value = compute_ks_test(current, baseline)
        assert p_value > 0.9  # Very similar distributions
    
    def test_different_distributions(self):
        """Test KS test detects different distributions"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(3, 1, 1000)  # Shifted
        stat, p_value = compute_ks_test(current, baseline)
        assert stat > 0.0
        assert p_value < 0.05  # Should detect difference
    
    def test_single_values(self):
        """Test KS test with single values"""
        baseline = np.array([5.0])
        current = np.array([5.0])
        stat, p_value = compute_ks_test(current, baseline)
        assert isinstance(stat, float)
        assert isinstance(p_value, float)
    
    def test_various_distributions(self):
        """Test KS test with various distribution shapes"""
        # Uniform vs Normal
        baseline = np.random.uniform(0, 10, 1000)
        current = np.random.normal(5, 2, 1000)
        stat, p_value = compute_ks_test(current, baseline)
        assert stat >= 0.0
        assert 0.0 <= p_value <= 1.0

class TestComputeJensenShannon:
    """Tests for compute_jensen_shannon function"""
    
    def test_empty_arrays(self):
        """Test JS divergence with empty arrays"""
        assert compute_jensen_shannon(np.array([]), np.array([])) == 0.0
        assert compute_jensen_shannon(np.array([[]]), np.array([[]])) == 0.0
    
    def test_identical_embeddings(self):
        """Test JS divergence with identical embeddings should be low"""
        embeddings = np.random.randn(100, 384)
        js = compute_jensen_shannon(embeddings, embeddings)
        assert js >= 0.0
        assert js < 0.1  # Very similar
    
    def test_different_embeddings(self):
        """Test JS divergence detects different embeddings"""
        baseline = np.random.randn(100, 384)
        current = np.random.randn(100, 384) * 2  # Different scale
        js = compute_jensen_shannon(current, baseline)
        assert js >= 0.0
        assert js <= 1.0  # JS is bounded [0, 1]
    
    def test_1d_embeddings(self):
        """Test JS divergence with 1D embeddings"""
        baseline = np.random.randn(100)
        current = np.random.randn(100)
        js = compute_jensen_shannon(current, baseline)
        assert 0.0 <= js <= 1.0
    
    def test_high_dimensional(self):
        """Test JS divergence with high-dimensional embeddings"""
        baseline = np.random.randn(50, 768)
        current = np.random.randn(50, 768)
        js = compute_jensen_shannon(current, baseline)
        assert 0.0 <= js <= 1.0
    
    def test_same_mean_different_variance(self):
        """Test JS divergence detects variance changes"""
        baseline = np.random.normal(0, 1, (100, 384))
        current = np.random.normal(0, 3, (100, 384))  # Higher variance
        js = compute_jensen_shannon(current, baseline)
        assert js > 0.0
    
    def test_custom_bins(self):
        """Test JS divergence with custom bins"""
        baseline = np.random.randn(100, 384)
        current = np.random.randn(100, 384)
        js_20 = compute_jensen_shannon(current, baseline, n_bins=20)
        js_50 = compute_jensen_shannon(current, baseline, n_bins=50)
        assert abs(js_20 - js_50) < 0.5  # Should be similar

class TestComputeWassersteinDistance:
    """Tests for compute_wasserstein_distance function"""
    
    def test_empty_arrays(self):
        """Test Wasserstein with empty arrays"""
        assert compute_wasserstein_distance(np.array([]), np.array([])) == 0.0
    
    def test_identical_embeddings(self):
        """Test Wasserstein with identical embeddings"""
        embeddings = np.random.randn(100, 384)
        dist = compute_wasserstein_distance(embeddings, embeddings)
        assert dist >= 0.0
        assert dist < 0.1  # Should be very small
    
    def test_different_embeddings(self):
        """Test Wasserstein detects different embeddings"""
        baseline = np.random.randn(100, 384)
        current = np.random.randn(100, 384) * 2
        dist = compute_wasserstein_distance(current, baseline)
        assert dist > 0.0
    
    def test_1d_embeddings(self):
        """Test Wasserstein with 1D embeddings"""
        baseline = np.random.randn(100)
        current = np.random.randn(100)
        dist = compute_wasserstein_distance(current, baseline)
        assert dist >= 0.0
    
    def test_high_dimensional(self):
        """Test Wasserstein with high-dimensional embeddings (uses PCA)"""
        baseline = np.random.randn(100, 500)
        current = np.random.randn(100, 500)
        dist = compute_wasserstein_distance(current, baseline)
        assert dist >= 0.0
    
    def test_small_dimensional(self):
        """Test Wasserstein with small dimensional embeddings"""
        baseline = np.random.randn(100, 10)
        current = np.random.randn(100, 10)
        dist = compute_wasserstein_distance(current, baseline)
        assert dist >= 0.0

class TestComputeChiSquare:
    """Tests for compute_chi_square function"""
    
    def test_empty_dicts(self):
        """Test Chi-Square with empty dictionaries"""
        chi2, p_value, details = compute_chi_square({}, {})
        assert chi2 == 0.0
        assert p_value == 1.0
        assert details == {}
    
    def test_identical_distributions(self):
        """Test Chi-Square with identical category distributions"""
        baseline = {"A": 100, "B": 200, "C": 300}
        current = {"A": 100, "B": 200, "C": 300}
        chi2, p_value, details = compute_chi_square(baseline, current)
        assert p_value > 0.05  # Should not reject null hypothesis
    
    def test_shifted_distribution(self):
        """Test Chi-Square detects category shift"""
        baseline = {"A": 100, "B": 200, "C": 300}
        current = {"A": 50, "B": 100, "C": 450}  # C increased
        chi2, p_value, details = compute_chi_square(baseline, current)
        assert chi2 > 0.0
        assert p_value < 0.05  # Should detect significant difference
    
    def test_sparse_categories(self):
        """Test Chi-Square with sparse categories (merged into other)"""
        baseline = {"A": 100, "B": 200, "C": 1, "D": 1}
        current = {"A": 100, "B": 200, "C": 1, "D": 1}
        chi2, p_value, details = compute_chi_square(baseline, current, min_expected_freq=5)
        # Should handle sparse categories gracefully
        assert isinstance(p_value, float)
    
    def test_new_categories(self):
        """Test Chi-Square with new categories in current"""
        baseline = {"A": 100, "B": 200}
        current = {"A": 100, "B": 200, "C": 100}  # New category
        chi2, p_value, details = compute_chi_square(baseline, current)
        assert chi2 > 0.0
    
    def test_category_details(self):
        """Test Chi-Square returns category details"""
        baseline = {"A": 100, "B": 200}
        current = {"A": 150, "B": 150}
        chi2, p_value, details = compute_chi_square(baseline, current)
        assert "A" in details
        assert "B" in details
        assert "baseline_count" in details["A"]
        assert "shift_percentage" in details["A"]
    
    def test_minimum_frequency(self):
        """Test Chi-Square minimum expected frequency parameter"""
        baseline = {"A": 10, "B": 10, "C": 2}
        current = {"A": 10, "B": 10, "C": 2}
        chi2, p_value, details = compute_chi_square(baseline, current, min_expected_freq=5)
        assert isinstance(p_value, float)

class TestComputePSIIncremental:
    """Tests for compute_psi_incremental function"""
    
    def test_empty_data(self):
        """Test incremental PSI with empty data"""
        baseline_hist = np.array([10, 20, 30])
        bin_edges = np.array([0, 1, 2, 3])
        assert compute_psi_incremental(baseline_hist, bin_edges, np.array([])) == 0.0
    
    def test_empty_histogram(self):
        """Test incremental PSI with empty histogram"""
        assert compute_psi_incremental(np.array([]), np.array([]), np.array([1, 2, 3])) == 0.0
    
    def test_identical_distributions(self):
        """Test incremental PSI matches full PSI for identical distributions"""
        baseline_data = np.random.normal(0, 1, 1000)
        current_data = baseline_data.copy()
        
        # Full PSI
        psi_full = calculate_psi(baseline_data, current_data)
        
        # Incremental PSI
        cache = create_histogram_cache(baseline_data, bins=10)
        baseline_hist = np.array(cache["counts"])
        bin_edges = np.array(cache["bin_edges"])
        psi_inc = compute_psi_incremental(baseline_hist, bin_edges, current_data)
        
        # Should be similar (within small tolerance due to binning)
        assert abs(psi_full - psi_inc) < 0.1
    
    def test_shifted_distribution(self):
        """Test incremental PSI detects shifted distribution"""
        baseline_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(2, 1, 1000)
        
        cache = create_histogram_cache(baseline_data, bins=10)
        baseline_hist = np.array(cache["counts"])
        bin_edges = np.array(cache["bin_edges"])
        psi_inc = compute_psi_incremental(baseline_hist, bin_edges, current_data)
        
        assert psi_inc > 0.0

class TestCreateHistogramCache:
    """Tests for create_histogram_cache function"""
    
    def test_empty_data(self):
        """Test histogram cache with empty data"""
        cache = create_histogram_cache(np.array([]))
        assert cache["sample_size"] == 0
        assert len(cache["counts"]) == 0
    
    def test_constant_data(self):
        """Test histogram cache with constant data"""
        data = np.ones(100) * 5.0
        cache = create_histogram_cache(data)
        assert cache["sample_size"] == 100
        assert len(cache["counts"]) == 1
        assert cache["counts"][0] == 100
    
    def test_normal_data(self):
        """Test histogram cache with normal distribution"""
        data = np.random.normal(0, 1, 1000)
        cache = create_histogram_cache(data, bins=10)
        assert cache["sample_size"] == 1000
        assert len(cache["counts"]) == 10
        assert len(cache["bin_edges"]) == 11  # n_bins + 1
        assert "mean" in cache
        assert "std" in cache
    
    def test_custom_bins(self):
        """Test histogram cache with custom bins"""
        data = np.random.normal(0, 1, 1000)
        cache_10 = create_histogram_cache(data, bins=10)
        cache_20 = create_histogram_cache(data, bins=20)
        assert len(cache_10["counts"]) == 10
        assert len(cache_20["counts"]) == 20

class TestStratifiedSubsample:
    """Tests for stratified_subsample function"""
    
    def test_small_data(self):
        """Test subsampling with data smaller than max_samples"""
        data = list(range(100))
        categories = ["A"] * 50 + ["B"] * 50
        sampled_data, sampled_categories = stratified_subsample(data, categories, max_samples=200)
        assert len(sampled_data) == 100
        assert sampled_data == data
        assert sampled_categories == categories
    
    def test_preserves_proportions(self):
        """Test subsampling preserves category proportions"""
        data = list(range(1000))
        categories = ["A"] * 600 + ["B"] * 300 + ["C"] * 100
        sampled_data, sampled_categories = stratified_subsample(data, categories, max_samples=500)
        
        # Check proportions are roughly maintained
        counts = {}
        for cat in sampled_categories:
            counts[cat] = counts.get(cat, 0) + 1
        
        assert len(sampled_data) <= 500
        assert abs(counts["A"] / len(sampled_data) - 0.6) < 0.1
        assert abs(counts["B"] / len(sampled_data) - 0.3) < 0.1
    
    def test_min_per_category(self):
        """Test subsampling respects min_per_category"""
        data = list(range(1000))
        categories = ["A"] * 950 + ["B"] * 50
        sampled_data, sampled_categories = stratified_subsample(
            data, categories, max_samples=100, min_per_category=30
        )
        
        counts = {cat: sampled_categories.count(cat) for cat in set(sampled_categories)}
        assert counts["B"] >= 30  # Should have at least min_per_category
    
    def test_all_categories_present(self):
        """Test subsampling includes all categories"""
        data = list(range(1000))
        categories = ["A"] * 400 + ["B"] * 400 + ["C"] * 200
        sampled_data, sampled_categories = stratified_subsample(data, categories, max_samples=200)
        
        unique_categories = set(sampled_categories)
        assert "A" in unique_categories
        assert "B" in unique_categories
        assert "C" in unique_categories

class TestComputeDistributionFeatures:
    """Tests for compute_distribution_features function"""
    
    def test_empty_data(self):
        """Test distribution features with empty data"""
        features = compute_distribution_features(np.array([]))
        assert features["mean"] == 0.0
        assert features["std"] == 0.0
        assert features["median"] == 0.0
    
    def test_normal_data(self):
        """Test distribution features with normal data"""
        data = np.array([1, 2, 3, 4, 5])
        features = compute_distribution_features(data)
        assert features["mean"] == 3.0
        assert features["median"] == 3.0
        assert features["min"] == 1.0
        assert features["max"] == 5.0
        assert features["q25"] == 2.0
        assert features["q75"] == 4.0
    
    def test_single_value(self):
        """Test distribution features with single value"""
        data = np.array([5.0])
        features = compute_distribution_features(data)
        assert features["mean"] == 5.0
        assert features["std"] == 0.0
    
    def test_all_features_present(self):
        """Test all expected features are present"""
        data = np.random.normal(0, 1, 1000)
        features = compute_distribution_features(data)
        required_keys = ["mean", "std", "median", "min", "max", "q25", "q75"]
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], float)
