"""Statistical functions for drift detection"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, chi2_contingency
from typing import List, Tuple, Dict, Optional
from collections import Counter

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI measures how much a distribution has shifted over time.
    Threshold interpretation:
    - PSI < 0.15: Normal - No significant change detected
    - PSI 0.15-0.25: Warning - Moderate change, monitor closely
    - PSI 0.25-0.40: Critical - Significant change, action may be needed
    - PSI > 0.40: Emergency - Major shift, immediate action required
    
    Note: Thresholds are configurable via PSI_WARNING_THRESHOLD, PSI_CRITICAL_THRESHOLD, PSI_EMERGENCY_THRESHOLD
    
    Args:
        expected: Baseline/reference distribution
        actual: Current distribution to compare
        bins: Number of bins for histogram
        
    Returns:
        PSI score (float)
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Determine bin edges from expected distribution
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    if max_val == min_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate histograms
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Normalize to probabilities (add small epsilon to avoid division by zero)
    epsilon = 1e-10
    expected_probs = (expected_counts + epsilon) / np.sum(expected_counts + epsilon)
    actual_probs = (actual_counts + epsilon) / np.sum(actual_counts + epsilon)
    
    # Calculate PSI
    psi = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
    
    return float(psi)


def compute_ks_test(current_dist: np.ndarray, baseline_dist: np.ndarray) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test comparing current distribution to baseline.
    
    The KS test checks if two samples come from the same distribution.
    Lower p-value indicates distributions are more different.
    
    Threshold interpretation (p-value):
    - p-value > 0.05: Normal - Distributions appear similar
    - p-value 0.01-0.05: Warning - Moderate difference detected
    - p-value 0.001-0.01: Critical - Significant difference
    - p-value < 0.001: Emergency - Very different distributions
    
    Note: Thresholds are configurable via KS_WARNING_THRESHOLD, KS_CRITICAL_THRESHOLD, KS_EMERGENCY_THRESHOLD
    
    Args:
        current_dist: Current distribution
        baseline_dist: Baseline/reference distribution
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    if len(current_dist) == 0 or len(baseline_dist) == 0:
        return (0.0, 1.0)
    
    statistic, p_value = stats.ks_2samp(baseline_dist, current_dist)
    
    return (float(statistic), float(p_value))


def compute_jensen_shannon(embeddings_current: np.ndarray, embeddings_baseline: np.ndarray, 
                           n_bins: int = 20) -> float:
    """
    Calculate Jensen-Shannon divergence between two embedding distributions.
    
    JS divergence is a symmetric measure of similarity between probability distributions.
    Returns value between 0 (identical) and 1 (completely different).
    
    Threshold interpretation:
    - JS < 0.1: Normal - Similar distributions
    - JS 0.1-0.2: Warning - Moderate difference detected
    - JS 0.2-0.3: Critical - Significant difference
    - JS > 0.3: Emergency - Very different distributions
    
    Note: Thresholds are configurable via JS_WARNING_THRESHOLD, JS_CRITICAL_THRESHOLD, JS_EMERGENCY_THRESHOLD
    
    Args:
        embeddings_current: Current embedding vectors (n_samples, n_dimensions)
        embeddings_baseline: Baseline embedding vectors (n_samples, n_dimensions)
        n_bins: Number of bins for probability estimation
        
    Returns:
        Jensen-Shannon divergence (0 to 1)
    """
    if len(embeddings_current) == 0 or len(embeddings_baseline) == 0:
        return 0.0
    
    # Reduce dimensionality if needed (use first few dimensions for simplicity)
    # In production, you might use PCA or other dimensionality reduction
    max_dims = min(10, embeddings_current.shape[1] if embeddings_current.ndim > 1 else 1)
    
    if embeddings_current.ndim == 1:
        # 1D case - use directly
        embeddings_current_1d = embeddings_current
        embeddings_baseline_1d = embeddings_baseline
    else:
        # Multi-dimensional - use first dimension or mean across dimensions
        embeddings_current_1d = embeddings_current[:, 0] if embeddings_current.shape[1] > 0 else embeddings_current.mean(axis=1)
        embeddings_baseline_1d = embeddings_baseline[:, 0] if embeddings_baseline.shape[1] > 0 else embeddings_baseline.mean(axis=1)
    
    # Create probability distributions using histograms
    min_val = min(np.min(embeddings_current_1d), np.min(embeddings_baseline_1d))
    max_val = max(np.max(embeddings_current_1d), np.max(embeddings_baseline_1d))
    
    if max_val == min_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Calculate histograms
    current_counts, _ = np.histogram(embeddings_current_1d, bins=bin_edges)
    baseline_counts, _ = np.histogram(embeddings_baseline_1d, bins=bin_edges)
    
    # Normalize histogram counts to probabilities (must sum to ~1)
    epsilon = 1e-10
    current_probs = (current_counts + epsilon) / (np.sum(current_counts) + epsilon * len(current_counts))
    baseline_probs = (baseline_counts + epsilon) / (np.sum(baseline_counts) + epsilon * len(baseline_counts))
    
    # Calculate JS divergence
    js_divergence = jensenshannon(current_probs, baseline_probs)
    
    return float(js_divergence)


def compute_wasserstein_distance(embeddings_current: np.ndarray, embeddings_baseline: np.ndarray) -> float:
    """
    Calculate Wasserstein (Earth Mover's) Distance between embedding distributions.

    More accurate than JS divergence for high-dimensional embeddings as it preserves
    geometric structure and uses all dimensions with PCA dimensionality reduction.

    Wasserstein distance measures the minimum cost to transform one distribution
    into another, making it more sensitive to subtle semantic shifts.

    Interpretation:
    - Wasserstein < 0.5: Similar distributions
    - Wasserstein 0.5-1.5: Moderate difference
    - Wasserstein > 1.5: Significant difference

    Args:
        embeddings_current: Current embedding vectors (n_samples, n_dimensions)
        embeddings_baseline: Baseline embedding vectors (n_samples, n_dimensions)

    Returns:
        Average Wasserstein distance across principal components
    """
    if len(embeddings_current) == 0 or len(embeddings_baseline) == 0:
        return 0.0

    # Ensure 2D arrays
    if embeddings_current.ndim == 1:
        embeddings_current = embeddings_current.reshape(-1, 1)
    if embeddings_baseline.ndim == 1:
        embeddings_baseline = embeddings_baseline.reshape(-1, 1)

    # For high-dimensional embeddings, use PCA to reduce dimensions while preserving variance
    n_dims = embeddings_current.shape[1]

    if n_dims > 50:
        # Use PCA for dimensionality reduction
        try:
            from sklearn.decomposition import PCA

            # Reduce to components explaining 95% variance, max 50 components
            n_components = min(50, n_dims, len(embeddings_current), len(embeddings_baseline))
            pca = PCA(n_components=n_components)

            # Fit on combined data for consistent transformation
            combined = np.vstack([embeddings_baseline, embeddings_current])
            pca.fit(combined)

            current_reduced = pca.transform(embeddings_current)
            baseline_reduced = pca.transform(embeddings_baseline)
        except ImportError:
            # Fallback: use first 50 dimensions if sklearn not available
            current_reduced = embeddings_current[:, :50]
            baseline_reduced = embeddings_baseline[:, :50]
    else:
        current_reduced = embeddings_current
        baseline_reduced = embeddings_baseline

    # Calculate Wasserstein distance for each dimension
    distances = []
    for i in range(current_reduced.shape[1]):
        try:
            dist = wasserstein_distance(current_reduced[:, i], baseline_reduced[:, i])
            distances.append(dist)
        except Exception:
            # Skip dimensions that cause errors
            continue

    if not distances:
        return 0.0

    # Return mean Wasserstein distance across all dimensions
    return float(np.mean(distances))


def compute_distribution_features(data: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical features of a distribution for comparison.

    Args:
        data: Input data array

    Returns:
        Dictionary of statistical features
    """
    if len(data) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0
        }

    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75))
    }


def compute_chi_square(baseline_categories: Dict[str, int], 
                       current_categories: Dict[str, int],
                       min_expected_freq: int = 5) -> Tuple[float, float, Dict]:
    """
    Chi-Square test for categorical drift detection.
    
    Tests whether the distribution of categories has changed significantly
    between baseline and current periods.
    
    Threshold interpretation (p-value):
    - p-value > 0.05: Normal - Category distribution appears similar
    - p-value 0.01-0.05: Warning - Moderate category shift detected
    - p-value 0.001-0.01: Critical - Significant category shift
    - p-value < 0.001: Emergency - Very different category distribution
    
    Args:
        baseline_categories: Dict mapping category names to counts in baseline
        current_categories: Dict mapping category names to counts in current window
        min_expected_freq: Minimum expected frequency per category (categories below
                          this are merged into "other")
    
    Returns:
        Tuple of (chi2_statistic, p_value, category_details)
    """
    if not baseline_categories or not current_categories:
        return (0.0, 1.0, {})
    
    # Get all categories
    all_categories = set(baseline_categories.keys()) | set(current_categories.keys())
    
    # Handle sparse categories by merging those with low counts
    baseline_total = sum(baseline_categories.values())
    current_total = sum(current_categories.values())
    
    if baseline_total == 0 or current_total == 0:
        return (0.0, 1.0, {})
    
    # Build contingency table
    # Merge categories with expected frequency < min_expected_freq into "other"
    merged_baseline = {}
    merged_current = {}
    other_baseline = 0
    other_current = 0
    
    for cat in all_categories:
        baseline_count = baseline_categories.get(cat, 0)
        current_count = current_categories.get(cat, 0)
        
        # Calculate expected frequency under null hypothesis
        expected_baseline = (baseline_count + current_count) * baseline_total / (baseline_total + current_total)
        expected_current = (baseline_count + current_count) * current_total / (baseline_total + current_total)
        
        if expected_baseline < min_expected_freq or expected_current < min_expected_freq:
            other_baseline += baseline_count
            other_current += current_count
        else:
            merged_baseline[cat] = baseline_count
            merged_current[cat] = current_count
    
    # Add "other" category if any categories were merged
    if other_baseline > 0 or other_current > 0:
        merged_baseline["_other"] = other_baseline
        merged_current["_other"] = other_current
    
    # Need at least 2 categories for chi-square
    if len(merged_baseline) < 2:
        return (0.0, 1.0, {})
    
    # Build contingency table
    categories = list(merged_baseline.keys())
    observed = np.array([
        [merged_baseline.get(cat, 0) for cat in categories],
        [merged_current.get(cat, 0) for cat in categories]
    ])
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        # Calculate per-category contribution to chi-square
        category_details = {}
        for i, cat in enumerate(categories):
            if cat == "_other":
                continue
            baseline_pct = (merged_baseline.get(cat, 0) / baseline_total * 100) if baseline_total > 0 else 0
            current_pct = (merged_current.get(cat, 0) / current_total * 100) if current_total > 0 else 0
            shift = current_pct - baseline_pct
            
            # Chi-square contribution for this category
            obs_baseline = merged_baseline.get(cat, 0)
            obs_current = merged_current.get(cat, 0)
            exp_baseline = expected[0][i]
            exp_current = expected[1][i]
            
            contribution = 0
            if exp_baseline > 0:
                contribution += (obs_baseline - exp_baseline) ** 2 / exp_baseline
            if exp_current > 0:
                contribution += (obs_current - exp_current) ** 2 / exp_current
            
            category_details[cat] = {
                "baseline_count": merged_baseline.get(cat, 0),
                "current_count": merged_current.get(cat, 0),
                "baseline_percentage": round(baseline_pct, 2),
                "current_percentage": round(current_pct, 2),
                "shift_percentage": round(shift, 2),
                "chi2_contribution": round(contribution, 4),
            }
        
        return (float(chi2), float(p_value), category_details)
    
    except Exception:
        return (0.0, 1.0, {})


def compute_psi_incremental(baseline_histogram: np.ndarray, 
                            baseline_bin_edges: np.ndarray,
                            actual: np.ndarray) -> float:
    """
    Calculate PSI using pre-computed baseline histogram (incremental approach).
    
    Much faster than recomputing baseline histogram every time.
    
    Args:
        baseline_histogram: Pre-computed histogram counts for baseline
        baseline_bin_edges: Bin edges used for baseline histogram
        actual: Current distribution data points
        
    Returns:
        PSI score (float)
    """
    if len(actual) == 0 or len(baseline_histogram) == 0:
        return 0.0
    
    # Calculate actual histogram using same bins
    actual_counts, _ = np.histogram(actual, bins=baseline_bin_edges)
    
    # Normalize to probabilities
    epsilon = 1e-10
    baseline_probs = (baseline_histogram + epsilon) / np.sum(baseline_histogram + epsilon)
    actual_probs = (actual_counts + epsilon) / np.sum(actual_counts + epsilon)
    
    # Calculate PSI
    psi = np.sum((actual_probs - baseline_probs) * np.log(actual_probs / baseline_probs))
    
    return float(psi)


def create_histogram_cache(data: np.ndarray, bins: int = 10) -> Dict:
    """
    Create a cacheable histogram representation for incremental PSI computation.
    
    Args:
        data: Array of values to create histogram from
        bins: Number of bins
        
    Returns:
        Dictionary with histogram counts and bin edges
    """
    if len(data) == 0:
        return {"counts": [], "bin_edges": [], "sample_size": 0}
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val == min_val:
        return {"counts": [len(data)], "bin_edges": [min_val, max_val + 1], "sample_size": len(data)}
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    counts, _ = np.histogram(data, bins=bin_edges)
    
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "sample_size": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(min_val),
        "max": float(max_val),
    }


def stratified_subsample(data: List, categories: List[str], 
                         max_samples: int = 10000,
                         min_per_category: int = 50) -> Tuple[List, List]:
    """
    Stratified subsampling to maintain category proportions.
    
    For large datasets, intelligently sample while preserving distribution.
    
    Args:
        data: List of data items
        categories: List of category labels (same length as data)
        max_samples: Maximum total samples to return
        min_per_category: Minimum samples per category
        
    Returns:
        Tuple of (sampled_data, sampled_categories)
    """
    if len(data) <= max_samples:
        return data, categories
    
    # Count per category
    category_counts = Counter(categories)
    total_count = len(data)
    
    # Calculate target samples per category (proportional)
    sample_targets = {}
    for cat, count in category_counts.items():
        proportion = count / total_count
        target = max(min_per_category, int(proportion * max_samples))
        sample_targets[cat] = min(target, count)  # Can't sample more than available
    
    # Adjust if we're over max_samples
    total_target = sum(sample_targets.values())
    if total_target > max_samples:
        scale = max_samples / total_target
        sample_targets = {cat: max(min_per_category, int(target * scale)) 
                         for cat, target in sample_targets.items()}
    
    # Group data by category
    category_indices = {}
    for i, cat in enumerate(categories):
        if cat not in category_indices:
            category_indices[cat] = []
        category_indices[cat].append(i)
    
    # Sample from each category
    sampled_indices = []
    for cat, indices in category_indices.items():
        target = sample_targets.get(cat, min_per_category)
        if len(indices) <= target:
            sampled_indices.extend(indices)
        else:
            sampled_indices.extend(np.random.choice(indices, size=target, replace=False).tolist())
    
    # Return sampled data
    sampled_data = [data[i] for i in sampled_indices]
    sampled_categories = [categories[i] for i in sampled_indices]
    
    return sampled_data, sampled_categories
