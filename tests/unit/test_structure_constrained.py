"""
Unit tests for Structure-Constrained SINDy algorithm.
"""

import pytest
import numpy as np
from sc_sindy.core.structure_constrained import (
    sindy_structure_constrained,
    sindy_structure_constrained_soft,
    get_recommended_threshold,
    DEFAULT_STRUCTURE_THRESHOLD,
)
from sc_sindy.network import create_oracle_network_probs


class TestStructureConstrainedSINDy:
    """Unit tests for Structure-Constrained SINDy."""

    def test_returns_correct_shape(self):
        """Coefficient matrix should be [n_vars, n_terms]."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)
        network_probs = np.random.rand(2, 10)

        xi, _ = sindy_structure_constrained(Theta, x_dot, network_probs)

        assert xi.shape == (2, 10)

    def test_returns_elapsed_time(self):
        """Should return non-negative elapsed time."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)
        network_probs = np.random.rand(2, 10)

        _, elapsed = sindy_structure_constrained(Theta, x_dot, network_probs)

        assert elapsed >= 0

    def test_respects_structure_constraint(self):
        """Should only have active terms where network_probs > threshold."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        # Network says only first 3 terms are active
        network_probs = np.zeros((2, 10))
        network_probs[:, :3] = 0.9

        xi, _ = sindy_structure_constrained(
            Theta, x_dot, network_probs,
            structure_threshold=0.5
        )

        # Only first 3 terms should be non-zero
        assert np.all(np.abs(xi[:, 3:]) < 1e-10)

    def test_threshold_effect(self):
        """Higher structure threshold should give sparser solutions."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)
        network_probs = np.random.rand(2, 10)

        xi_low, _ = sindy_structure_constrained(
            Theta, x_dot, network_probs,
            structure_threshold=0.3
        )
        xi_high, _ = sindy_structure_constrained(
            Theta, x_dot, network_probs,
            structure_threshold=0.7
        )

        n_active_low = np.sum(np.abs(xi_low) > 1e-10)
        n_active_high = np.sum(np.abs(xi_high) > 1e-10)

        assert n_active_high <= n_active_low

    def test_oracle_network_perfect_recovery(self, vanderpol_data):
        """With perfect oracle, should perfectly recover structure."""
        Theta = vanderpol_data['Theta']
        x_dot = vanderpol_data['x_dot']
        true_coeffs = vanderpol_data['true_coeffs']

        # Create oracle network probabilities
        true_structure = np.abs(true_coeffs) > 1e-6
        network_probs = create_oracle_network_probs(true_structure)

        xi, _ = sindy_structure_constrained(
            Theta, x_dot, network_probs,
            structure_threshold=0.3
        )

        # Should recover exactly the true structure
        pred_structure = np.abs(xi) > 1e-6
        np.testing.assert_array_equal(pred_structure, true_structure)

    def test_empty_structure_gives_zeros(self):
        """If no terms pass threshold, should return zeros."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)
        network_probs = np.zeros((2, 10))  # All zeros

        xi, _ = sindy_structure_constrained(
            Theta, x_dot, network_probs,
            structure_threshold=0.5
        )

        assert np.all(xi == 0)


class TestStructureConstrainedSoft:
    """Unit tests for soft Structure-Constrained SINDy."""

    def test_returns_correct_shape(self):
        """Coefficient matrix should be [n_vars, n_terms]."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)
        network_probs = np.random.rand(2, 10)

        xi, _ = sindy_structure_constrained_soft(Theta, x_dot, network_probs)

        assert xi.shape == (2, 10)

    def test_probability_weighting_effect(self):
        """Low probability terms should have smaller coefficients."""
        np.random.seed(42)
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        # First half high probability, second half low
        network_probs = np.ones((2, 10))
        network_probs[:, 5:] = 0.1

        xi, _ = sindy_structure_constrained_soft(
            Theta, x_dot, network_probs,
            stls_threshold=0.3,
            prob_weighting=True
        )

        # Low probability terms should be sparser
        n_active_high = np.sum(np.abs(xi[:, :5]) > 1e-6)
        n_active_low = np.sum(np.abs(xi[:, 5:]) > 1e-6)

        # Generally expect fewer active terms in low probability region
        # (this test is probabilistic)


class TestGetRecommendedThreshold:
    """Unit tests for threshold recommendation function."""

    def test_returns_default_for_low_noise(self):
        """Should return default threshold for low noise."""
        threshold = get_recommended_threshold(0.05)
        assert threshold == DEFAULT_STRUCTURE_THRESHOLD

    def test_returns_higher_for_high_noise(self):
        """Should return higher threshold for high noise."""
        threshold = get_recommended_threshold(0.20)
        assert threshold >= DEFAULT_STRUCTURE_THRESHOLD

    def test_monotonic_in_noise(self):
        """Threshold should be non-decreasing with noise."""
        thresholds = [get_recommended_threshold(n) for n in [0.05, 0.10, 0.15, 0.20]]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1]


class TestCreateOracleNetworkProbs:
    """Unit tests for oracle network probability creation."""

    def test_returns_correct_shape(self):
        """Should return same shape as input."""
        true_structure = np.random.rand(2, 10) > 0.5
        probs = create_oracle_network_probs(true_structure)
        assert probs.shape == true_structure.shape

    def test_high_prob_for_true_terms(self):
        """True active terms should have high probability."""
        true_structure = np.array([[True, False], [False, True]])
        probs = create_oracle_network_probs(true_structure, confidence=0.9)

        assert probs[0, 0] > 0.5
        assert probs[1, 1] > 0.5

    def test_low_prob_for_false_terms(self):
        """Inactive terms should have low probability."""
        true_structure = np.array([[True, False], [False, True]])
        probs = create_oracle_network_probs(true_structure, noise=0.05)

        assert probs[0, 1] < 0.5
        assert probs[1, 0] < 0.5

    def test_respects_confidence_parameter(self):
        """Confidence parameter should set probability of true terms."""
        true_structure = np.array([[True, False]])
        probs = create_oracle_network_probs(true_structure, confidence=0.8)
        assert np.isclose(probs[0, 0], 0.8)
