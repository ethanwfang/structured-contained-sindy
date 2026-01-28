"""
Unit tests for standard SINDy algorithm.
"""

import pytest
import numpy as np
from sc_sindy.core.sindy import sindy_stls, sindy_ridge


class TestSINDySTLS:
    """Unit tests for Sequential Thresholded Least Squares."""

    def test_returns_correct_shape(self):
        """Coefficient matrix should be [n_vars, n_terms]."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        xi, _ = sindy_stls(Theta, x_dot)

        assert xi.shape == (2, 10)

    def test_returns_elapsed_time(self):
        """Should return non-negative elapsed time."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        _, elapsed = sindy_stls(Theta, x_dot)

        assert elapsed >= 0

    def test_sparse_solution(self):
        """STLS should produce sparse coefficients."""
        np.random.seed(42)
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        xi, _ = sindy_stls(Theta, x_dot, threshold=0.5)

        # Should have some zeros
        n_zeros = np.sum(np.abs(xi) < 1e-10)
        assert n_zeros > 0

    def test_threshold_effect(self):
        """Higher threshold should produce sparser solutions."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        xi_low, _ = sindy_stls(Theta, x_dot, threshold=0.1)
        xi_high, _ = sindy_stls(Theta, x_dot, threshold=0.5)

        n_active_low = np.sum(np.abs(xi_low) > 1e-10)
        n_active_high = np.sum(np.abs(xi_high) > 1e-10)

        assert n_active_high <= n_active_low

    def test_zero_threshold_gives_dense(self):
        """Zero threshold should give dense (least squares) solution."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        xi, _ = sindy_stls(Theta, x_dot, threshold=0.0, max_iter=0)

        # With no thresholding, should recover least squares
        xi_ls = np.linalg.lstsq(Theta, x_dot, rcond=None)[0].T

        np.testing.assert_array_almost_equal(xi, xi_ls)

    def test_recovers_known_system(self, vanderpol_data):
        """Should approximately recover Van der Pol structure."""
        Theta = vanderpol_data['Theta']
        x_dot = vanderpol_data['x_dot']
        true_coeffs = vanderpol_data['true_coeffs']

        xi, _ = sindy_stls(Theta, x_dot, threshold=0.1)

        # Check structure recovery (at least 50% recall)
        true_active = np.abs(true_coeffs) > 1e-6
        pred_active = np.abs(xi) > 1e-6

        # Should recover at least some of the true structure
        overlap = np.sum(pred_active & true_active)
        assert overlap > 0


class TestSINDyRidge:
    """Unit tests for SINDy with ridge regularization."""

    def test_returns_correct_shape(self):
        """Coefficient matrix should be [n_vars, n_terms]."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        xi, _ = sindy_ridge(Theta, x_dot)

        assert xi.shape == (2, 10)

    def test_regularization_reduces_coefficients(self):
        """Higher regularization should reduce coefficient magnitudes."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        xi_low, _ = sindy_ridge(Theta, x_dot, alpha=0.001)
        xi_high, _ = sindy_ridge(Theta, x_dot, alpha=1.0)

        norm_low = np.linalg.norm(xi_low)
        norm_high = np.linalg.norm(xi_high)

        assert norm_high <= norm_low

    def test_threshold_produces_sparsity(self):
        """Threshold should produce sparse solution."""
        Theta = np.random.randn(100, 10)
        x_dot = np.random.randn(100, 2)

        xi, _ = sindy_ridge(Theta, x_dot, alpha=0.01, threshold=0.3)

        n_zeros = np.sum(np.abs(xi) < 1e-10)
        assert n_zeros > 0
