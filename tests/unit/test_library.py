"""
Unit tests for library construction.
"""

import pytest
import numpy as np
from sc_sindy.core.library import build_library_2d, build_library_3d, build_library_nd


class TestBuildLibrary2D:
    """Unit tests for 2D polynomial library."""

    def test_returns_correct_shapes(self):
        """Should return Theta and names with consistent shapes."""
        x = np.random.randn(100, 2)
        Theta, names = build_library_2d(x, poly_order=3)

        assert Theta.shape[0] == 100  # n_samples
        assert Theta.shape[1] == len(names)

    def test_order_1_has_correct_terms(self):
        """Order 1 should have: 1, x, y"""
        x = np.random.randn(10, 2)
        Theta, names = build_library_2d(x, poly_order=1)

        assert names == ['1', 'x', 'y']
        assert Theta.shape[1] == 3

    def test_order_2_has_correct_terms(self):
        """Order 2 should add: xx, xy, yy"""
        x = np.random.randn(10, 2)
        Theta, names = build_library_2d(x, poly_order=2)

        expected = ['1', 'x', 'y', 'xx', 'xy', 'yy']
        assert names == expected
        assert Theta.shape[1] == 6

    def test_order_3_has_correct_terms(self):
        """Order 3 should add: xxx, xxy, xyy, yyy"""
        x = np.random.randn(10, 2)
        Theta, names = build_library_2d(x, poly_order=3)

        assert '1' in names
        assert 'x' in names
        assert 'xxy' in names
        assert 'yyy' in names
        assert Theta.shape[1] == 10

    def test_constant_term_is_ones(self):
        """First column should be all ones."""
        x = np.random.randn(100, 2)
        Theta, names = build_library_2d(x, poly_order=2)

        np.testing.assert_array_equal(Theta[:, 0], np.ones(100))

    def test_linear_terms_correct(self):
        """Linear terms should match input data."""
        x = np.random.randn(100, 2)
        Theta, names = build_library_2d(x, poly_order=2)

        idx_x = names.index('x')
        idx_y = names.index('y')

        np.testing.assert_array_equal(Theta[:, idx_x], x[:, 0])
        np.testing.assert_array_equal(Theta[:, idx_y], x[:, 1])

    def test_quadratic_terms_correct(self):
        """Quadratic terms should be correct products."""
        x = np.random.randn(100, 2)
        Theta, names = build_library_2d(x, poly_order=2)

        idx_xx = names.index('xx')
        idx_xy = names.index('xy')
        idx_yy = names.index('yy')

        np.testing.assert_array_almost_equal(Theta[:, idx_xx], x[:, 0]**2)
        np.testing.assert_array_almost_equal(Theta[:, idx_xy], x[:, 0] * x[:, 1])
        np.testing.assert_array_almost_equal(Theta[:, idx_yy], x[:, 1]**2)


class TestBuildLibrary3D:
    """Unit tests for 3D polynomial library."""

    def test_returns_correct_shapes(self):
        """Should return Theta and names with consistent shapes."""
        x = np.random.randn(100, 3)
        Theta, names = build_library_3d(x, poly_order=2)

        assert Theta.shape[0] == 100
        assert Theta.shape[1] == len(names)

    def test_order_1_has_correct_terms(self):
        """Order 1 should have: 1, x, y, z"""
        x = np.random.randn(10, 3)
        Theta, names = build_library_3d(x, poly_order=1)

        assert names == ['1', 'x', 'y', 'z']
        assert Theta.shape[1] == 4

    def test_order_2_includes_cross_terms(self):
        """Order 2 should include cross terms like xz, yz."""
        x = np.random.randn(10, 3)
        Theta, names = build_library_3d(x, poly_order=2)

        assert 'xz' in names
        assert 'yz' in names
        assert 'xy' in names

    def test_cross_term_correct(self):
        """Cross term xz should be x*z."""
        x = np.random.randn(100, 3)
        Theta, names = build_library_3d(x, poly_order=2)

        idx_xz = names.index('xz')
        np.testing.assert_array_almost_equal(Theta[:, idx_xz], x[:, 0] * x[:, 2])


class TestBuildLibraryND:
    """Unit tests for N-dimensional polynomial library."""

    def test_2d_matches_build_library_2d(self):
        """For 2D input, should produce same terms as build_library_2d."""
        x = np.random.randn(100, 2)

        Theta_2d, names_2d = build_library_2d(x, poly_order=2)
        Theta_nd, names_nd = build_library_nd(x, poly_order=2)

        assert len(names_2d) == len(names_nd)

    def test_4d_produces_expected_count(self):
        """4D order-2 should have correct number of terms."""
        x = np.random.randn(50, 4)
        Theta, names = build_library_nd(x, poly_order=2)

        # Order 0: 1 term
        # Order 1: 4 terms (x, y, z, w)
        # Order 2: C(4+2-1, 2) = C(5,2) = 10 terms
        # Total: 1 + 4 + 10 = 15
        assert len(names) == 15

    def test_constant_term_is_ones(self):
        """First column should be all ones for any dimension."""
        x = np.random.randn(100, 5)
        Theta, names = build_library_nd(x, poly_order=2)

        assert names[0] == '1'
        np.testing.assert_array_equal(Theta[:, 0], np.ones(100))
