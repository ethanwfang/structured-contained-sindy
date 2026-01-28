"""
Unit tests for dynamical systems.
"""

import pytest
import numpy as np
from sc_sindy.systems import (
    VanDerPol,
    LotkaVolterra,
    Lorenz,
    SelkovGlycolysis,
    CoupledBrusselator,
    get_system,
    list_systems,
    get_benchmark_systems,
)


class TestVanDerPol:
    """Unit tests for Van der Pol oscillator."""

    def test_correct_dimension(self):
        """Should be 2-dimensional."""
        system = VanDerPol()
        assert system.dim == 2

    def test_has_mu_parameter(self):
        """Should have mu parameter."""
        system = VanDerPol(mu=2.0)
        assert system.params['mu'] == 2.0

    def test_derivatives_shape(self):
        """Derivatives should have same shape as state."""
        system = VanDerPol()
        state = np.array([1.0, 0.5])
        deriv = system.derivatives(state, 0.0)
        assert len(deriv) == 2

    def test_trajectory_shape(self):
        """Generated trajectory should have correct shape."""
        system = VanDerPol()
        t = np.linspace(0, 10, 100)
        x0 = np.array([2.0, 0.0])
        x = system.generate_trajectory(x0, t)
        assert x.shape == (100, 2)

    def test_noise_adds_variability(self):
        """Noise should add variability to trajectory."""
        system = VanDerPol()
        t = np.linspace(0, 10, 100)
        x0 = np.array([2.0, 0.0])

        np.random.seed(42)
        x_clean = system.generate_trajectory(x0, t, noise_level=0.0)
        np.random.seed(42)
        x_noisy = system.generate_trajectory(x0, t, noise_level=0.1)

        # Should be different
        assert not np.allclose(x_clean, x_noisy)

    def test_true_structure(self):
        """Should return correct structure mask."""
        system = VanDerPol()
        term_names = ['1', 'x', 'y', 'xx', 'xy', 'yy', 'xxx', 'xxy', 'xyy', 'yyy']
        mask = system.get_true_structure(term_names)

        # dx/dt = y (only y active)
        assert mask[0, term_names.index('y')] == True
        assert mask[0, term_names.index('x')] == False

        # dy/dt = -x + mu*y - mu*x^2*y (x, y, xxy active)
        assert mask[1, term_names.index('x')] == True
        assert mask[1, term_names.index('y')] == True
        assert mask[1, term_names.index('xxy')] == True


class TestLotkaVolterra:
    """Unit tests for Lotka-Volterra system."""

    def test_correct_dimension(self):
        """Should be 2-dimensional."""
        system = LotkaVolterra()
        assert system.dim == 2

    def test_has_all_parameters(self):
        """Should have alpha, beta, delta, gamma."""
        system = LotkaVolterra(alpha=1.5, beta=0.7, delta=0.3, gamma=0.9)
        assert system.params['alpha'] == 1.5
        assert system.params['beta'] == 0.7
        assert system.params['delta'] == 0.3
        assert system.params['gamma'] == 0.9

    def test_true_structure(self):
        """Should return correct structure mask."""
        system = LotkaVolterra()
        term_names = ['1', 'x', 'y', 'xx', 'xy', 'yy']
        mask = system.get_true_structure(term_names)

        # dx/dt uses x, xy
        assert mask[0, term_names.index('x')] == True
        assert mask[0, term_names.index('xy')] == True
        assert mask[0, term_names.index('y')] == False

        # dy/dt uses y, xy
        assert mask[1, term_names.index('y')] == True
        assert mask[1, term_names.index('xy')] == True


class TestLorenz:
    """Unit tests for Lorenz system."""

    def test_correct_dimension(self):
        """Should be 3-dimensional."""
        system = Lorenz()
        assert system.dim == 3

    def test_default_parameters(self):
        """Should have standard Lorenz parameters."""
        system = Lorenz()
        assert system.params['sigma'] == 10.0
        assert system.params['rho'] == 28.0
        assert np.isclose(system.params['beta'], 8.0/3.0)

    def test_trajectory_shape(self):
        """Generated trajectory should have correct shape."""
        system = Lorenz()
        t = np.linspace(0, 10, 500)
        x0 = np.array([1.0, 0.0, 0.0])
        x = system.generate_trajectory(x0, t)
        assert x.shape == (500, 3)


class TestSystemRegistry:
    """Unit tests for system registry functions."""

    def test_get_system_by_name(self):
        """Should retrieve system by name."""
        system = get_system("vanderpol")
        assert isinstance(system, VanDerPol)

    def test_get_system_with_params(self):
        """Should pass parameters to system."""
        system = get_system("vanderpol", mu=3.0)
        assert system.params['mu'] == 3.0

    def test_get_system_case_insensitive(self):
        """Should handle different cases."""
        system1 = get_system("VanDerPol")
        system2 = get_system("VANDERPOL")
        assert type(system1) == type(system2)

    def test_unknown_system_raises(self):
        """Should raise error for unknown system."""
        with pytest.raises(ValueError):
            get_system("unknown_system")

    def test_list_systems_returns_list(self):
        """Should return list of strings."""
        systems = list_systems()
        assert isinstance(systems, list)
        assert all(isinstance(s, str) for s in systems)
        assert len(systems) > 0

    def test_list_systems_by_category(self):
        """Should filter by category."""
        systems_2d = list_systems("2d")
        assert "vanderpol" in systems_2d
        assert "lorenz" not in systems_2d

    def test_benchmark_systems(self):
        """Should return list of system instances."""
        systems = get_benchmark_systems()
        assert len(systems) > 0
        assert all(hasattr(s, 'generate_trajectory') for s in systems)
