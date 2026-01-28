"""
Pytest fixtures for Structure-Constrained SINDy tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sc_sindy.systems import VanDerPol, LotkaVolterra, Lorenz
from sc_sindy.core.library import build_library_2d, build_library_3d
from sc_sindy.derivatives import compute_derivatives_finite_diff


@pytest.fixture
def vanderpol_system():
    """Van der Pol oscillator fixture."""
    return VanDerPol(mu=1.5)


@pytest.fixture
def lotka_volterra_system():
    """Lotka-Volterra system fixture."""
    return LotkaVolterra(alpha=1.0, beta=0.5, delta=0.5, gamma=1.0)


@pytest.fixture
def lorenz_system():
    """Lorenz system fixture."""
    return Lorenz(sigma=10.0, rho=28.0, beta=8.0/3.0)


@pytest.fixture
def time_array():
    """Standard time array for tests."""
    return np.linspace(0, 50, 5000)


@pytest.fixture
def vanderpol_trajectory(vanderpol_system, time_array):
    """Generate Van der Pol trajectory."""
    x0 = np.array([2.0, 0.0])
    x = vanderpol_system.generate_trajectory(x0, time_array, noise_level=0.0)
    return x


@pytest.fixture
def vanderpol_noisy_trajectory(vanderpol_system, time_array):
    """Generate noisy Van der Pol trajectory."""
    x0 = np.array([2.0, 0.0])
    x = vanderpol_system.generate_trajectory(x0, time_array, noise_level=0.1)
    return x


@pytest.fixture
def lorenz_trajectory(lorenz_system, time_array):
    """Generate Lorenz trajectory."""
    x0 = np.array([1.0, 0.0, 0.0])
    x = lorenz_system.generate_trajectory(x0, time_array, noise_level=0.0)
    return x


@pytest.fixture
def sample_library_2d():
    """Sample 2D polynomial library."""
    x = np.random.randn(100, 2)
    Theta, names = build_library_2d(x, poly_order=3)
    return Theta, names


@pytest.fixture
def sample_library_3d():
    """Sample 3D polynomial library."""
    x = np.random.randn(100, 3)
    Theta, names = build_library_3d(x, poly_order=2)
    return Theta, names


@pytest.fixture
def vanderpol_data(vanderpol_system, time_array, vanderpol_trajectory):
    """Complete data for Van der Pol tests."""
    x = vanderpol_trajectory
    dt = time_array[1] - time_array[0]
    x_dot = compute_derivatives_finite_diff(x, dt)

    # Trim edges
    trim = 50
    x_trim = x[trim:-trim]
    x_dot_trim = x_dot[trim:-trim]

    Theta, term_names = build_library_2d(x_trim, poly_order=3)
    true_coeffs = vanderpol_system.get_true_coefficients(term_names)

    return {
        'x': x_trim,
        'x_dot': x_dot_trim,
        'Theta': Theta,
        'term_names': term_names,
        'true_coeffs': true_coeffs,
        'dt': dt,
    }


@pytest.fixture
def random_coefficients():
    """Random coefficient matrix for testing."""
    return np.random.randn(2, 10) * 0.5
