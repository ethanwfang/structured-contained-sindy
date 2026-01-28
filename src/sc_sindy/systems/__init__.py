"""
Dynamical systems library.

This module provides a collection of dynamical systems for testing
and benchmarking SINDy algorithms.
"""

from .base import DynamicalSystem

from .oscillators import (
    VanDerPol,
    DuffingOscillator,
    DampedHarmonicOscillator,
    ForcedOscillator,
)

from .biological import (
    LotkaVolterra,
    SelkovGlycolysis,
    CoupledBrusselator,
    SIRModel,
)

from .chaotic import (
    Lorenz,
    Rossler,
    ChenSystem,
    DoublePendulum,
)

from .registry import (
    get_system,
    list_systems,
    get_benchmark_systems,
    get_2d_benchmark_systems,
    system_info,
    SYSTEM_REGISTRY,
    SYSTEM_CATEGORIES,
)

__all__ = [
    # Base class
    "DynamicalSystem",
    # Oscillators
    "VanDerPol",
    "DuffingOscillator",
    "DampedHarmonicOscillator",
    "ForcedOscillator",
    # Biological
    "LotkaVolterra",
    "SelkovGlycolysis",
    "CoupledBrusselator",
    "SIRModel",
    # Chaotic
    "Lorenz",
    "Rossler",
    "ChenSystem",
    "DoublePendulum",
    # Registry
    "get_system",
    "list_systems",
    "get_benchmark_systems",
    "get_2d_benchmark_systems",
    "system_info",
    "SYSTEM_REGISTRY",
    "SYSTEM_CATEGORIES",
]
