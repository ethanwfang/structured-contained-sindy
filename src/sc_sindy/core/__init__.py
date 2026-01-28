"""
Core SINDy algorithms.

This module provides the fundamental algorithms for Sparse Identification
of Nonlinear Dynamics (SINDy), including standard STLS and the
Structure-Constrained variant.
"""

from .sindy import (
    sindy_stls,
    sindy_ridge,
    DEFAULT_STLS_THRESHOLD,
)

from .structure_constrained import (
    sindy_structure_constrained,
    sindy_structure_constrained_soft,
    get_recommended_threshold,
    DEFAULT_STRUCTURE_THRESHOLD,
)

from .library import (
    build_library_2d,
    build_library_3d,
    build_library_nd,
)

__all__ = [
    # Standard SINDy
    "sindy_stls",
    "sindy_ridge",
    "DEFAULT_STLS_THRESHOLD",
    # Structure-Constrained SINDy
    "sindy_structure_constrained",
    "sindy_structure_constrained_soft",
    "get_recommended_threshold",
    "DEFAULT_STRUCTURE_THRESHOLD",
    # Library construction
    "build_library_2d",
    "build_library_3d",
    "build_library_nd",
]
