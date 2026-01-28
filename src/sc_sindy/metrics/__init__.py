"""
Evaluation metrics for SINDy.

This module provides comprehensive metrics for evaluating SINDy algorithms,
including structure recovery, coefficient accuracy, and trajectory reconstruction.
"""

from .structure import (
    compute_structure_metrics,
    compute_per_equation_metrics,
    structure_similarity,
    sparsity_ratio,
    compare_structures,
)

from .coefficient import (
    compute_coefficient_error,
    compute_coefficient_rmse,
    compute_relative_coefficient_error,
    compute_active_coefficient_error,
    coefficient_correlation,
    compute_coefficient_metrics,
    per_equation_coefficient_error,
)

from .reconstruction import (
    compute_reconstruction_error,
    compute_derivative_reconstruction_error,
    compute_normalized_error,
    compute_r2_score,
    integrate_discovered_model,
    compute_forward_prediction_error,
    compute_lyapunov_time_error,
    compute_reconstruction_metrics,
)

__all__ = [
    # Structure metrics
    "compute_structure_metrics",
    "compute_per_equation_metrics",
    "structure_similarity",
    "sparsity_ratio",
    "compare_structures",
    # Coefficient metrics
    "compute_coefficient_error",
    "compute_coefficient_rmse",
    "compute_relative_coefficient_error",
    "compute_active_coefficient_error",
    "coefficient_correlation",
    "compute_coefficient_metrics",
    "per_equation_coefficient_error",
    # Reconstruction metrics
    "compute_reconstruction_error",
    "compute_derivative_reconstruction_error",
    "compute_normalized_error",
    "compute_r2_score",
    "integrate_discovered_model",
    "compute_forward_prediction_error",
    "compute_lyapunov_time_error",
    "compute_reconstruction_metrics",
]
