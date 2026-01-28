#!/usr/bin/env python3
"""
Discover Equations - CLI for equation discovery using Structure-Constrained SINDy.

This script provides a command-line interface for running SINDy on dynamical
systems data.

Usage:
    python discover_equations.py --system vanderpol --noise 0.1
    python discover_equations.py --system lorenz --method structure-constrained
    python discover_equations.py --data my_trajectory.csv
"""

import argparse
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, '../../src')

from sc_sindy import (
    sindy_stls,
    sindy_structure_constrained,
    build_library_2d,
    build_library_3d,
    get_system,
    list_systems,
)
from sc_sindy.derivatives import compute_derivatives_finite_diff
from sc_sindy.metrics import compute_structure_metrics, compute_coefficient_error
from sc_sindy.utils import print_equations, load_trajectory
from sc_sindy.network import create_oracle_network_probs


def main():
    parser = argparse.ArgumentParser(
        description="Discover governing equations from data using SINDy"
    )

    parser.add_argument(
        "--system",
        type=str,
        default="vanderpol",
        help=f"Dynamical system to use. Options: {', '.join(list_systems())}"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to trajectory data CSV file (alternative to --system)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["standard", "structure-constrained"],
        default="both",
        help="SINDy method to use"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Noise level (fraction of signal std)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=50.0,
        help="Simulation duration"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=5000,
        help="Number of data points"
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        default=3,
        help="Polynomial order for library"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="STLS threshold"
    )
    parser.add_argument(
        "--structure-threshold",
        type=float,
        default=0.3,
        help="Structure threshold for Structure-Constrained SINDy"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Load or generate data
    if args.data:
        print(f"Loading data from {args.data}")
        x, t, var_names = load_trajectory(args.data)
    else:
        print(f"Generating data from {args.system} system")
        system = get_system(args.system)
        t = np.linspace(0, args.duration, args.n_points)
        x0 = np.random.randn(system.dim) * 2
        x = system.generate_trajectory(x0, t, noise_level=args.noise)
        var_names = None

    dt = t[1] - t[0]
    n_vars = x.shape[1]

    print(f"\nData: {len(x)} points, {n_vars} variables")
    print(f"Time step: {dt:.6f}")
    if args.noise > 0:
        print(f"Noise level: {args.noise:.1%}")

    # Compute derivatives
    print("\nComputing derivatives...")
    x_dot = compute_derivatives_finite_diff(x, dt, order=2)

    # Trim edges to remove boundary artifacts
    trim = 50
    x_trim = x[trim:-trim]
    x_dot_trim = x_dot[trim:-trim]

    # Build library
    print(f"\nBuilding polynomial library (order {args.poly_order})...")
    if n_vars == 2:
        Theta, term_names = build_library_2d(x_trim, poly_order=args.poly_order)
    elif n_vars == 3:
        Theta, term_names = build_library_3d(x_trim, poly_order=min(args.poly_order, 2))
    else:
        print(f"Error: {n_vars}D systems not supported yet")
        return

    print(f"Library size: {len(term_names)} terms")
    if args.verbose:
        print(f"Terms: {term_names}")

    # Get true structure if using a known system
    true_coeffs = None
    if not args.data:
        system = get_system(args.system)
        true_coeffs = system.get_true_coefficients(term_names)
        if args.verbose:
            print("\nTrue equations:")
            print_equations(true_coeffs, term_names)

    # Run SINDy
    print("\n" + "=" * 60)

    if args.method in ["standard", "both"]:
        print("\n--- Standard SINDy ---")
        xi_std, t_std = sindy_stls(Theta, x_dot_trim, threshold=args.threshold)
        print(f"Time: {t_std*1000:.2f} ms")
        print(f"Active terms: {(np.abs(xi_std) > 1e-6).sum()}")
        print_equations(xi_std, term_names)

        if true_coeffs is not None:
            metrics = compute_structure_metrics(xi_std, true_coeffs)
            mae = compute_coefficient_error(xi_std, true_coeffs)
            print(f"\nMetrics: F1={metrics['f1']:.3f}, MAE={mae:.4f}")

    if args.method in ["structure-constrained", "both"]:
        print("\n--- Structure-Constrained SINDy ---")

        if true_coeffs is not None:
            # Use oracle network probabilities
            true_structure = np.abs(true_coeffs) > 1e-6
            network_probs = create_oracle_network_probs(true_structure)
        else:
            # Without known structure, use uniform priors
            print("Note: Using uniform priors (no trained network)")
            network_probs = np.ones((n_vars, len(term_names))) * 0.5

        xi_sc, t_sc = sindy_structure_constrained(
            Theta, x_dot_trim, network_probs,
            structure_threshold=args.structure_threshold,
            stls_threshold=args.threshold
        )
        print(f"Time: {t_sc*1000:.2f} ms")
        print(f"Active terms: {(np.abs(xi_sc) > 1e-6).sum()}")
        print_equations(xi_sc, term_names)

        if true_coeffs is not None:
            metrics = compute_structure_metrics(xi_sc, true_coeffs)
            mae = compute_coefficient_error(xi_sc, true_coeffs)
            print(f"\nMetrics: F1={metrics['f1']:.3f}, MAE={mae:.4f}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
