#!/usr/bin/env python3
"""
Run Benchmark - Compare Standard vs Structure-Constrained SINDy.

This script runs comprehensive benchmarks across multiple systems and noise levels.

Usage:
    python run_benchmark.py
    python run_benchmark.py --n-trials 10 --verbose
"""

import argparse
import numpy as np
import sys
from typing import Dict, List

sys.path.insert(0, '../../src')

from sc_sindy import (
    sindy_stls,
    sindy_structure_constrained,
    build_library_2d,
    get_2d_benchmark_systems,
)
from sc_sindy.derivatives import compute_derivatives_finite_diff
from sc_sindy.metrics import compute_structure_metrics, compute_coefficient_error
from sc_sindy.network import create_oracle_network_probs


def run_benchmark(
    systems: List,
    noise_levels: List[float],
    n_trials: int = 5,
    verbose: bool = False
) -> Dict:
    """
    Run benchmark across systems and noise levels.

    Returns
    -------
    results : Dict
        Nested dictionary: results[system_name][noise_level] = {metrics}
    """
    results = {}

    for system in systems:
        if verbose:
            print(f"\n{'='*60}")
            print(f"System: {system.name}")
            print(f"{'='*60}")

        system_results = {}

        for noise in noise_levels:
            if verbose:
                print(f"\nNoise level: {noise:.0%}")

            std_metrics = []
            sc_metrics = []

            for trial in range(n_trials):
                # Generate data
                t = np.linspace(0, 50, 5000)
                dt = t[1] - t[0]
                x0 = np.random.randn(2) * 2
                x = system.generate_trajectory(x0, t, noise_level=noise)

                # Compute derivatives and trim
                x_dot = compute_derivatives_finite_diff(x, dt)
                trim = 100
                x_trim = x[trim:-trim]
                x_dot_trim = x_dot[trim:-trim]

                # Build library
                Theta, term_names = build_library_2d(x_trim, poly_order=3)

                # Get true structure
                true_coeffs = system.get_true_coefficients(term_names)
                true_structure = np.abs(true_coeffs) > 1e-6
                network_probs = create_oracle_network_probs(true_structure)

                # Standard SINDy
                xi_std, t_std = sindy_stls(Theta, x_dot_trim, threshold=0.1)
                metrics_std = compute_structure_metrics(xi_std, true_coeffs)
                metrics_std['mae'] = compute_coefficient_error(xi_std, true_coeffs)
                metrics_std['time'] = t_std
                std_metrics.append(metrics_std)

                # Structure-Constrained SINDy
                xi_sc, t_sc = sindy_structure_constrained(
                    Theta, x_dot_trim, network_probs,
                    structure_threshold=0.3
                )
                metrics_sc = compute_structure_metrics(xi_sc, true_coeffs)
                metrics_sc['mae'] = compute_coefficient_error(xi_sc, true_coeffs)
                metrics_sc['time'] = t_sc
                sc_metrics.append(metrics_sc)

            # Average metrics
            def avg_metrics(metrics_list):
                keys = metrics_list[0].keys()
                return {k: np.mean([m[k] for m in metrics_list]) for k in keys}

            system_results[noise] = {
                'standard': avg_metrics(std_metrics),
                'structure_constrained': avg_metrics(sc_metrics),
            }

            if verbose:
                std = system_results[noise]['standard']
                sc = system_results[noise]['structure_constrained']
                print(f"  Standard:       F1={std['f1']:.3f}, MAE={std['mae']:.4f}, "
                      f"Time={std['time']*1000:.2f}ms")
                print(f"  Structure-Const: F1={sc['f1']:.3f}, MAE={sc['mae']:.4f}, "
                      f"Time={sc['time']*1000:.2f}ms")

                if std['mae'] > 0:
                    improvement = std['mae'] / sc['mae'] if sc['mae'] > 0 else float('inf')
                    print(f"  Improvement: {improvement:.1f}x MAE reduction")

        results[system.name] = system_results

    return results


def print_summary(results: Dict):
    """Print summary table of benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    for system_name, system_results in results.items():
        print(f"\n{system_name}")
        print("-" * 80)
        print(f"{'Noise':<10} | {'Standard F1':<12} | {'SC F1':<12} | "
              f"{'Std MAE':<12} | {'SC MAE':<12} | {'Improvement':<12}")
        print("-" * 80)

        for noise in sorted(system_results.keys()):
            std = system_results[noise]['standard']
            sc = system_results[noise]['structure_constrained']

            improvement = std['mae'] / sc['mae'] if sc['mae'] > 0 else float('inf')

            print(f"{noise:<10.0%} | {std['f1']:<12.3f} | {sc['f1']:<12.3f} | "
                  f"{std['mae']:<12.4f} | {sc['mae']:<12.4f} | {improvement:<12.1f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Run SINDy benchmark comparison"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of trials per configuration"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    print("Structure-Constrained SINDy Benchmark")
    print("=" * 60)

    # Get benchmark systems
    systems = get_2d_benchmark_systems()
    print(f"\nSystems: {[s.name for s in systems]}")

    noise_levels = [0.05, 0.10, 0.15, 0.20]
    print(f"Noise levels: {noise_levels}")
    print(f"Trials per configuration: {args.n_trials}")

    # Run benchmark
    results = run_benchmark(
        systems=systems,
        noise_levels=noise_levels,
        n_trials=args.n_trials,
        verbose=args.verbose
    )

    # Print summary
    print_summary(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
