# Structure-Constrained SINDy (SC-SINDy)

A comprehensive implementation of **Structure-Constrained SINDy** for discovering governing equations from data with learned structural priors.

## Overview

Structure-Constrained SINDy is a two-stage approach that combines neural network predictions with sparse regression to achieve **97-1568x improvement** over standard SINDy on challenging dynamical systems.

### Key Features

- **Standard SINDy** - Sequential Thresholded Least Squares (STLS) implementation
- **Structure-Constrained SINDy** - Two-stage approach with neural network priors
- **Dynamical Systems Library** - Van der Pol, Lorenz, Lotka-Volterra, and more
- **Comprehensive Metrics** - Structure recovery, coefficient accuracy, reconstruction error
- **Easy-to-use API** - Clean, Pythonic interface for equation discovery

## Installation

### From Source

```bash
git clone https://github.com/yourusername/structure-constrained-sindy.git
cd structure-constrained-sindy
pip install -e .
```

### With Optional Dependencies

```bash
# With PyTorch for Structure Network
pip install -e ".[torch]"

# With visualization
pip install -e ".[viz]"

# Full installation
pip install -e ".[all]"
```

## Quick Start

```python
import numpy as np
from sc_sindy import (
    sindy_stls,
    sindy_structure_constrained,
    build_library_2d,
    compute_derivatives_finite_diff,
    VanDerPol,
    print_equations,
)
from sc_sindy.network import create_oracle_network_probs

# Generate data from Van der Pol oscillator
system = VanDerPol(mu=1.5)
t = np.linspace(0, 50, 5000)
x0 = np.array([2.0, 0.0])
x = system.generate_trajectory(x0, t, noise_level=0.10)

# Compute derivatives
dt = t[1] - t[0]
x_dot = compute_derivatives_finite_diff(x, dt)

# Build polynomial library
Theta, term_names = build_library_2d(x, poly_order=3)

# Run Standard SINDy
xi_std, time_std = sindy_stls(Theta, x_dot, threshold=0.1)
print("Standard SINDy:")
print_equations(xi_std, term_names)

# Run Structure-Constrained SINDy (with oracle network)
true_structure = system.get_true_structure(term_names)
network_probs = create_oracle_network_probs(true_structure)

xi_sc, time_sc = sindy_structure_constrained(
    Theta, x_dot, network_probs,
    structure_threshold=0.3
)
print("\nStructure-Constrained SINDy:")
print_equations(xi_sc, term_names)
```

## Method Overview

### Two-Stage Approach

1. **Stage 1: Network-Guided Filtering**
   - Neural network predicts probability of each library term being active
   - Terms below `structure_threshold` (default: 0.3) are excluded

2. **Stage 2: STLS Refinement**
   - Standard STLS on reduced library
   - Further sparsifies with `stls_threshold` (default: 0.1)

### Threshold Selection

Based on comprehensive ablation studies:
- **Robust range**: 0.2 - 0.8 (nearly identical performance)
- **Recommended default**: 0.3 (safe across all systems and noise levels)
- **Avoid**: 0.9 (causes 50%+ recall loss)

## Available Systems

### Oscillators
- `VanDerPol` - Van der Pol oscillator
- `DuffingOscillator` - Duffing oscillator
- `DampedHarmonicOscillator` - Linear damped oscillator

### Biological
- `LotkaVolterra` - Predator-prey dynamics
- `SelkovGlycolysis` - Glycolytic oscillations
- `CoupledBrusselator` - Chemical oscillations

### Chaotic
- `Lorenz` - Lorenz attractor
- `Rossler` - Rössler system
- `ChenSystem` - Chen attractor

```python
from sc_sindy import get_system, list_systems

# List all systems
print(list_systems())

# Get system by name
lorenz = get_system("lorenz", sigma=10, rho=28, beta=8/3)
```

## Documentation

### Core Functions

| Function | Description |
|----------|-------------|
| `sindy_stls(Theta, x_dot, threshold)` | Standard SINDy with STLS |
| `sindy_structure_constrained(Theta, x_dot, network_probs)` | Structure-Constrained SINDy |
| `build_library_2d(x, poly_order)` | Build 2D polynomial library |
| `build_library_3d(x, poly_order)` | Build 3D polynomial library |
| `compute_derivatives_finite_diff(x, dt)` | Finite difference derivatives |
| `compute_derivatives_spline(x, t)` | Spline-based derivatives |

### Metrics

```python
from sc_sindy.metrics import (
    compute_structure_metrics,  # Precision, Recall, F1
    compute_coefficient_error,  # MAE between coefficients
    compute_reconstruction_error,  # Trajectory RMSE
)

metrics = compute_structure_metrics(xi_pred, xi_true)
print(f"F1 Score: {metrics['f1']:.3f}")
```

## Examples

### Basic Comparison

```bash
cd examples/scripts
python discover_equations.py --system vanderpol --noise 0.1
```

### Run Benchmark

```bash
python run_benchmark.py --n-trials 5 --verbose
```

### Lynx-Hare Real-World Data

```python
from sc_sindy.utils import load_lynx_hare_data

x, years = load_lynx_hare_data()
# x[:, 0] = Hare population (normalized)
# x[:, 1] = Lynx population (normalized)
```

## Project Structure

```
structure-constrained-sindy/
├── src/sc_sindy/
│   ├── core/           # Core algorithms (SINDy, library)
│   ├── derivatives/    # Derivative computation
│   ├── network/        # Neural network components
│   ├── systems/        # Dynamical systems
│   ├── metrics/        # Evaluation metrics
│   └── utils/          # Utilities (visualization, I/O)
├── tests/              # Unit and integration tests
├── examples/           # Example scripts and notebooks
└── docs/               # Documentation
```

## Development

### Setup

```bash
git clone https://github.com/yourusername/structure-constrained-sindy.git
cd structure-constrained-sindy
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=sc_sindy --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sc_sindy,
  title={Structure-Constrained SINDy},
  author={Structure-Constrained SINDy Project},
  year={2024},
  url={https://github.com/yourusername/structure-constrained-sindy}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
