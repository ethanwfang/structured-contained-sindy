# API Reference

## Core Algorithms

```{eval-rst}
.. automodule:: sc_sindy.core
   :members:
   :undoc-members:
```

### sindy_stls

Sequential Thresholded Least Squares algorithm.

```python
from sc_sindy import sindy_stls

xi, iterations = sindy_stls(Theta, X_dot, threshold=0.1)
```

### sindy_structure_constrained

Structure-Constrained SINDy with neural network priors.

```python
from sc_sindy import sindy_structure_constrained

xi, info = sindy_structure_constrained(
    Theta, X_dot, X,
    threshold=0.1,
    structure_threshold=0.3
)
```

## Dynamical Systems

```{eval-rst}
.. automodule:: sc_sindy.systems
   :members:
   :undoc-members:
```

### Available Systems

- **Oscillators**: `VanDerPol`, `DuffingOscillator`, `DampedHarmonicOscillator`
- **Biological**: `LotkaVolterra`, `SelkovGlycolysis`, `CoupledBrusselator`
- **Chaotic**: `Lorenz`, `Rossler`, `ChenSystem`

## Network Components

```{eval-rst}
.. automodule:: sc_sindy.network
   :members:
   :undoc-members:
```

## Metrics

```{eval-rst}
.. automodule:: sc_sindy.metrics
   :members:
   :undoc-members:
```

### Structure Metrics

- `compute_structure_metrics`: Precision, Recall, F1 for structure recovery

### Coefficient Metrics

- `compute_coefficient_error`: MAE, RMSE for coefficient accuracy

### Reconstruction Metrics

- `compute_reconstruction_error`: Trajectory reconstruction error
