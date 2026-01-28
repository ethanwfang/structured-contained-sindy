# Getting Started

This tutorial introduces Structure-Constrained SINDy.

## Installation

```bash
pip install sc-sindy
```

## Basic Usage

```python
from sc_sindy import sindy_stls, build_library_2d, VanDerPol

# Create a dynamical system
system = VanDerPol(mu=1.0)

# Simulate trajectory
t, X = system.simulate([1.0, 0.0], t_span=(0, 10), dt=0.01)

# Discover equations
Theta, labels = build_library_2d(X)
# ... continue with SINDy
```

## Next Steps

- Learn about [custom dynamical systems](custom_systems.md)
- Explore [training structure networks](training_networks.md)
- Apply to [real-world data](real_world_data.md)
