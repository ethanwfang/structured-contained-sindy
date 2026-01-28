# Custom Dynamical Systems

Learn how to define your own dynamical systems.

## Subclassing DynamicalSystem

```python
from sc_sindy.systems import DynamicalSystem

class MySystem(DynamicalSystem):
    def __init__(self, param=1.0):
        self.param = param
        self.dim = 2
        self.name = "MySystem"

    def dynamics(self, t, state):
        x, y = state
        dxdt = self.param * y
        dydt = -x
        return [dxdt, dydt]

    def true_coefficients(self, library_labels):
        # Return ground truth coefficients
        pass
```

## Registering Your System

To add your system to the registry:

```python
from sc_sindy.systems.registry import register_system

@register_system
class MySystem(DynamicalSystem):
    ...
```

## Example: FitzHugh-Nagumo

```python
class FitzHughNagumo(DynamicalSystem):
    def __init__(self, a=0.7, b=0.8, tau=12.5, I=0.5):
        self.a = a
        self.b = b
        self.tau = tau
        self.I = I
        self.dim = 2
        self.name = "FitzHughNagumo"

    def dynamics(self, t, state):
        v, w = state
        dvdt = v - v**3/3 - w + self.I
        dwdt = (v + self.a - self.b * w) / self.tau
        return [dvdt, dwdt]
```
