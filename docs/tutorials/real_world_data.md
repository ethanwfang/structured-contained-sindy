# Working with Real-World Data

Apply SC-SINDy to real experimental data.

## Lynx-Hare Example

```python
import pandas as pd
from sc_sindy import sindy_structure_constrained

# Load data
data = pd.read_csv("data/raw/lynx_hare.csv")

# Preprocess and discover equations
# ...
```

## Tips for Real Data

### 1. Handle Missing Values

```python
# Interpolate gaps
data = data.interpolate(method='spline', order=3)
```

### 2. Apply Appropriate Smoothing

```python
from scipy.ndimage import gaussian_filter1d
X_smooth = gaussian_filter1d(X, sigma=2, axis=0)
```

### 3. Consider Noise Levels

Higher noise requires:
- Higher thresholds
- More data points
- Spline-based derivatives instead of finite differences

### 4. Normalize Your Data

```python
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
```

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| Non-uniform sampling | Interpolate to uniform grid |
| High noise | Use spline derivatives, higher threshold |
| Multiple timescales | Consider rescaling time |
| Missing variables | Use delay embedding |
