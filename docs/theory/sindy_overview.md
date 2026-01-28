# SINDy Overview

## Sparse Identification of Nonlinear Dynamics

SINDy discovers governing equations from time-series data.

### Mathematical Formulation

Given trajectory data $\mathbf{X}$ and derivatives $\dot{\mathbf{X}}$:

$$\dot{\mathbf{X}} = \Theta(\mathbf{X}) \Xi$$

where $\Theta$ is a library of candidate functions and $\Xi$ contains sparse coefficients.

### Library Construction

For a 2D system with variables $(x, y)$, a polynomial library might include:

$$\Theta = [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, ...]$$

### Sequential Thresholded Least Squares (STLS)

1. Solve least squares: $\Xi = (\Theta^T \Theta)^{-1} \Theta^T \dot{\mathbf{X}}$
2. Threshold small coefficients: $\Xi_{ij} = 0$ if $|\Xi_{ij}| < \lambda$
3. Repeat until convergence

### Algorithm

```
function STLS(Θ, Ẋ, λ):
    Ξ = lstsq(Θ, Ẋ)
    while not converged:
        small_idx = |Ξ| < λ
        Ξ[small_idx] = 0
        for each column i:
            big_idx = |Ξ[:, i]| >= λ
            Ξ[big_idx, i] = lstsq(Θ[:, big_idx], Ẋ[:, i])
    return Ξ
```

### References

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. PNAS.
