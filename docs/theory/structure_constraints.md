# Structure Constraints

## Motivation

Standard SINDy can struggle with:
- High noise levels
- Limited data
- Complex dynamics

## Neural Network Priors

We train a neural network to predict which library terms are likely active:

$$p(\text{term}_i \text{ active} | \mathbf{X}) = \sigma(f_\theta(\mathbf{X}))$$

where $f_\theta$ is a neural network and $\sigma$ is the sigmoid function.

## Two-Stage Approach

### Stage 1: Structure Prediction

1. Extract features from trajectory $\mathbf{X}$
2. Pass through trained network
3. Threshold predictions to get candidate active terms

### Stage 2: STLS Refinement

1. Restrict library to candidate terms
2. Run standard STLS on reduced library
3. Further sparsification through thresholding

## Benefits

| Aspect | Standard SINDy | SC-SINDy |
|--------|---------------|----------|
| Noise tolerance | Low | High |
| Sample efficiency | Moderate | High |
| False positives | Common | Rare |
| Coefficient accuracy | Variable | Improved |

## Performance

SC-SINDy achieves 97-1568x improvement over standard SINDy on challenging systems with noise.

## Threshold Selection

The default structure threshold of 0.3 is robust across:
- Different dynamical systems
- Various noise levels
- Different data lengths

## References

- Original SINDy: Brunton et al. (2016)
- PySINDy: de Silva et al. (2020)
