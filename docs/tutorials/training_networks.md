# Training Structure Networks

Learn how to train neural networks for structure prediction.

## Overview

The structure network predicts which library terms are active based on trajectory features.

## Training

```python
from sc_sindy.network import train_structure_network

# Train the network
model = train_structure_network(
    systems=["VanDerPol", "LotkaVolterra", "Lorenz"],
    epochs=100,
    batch_size=32
)

# Save the model
import torch
torch.save(model.state_dict(), "models/structure_network.pt")
```

## Command Line Training

```bash
python examples/scripts/train_network.py --dim 2 --epochs 100
```

## Network Architecture

The `StructureNetwork` uses:
- Input: Trajectory features (spectral, statistical)
- Hidden layers: Configurable MLP
- Output: Probability for each library term

## Feature Extraction

Features extracted from trajectories:
- FFT magnitudes (spectral content)
- Statistical moments (mean, std, skew, kurtosis)
- Cross-correlation features
- Derivative statistics
