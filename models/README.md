# Pretrained Models

This directory contains pretrained Structure Network models.

## Available Models

| Model | Description | Input Dim | Output Dim |
|-------|-------------|-----------|------------|
| `structure_network_2d.pt` | Network for 2D systems | 19 | 20 |
| `structure_network_3d.pt` | Network for 3D systems | 27 | 30 |

## Usage

```python
from sc_sindy.network import StructureNetwork

# Load pretrained model
model = StructureNetwork.load('models/structure_network_2d.pt')

# Make predictions
features = extract_trajectory_features(x, dt)
probs = model.predict(features)
```

## Training Your Own

See `examples/scripts/train_network.py` for training custom models.
