"""Train structure prediction network."""

import argparse
from pathlib import Path

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required for training. Install with: pip install torch")

from sc_sindy.network import StructureNetwork, train_structure_network
from sc_sindy.systems import get_all_systems


def main():
    parser = argparse.ArgumentParser(description="Train structure prediction network")
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=[2, 3],
        help="System dimension (2 or 3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for model",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of training samples per system",
    )
    args = parser.parse_args()

    # Get systems for the specified dimension
    systems = get_all_systems(dim=args.dim)
    print(f"Training on {len(systems)} {args.dim}D systems")

    # Train network
    model = train_structure_network(
        systems=systems,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_samples_per_system=args.n_samples,
    )

    # Save model
    output_path = args.output or f"models/structure_network_{args.dim}d.pt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
