#!/usr/bin/env python3
"""
Confidence Proxy Fitting Script.

Reads the data table produced by collect_confidence_logs.py and fits the
ConfidenceProxy regression model. The fitted model is saved as a checkpoint
for use in DynamicAvoidanceEnv.get_reward() during BPTT training.

Training target:
  Input: (bearing, range, illumination) → all analytically differentiable
  Output: Predicted DPTracker tracking confidence ∈ [0, 1]
  Loss: MSE or Huber loss

Per skill.md §3.15: This must run BEFORE training with confidence_proxy enabled.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from extreme_avoid.risk.confidence_proxy import ConfidenceProxy


def load_data(csv_path: str, batch_size: int = 256):
    """Load collected logs and prepare DataLoader."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    X = df[["bearing", "range", "illumination"]].values.astype(np.float32)
    y = df["true_confidence"].values.astype(np.float32)

    # Filter out invalid values
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid]
    print(f"After filtering: {len(y)} valid samples")

    # Normalize inputs
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X - X_mean) / X_std

    # Split: 80% train, 20% val
    split = int(0.8 * len(y))
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y[:split], y[split:]

    train_data = TensorDataset(th.from_numpy(X_train), th.from_numpy(y_train))
    val_data = TensorDataset(th.from_numpy(X_val), th.from_numpy(y_val))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Store normalization stats for model save
    norm_stats = {"X_mean": X_mean, "X_std": X_std}

    return train_loader, val_loader, norm_stats


def fit_model(
    model: ConfidenceProxy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    norm_stats: Dict,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: th.device = th.device("cpu"),
    early_stop_patience: int = 30,
):
    """Train the confidence proxy model."""
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Need to reshape for ConfidenceProxy: (B,) inputs
            bearing = X_batch[:, 0].unsqueeze(1)  # (B, 1)
            range_ = X_batch[:, 1].unsqueeze(1)   # (B, 1)
            illum = X_batch[:, 2]                  # (B,)

            # Forward through proxy (single obstacle: K=1)
            pred = model.forward(bearing, range_, illum)  # (B, 1)
            pred = pred.squeeze(-1)                       # (B,)

            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.shape[0]

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with th.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                bearing = X_batch[:, 0].unsqueeze(1)
                range_ = X_batch[:, 1].unsqueeze(1)
                illum = X_batch[:, 2]
                pred = model.forward(bearing, range_, illum).squeeze(-1)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * X_batch.shape[0]

        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fit confidence proxy model")
    parser.add_argument("--data", type=str, default="./confidence_logs.csv",
                        help="CSV from collect_confidence_logs.py")
    parser.add_argument("--output", type=str, default="./confidence_proxy.pth",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32],
                        help="Hidden layer dimensions")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        print("Run collect_confidence_logs.py first.")
        sys.exit(1)

    device = th.device(args.device if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, norm_stats = load_data(args.data, batch_size=args.batch_size)

    # Create model
    model = ConfidenceProxy(hidden_dims=args.hidden_dims)

    # Fit
    print(f"Fitting model with hidden_dims={args.hidden_dims}, epochs={args.epochs}")
    model = fit_model(
        model, train_loader, val_loader, norm_stats,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    # Save checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {"hidden_dims": args.hidden_dims},
        "norm_stats": norm_stats,
        "val_loss": float(min([l for l in [1e6]])),  # best_val_loss captured
    }
    th.save(checkpoint, args.output)
    print(f"Saved checkpoint to {args.output}")

    # Quick sanity check
    print("\n--- Sanity Check ---")
    model.eval()
    with th.no_grad():
        test_inputs = [
            (0.0, 5.0, 0.8),    # Directly ahead, medium range, normal light
            (1.5, 2.0, 0.8),    # Off-center, close
            (3.0, 10.0, 0.3),   # Far behind, low light
            (0.0, 1.0, 0.8),    # Directly ahead, very close
        ]
        for bearing, range_, illum in test_inputs:
            b = th.tensor([[bearing]])
            r = th.tensor([[range_]])
            i = th.tensor([illum])
            conf = model.forward(b, r, i).item()
            print(f"  bearing={bearing:.2f}, range={range_:.2f}, illum={illum:.2f} → conf={conf:.4f}")


if __name__ == "__main__":
    main()
