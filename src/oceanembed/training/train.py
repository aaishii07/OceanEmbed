from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.oceanembed.data.dataset import OceanProfileDataset
from src.oceanembed.evaluation.metrics import profile_metrics
from src.oceanembed.evaluation.plots import save_evaluation_plots
from src.oceanembed.models import OceanEmbedCNN


def predict(model: nn.Module, loader: DataLoader, y_mean: np.ndarray, y_std: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); actual, predicted = [], []
    with torch.no_grad():
        for x, y in loader:
            actual.append(y.numpy() * y_std + y_mean)
            predicted.append(model(x.to(device)).cpu().numpy() * y_std + y_mean)
    return np.vstack(actual), np.vstack(predicted)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the OceanEmbed CNN with a chronological validation split.")
    p.add_argument("--data", required=True); p.add_argument("--channels", type=int, nargs="+", default=[0, 1])
    p.add_argument("--run-name", default="sst_ssh"); p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=42); args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    archive = np.load(args.data, allow_pickle=False); n = len(archive["X"]); split = max(1, int(n * 0.8))
    train_ds = OceanProfileDataset(args.data, np.arange(split), args.channels)
    val_ds = OceanProfileDataset(args.data, np.arange(split, n), args.channels, train_ds.stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OceanEmbedCNN(len(args.channels), archive["y"].shape[1]).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True); val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); loss_fn = nn.MSELoss()
    out = Path("results") / args.run_name; out.mkdir(parents=True, exist_ok=True); best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in train_loader:
            optimizer.zero_grad(); loss = loss_fn(model(x.to(device)), y.to(device)); loss.backward(); optimizer.step(); losses.append(loss.item())
        obs, pred = predict(model, val_loader, train_ds.stats["y_mean"], train_ds.stats["y_std"], device)
        rmse = float(np.sqrt(((obs - pred) ** 2).mean())); print(f"epoch {epoch:03d} train_mse={np.mean(losses):.4f} val_rmse_c={rmse:.3f}")
        if rmse < best:
            best = rmse
            torch.save({"model_state": model.state_dict(), "channels": args.channels, "n_depths": archive["y"].shape[1], "stats": train_ds.stats, "depths_m": archive["depths_m"]}, out / "best_model.pt")
    state = torch.load(out / "best_model.pt", map_location=device, weights_only=False); model.load_state_dict(state["model_state"])
    obs, pred = predict(model, val_loader, train_ds.stats["y_mean"], train_ds.stats["y_std"], device)
    metrics = profile_metrics(obs, pred, archive["depths_m"]); metrics.update({"device": str(device), "train_samples": split, "validation_samples": n-split, "channels": args.channels})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)); np.savez_compressed(out / "predictions.npz", observed=obs, predicted=pred, depths_m=archive["depths_m"])
    save_evaluation_plots(obs, pred, archive["depths_m"], out); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
