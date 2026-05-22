import os
import sys
import argparse
import time
import json
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.models import create_model
from src.losses import combined_loss
from src.dataset import create_dataloaders
from src.features import N_FEATURES
from utils import weighted_pearson_correlation


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(requested)
    return device


def validate(model, valid_loader, device) -> dict:
    """Per-target WPC + average, computed with the official scoring function."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets, mask in valid_loader:
            features = features.to(device)
            preds, _ = model(features)
            preds = preds.cpu().numpy()
            targets_np = targets.numpy()
            mask_np = mask.numpy().astype(bool)

            for b in range(preds.shape[0]):
                m = mask_np[b]
                all_preds.append(preds[b][m])
                all_targets.append(targets_np[b][m])

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    t0_wpc = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    t1_wpc = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    avg_wpc = (t0_wpc + t1_wpc) / 2

    return {"t0_wpc": t0_wpc, "t1_wpc": t1_wpc, "avg_wpc": avg_wpc}


def train_one_epoch(
    model, train_loader, optimizer, device,
    mse_weight: float, pearson_weight: float,
    grad_clip: float = 1.0, feature_dropout: float = 0.05,
    epoch: int = 0, total_epochs: int = 1,
    mse_mode: str = "plain", noise_std: float = 0.0,
    target_weights: tuple = (1.0, 1.0),
    feature_scale: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{total_epochs}",
        leave=False,
        ncols=120,
    )

    for features, targets, mask in pbar:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # cheap augmentation: per-sequence feature dropout + per-feature random scale + gaussian noise
        if model.training:
            if feature_dropout > 0:
                drop_mask = (torch.rand(features.shape[0], 1, features.shape[2], device=device) > feature_dropout).float()
                features = features * drop_mask
            if feature_scale > 0:
                scale = 1.0 + (torch.rand(features.shape[0], 1, features.shape[2], device=device) * 2 - 1) * feature_scale
                features = features * scale
            if noise_std > 0:
                features = features + torch.randn_like(features) * noise_std

        optimizer.zero_grad(set_to_none=True)

        preds, _ = model(features)

        loss = combined_loss(preds, targets, mask,
                             mse_weight=mse_weight, pearson_weight=pearson_weight,
                             mse_mode=mse_mode, target_weights=target_weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{total_loss / n_batches:.5f}"})

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train one LOB prediction model")
    parser.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mixer-layers", type=int, default=3)
    parser.add_argument("--head-hidden", type=int, default=64)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--feature-dropout", type=float, default=0.05)
    parser.add_argument("--phase2-start", type=float, default=0.3,
                        help="Fraction of training before switching to Pearson loss.")
    parser.add_argument("--mse-mode", type=str, default="plain",
                        choices=["plain", "soft", "huber"])
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stopping patience. 0 to disable.")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--feature-scale", type=float, default=0.1)
    parser.add_argument("--t1-weight", type=float, default=1.0,
                        help="Relative loss weight for t1. Higher = more focus on t1.")

    parser.add_argument("--train-path", type=str,
                        default=os.path.join(PROJECT_DIR, "datasets", "train.parquet"))
    parser.add_argument("--valid-path", type=str,
                        default=os.path.join(PROJECT_DIR, "datasets", "valid.parquet"))
    parser.add_argument("--fold", type=int, default=None,
                        help="K-fold CV index. If None, use the train/valid files directly.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42,
                        help="Seed for k-fold splitting. Hold constant across training seeds.")
    parser.add_argument("--use-all-data", action="store_true",
                        help="Train on every sequence with no held-out set. For final submissions.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_DIR, "checkpoints"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)

    run_name = f"{args.arch}_h{args.hidden_size}_l{args.num_layers}_s{args.seed}"
    if args.fold is not None:
        run_name += f"_fold{args.fold}"

    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  Arch: {args.arch.upper()} | Hidden: {args.hidden_size} | Layers: {args.num_layers}")
    print(f"  Batch: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"  Loss: {args.mse_mode} MSE | Phase 2 at epoch {int(args.epochs * args.phase2_start) + 1}")
    print(f"  Augmentation: dropout={args.feature_dropout}, scale={args.feature_scale}, noise={args.noise}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    print("\n--- Loading Data ---")
    train_loader, valid_loader = create_dataloaders(
        train_path=args.train_path,
        valid_path=args.valid_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=args.fold,
        n_folds=args.n_folds,
        seed=args.seed,
        fold_seed=args.fold_seed,
        use_all_data=args.use_all_data,
        use_features=True,
    )

    print("\n--- Creating Model ---")
    model = create_model(
        arch=args.arch,
        input_dim=N_FEATURES,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mixer_layers=args.mixer_layers,
        head_hidden=args.head_hidden,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch.upper()}, params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # T_max = full epoch budget. If training runs all the way through, the schedule
    # is monotonically decaying; if early stopping fires, we just used the head of it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    print("\n--- Training ---")
    print(f"{'Epoch':>6} {'Phase':>12} {'Loss':>10} {'t0_WPC':>8} {'t1_WPC':>8} "
          f"{'Avg_WPC':>8} {'LR':>10} {'Time':>7}")
    print("-" * 80)

    best_wpc = -float("inf")
    best_epoch = -1
    checkpoint_path = ""
    phase2_epoch = int(args.epochs * args.phase2_start)
    history = []
    patience_counter = 0

    for epoch in range(args.epochs):
        t_start = time.time()

        # Phase 1: pure MSE. Phase 2: ramp Pearson term in over 10 epochs, while
        # MSE weight drops 1.0 -> 0.5 so the loss magnitude stays sensible.
        if epoch < phase2_epoch:
            mse_weight, pearson_weight = 1.0, 0.0
            phase = "MSE"
        else:
            progress = min((epoch - phase2_epoch) / 10, 1.0)
            pearson_weight = 0.5 * progress
            mse_weight = 1.0 - pearson_weight
            phase = "MSE+Pearson"

        target_weights = (1.0, args.t1_weight)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            mse_weight=mse_weight, pearson_weight=pearson_weight,
            grad_clip=args.grad_clip, feature_dropout=args.feature_dropout,
            epoch=epoch, total_epochs=args.epochs,
            mse_mode=args.mse_mode, noise_std=args.noise,
            target_weights=target_weights,
            feature_scale=args.feature_scale,
        )
        scheduler.step()

        elapsed = time.time() - t_start
        lr = optimizer.param_groups[0]["lr"]

        if valid_loader is not None:
            metrics = validate(model, valid_loader, device)
            wpc = metrics["avg_wpc"]

            if wpc > best_wpc:
                best_wpc = wpc
                best_epoch = epoch
                patience_counter = 0
                checkpoint_path = os.path.join(args.output_dir, f"{run_name}_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "wpc": wpc,
                    "args": vars(args),
                }, checkpoint_path)
            else:
                patience_counter += 1

            marker = " *" if epoch == best_epoch else ""
            print(
                f"{epoch+1:>5d}/{args.epochs} {phase:>12s} {train_loss:>10.6f} "
                f"{metrics['t0_wpc']:>8.4f} {metrics['t1_wpc']:>8.4f} "
                f"{wpc:>8.4f}{marker:2s} {lr:>10.2e} {elapsed:>6.1f}s"
            )
            history.append({
                "epoch": epoch, "loss": train_loss,
                "t0_wpc": metrics["t0_wpc"], "t1_wpc": metrics["t1_wpc"],
                "avg_wpc": wpc, "lr": lr, "phase": phase,
            })

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break
        else:
            print(
                f"{epoch+1:>5d}/{args.epochs} {phase:>12s} {train_loss:>10.6f} "
                f"{'N/A':>8s} {'N/A':>8s} {'N/A':>8s}   {lr:>10.2e} {elapsed:>6.1f}s"
            )
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                cp = os.path.join(args.output_dir, f"{run_name}_ep{epoch+1}.pt")
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "args": vars(args)}, cp)

    final_path = os.path.join(args.output_dir, f"{run_name}_final.pt")
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, final_path)

    history_path = os.path.join(args.output_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    if valid_loader is not None:
        print(f"  Best WPC: {best_wpc:.4f} at epoch {best_epoch+1}")
        print(f"  Best checkpoint: {checkpoint_path}")
    print(f"  Final model: {final_path}")
    print(f"  History: {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
