"""Out-of-fold validation.

For each fold checkpoint, only score sequences that were in *that fold's* validation
set. That gives an honest ensemble estimate with no leakage between train and eval.

Discovers checkpoints automatically by parsing the filename:
    {arch}_h{hidden}_l{layers}_s{seed}_fold{fold}_best.pt

Examples:
    python src/oof_validate.py --checkpoint-dir checkpoints/ --n-folds 5
    python src/oof_validate.py --checkpoint-dir checkpoints/ --seeds 42 123 --optimize-weights
"""
import os
import sys
import re
import argparse
import json
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.features import OnlineFeatureEngine, N_FEATURES
from src.models import create_model
from src.dataset import load_sequences, split_sequences_kfold
from utils import weighted_pearson_correlation


def discover_checkpoints(checkpoint_dir: str):
    """Find every fold checkpoint and parse its config out of the filename."""
    pattern = re.compile(
        r"^(gru|lstm)_h(\d+)_l(\d+)_s(\d+)_fold(\d+)_best\.pt$"
    )
    checkpoints = []
    for fname in sorted(os.listdir(checkpoint_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        arch, hidden, layers, seed, fold = m.groups()
        model_id = fname.replace("_best.pt", "")
        checkpoints.append({
            "path": os.path.join(checkpoint_dir, fname),
            "arch": arch,
            "hidden_size": int(hidden),
            "num_layers": int(layers),
            "seed": int(seed),
            "fold": int(fold),
            "model_id": model_id,
        })
    return checkpoints


def optimize_weights(
    per_model_preds: dict,
    all_oof_targets: dict,
    model_ids: list,
) -> np.ndarray:
    """Optimize ensemble weights at the config-group level (grouping all folds
    of a given seed+arch+size together), not per-model. That keeps the search
    in 3-5 dims instead of 15-50 and avoids overfitting the weights.

    Returns a (n_models,) weight array that sums to 1.
    """
    from scipy.optimize import minimize

    n_models = len(model_ids)

    config_groups = {}  # prefix -> list of model indices
    for i, mid in enumerate(model_ids):
        prefix = "_".join(mid.split("_")[:-1])  # strip foldN
        config_groups.setdefault(prefix, []).append(i)

    group_names = sorted(config_groups.keys())
    n_groups = len(group_names)
    print(f"  Config groups ({n_groups}): {group_names}")

    gixs = sorted(per_model_preds.keys())
    all_preds_per_model = {mid: [] for mid in model_ids}
    all_targets_list = []

    for gix in gixs:
        targets = all_oof_targets[gix]
        all_targets_list.append(targets)
        for mid in model_ids:
            if mid in per_model_preds[gix]:
                all_preds_per_model[mid].append(per_model_preds[gix][mid])
            else:
                # this model wasn't trained on this fold's validation set
                all_preds_per_model[mid].append(np.zeros_like(targets))

    all_targets = np.concatenate(all_targets_list, axis=0)
    model_preds = np.stack(
        [np.concatenate(all_preds_per_model[mid], axis=0) for mid in model_ids],
        axis=0,
    )

    # Map group weights -> per-model weights (equal weight within each group)
    group_to_model = np.zeros((n_groups, n_models))
    for gi, gname in enumerate(group_names):
        for mi in config_groups[gname]:
            group_to_model[gi, mi] = 1.0
        group_to_model[gi] /= group_to_model[gi].sum()

    def neg_avg_wpc_group(log_group_weights):
        gw = np.exp(log_group_weights)
        gw = gw / gw.sum()
        per_model_w = gw @ group_to_model
        ensemble = np.einsum("m,mst->st", per_model_w, model_preds)
        t0 = weighted_pearson_correlation(all_targets[:, 0], ensemble[:, 0])
        t1 = weighted_pearson_correlation(all_targets[:, 1], ensemble[:, 1])
        return -(t0 + t1) / 2

    x0 = np.zeros(n_groups)
    maxiter = 500
    print(f"  Nelder-Mead: {n_groups} group params, maxiter={maxiter}, "
          f"{all_targets.shape[0]:,} steps")
    result = minimize(neg_avg_wpc_group, x0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-7})

    opt_group_weights = np.exp(result.x)
    opt_group_weights = opt_group_weights / opt_group_weights.sum()
    opt_weights = opt_group_weights @ group_to_model

    print(f"  Optimization: {result.message}")
    print(f"  Equal-weight WPC: {-neg_avg_wpc_group(np.zeros(n_groups)):.6f}")
    print(f"  Optimized WPC:    {-result.fun:.6f}")
    print(f"\n  Group weights:")
    for gname, gw in zip(group_names, opt_group_weights):
        print(f"    {gname}: {gw:.4f}")
    print(f"\n  Per-model weights:")
    for mid, w in zip(model_ids, opt_weights):
        print(f"    {mid}: {w:.4f}")

    return opt_weights


def run_oof_validation(
    checkpoint_dir: str,
    train_path: str,
    valid_path: str,
    n_folds: int = 5,
    fold_seed: int = 42,
    seeds: list = None,
    archs: list = None,
    hidden_sizes: list = None,
    model_ids_filter: list = None,
    do_optimize: bool = False,
    output_dir: str = None,
):
    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None

    if seeds:
        checkpoints = [c for c in checkpoints if c["seed"] in seeds]
    if archs:
        checkpoints = [c for c in checkpoints if c["arch"] in archs]
    if hidden_sizes:
        checkpoints = [c for c in checkpoints if c["hidden_size"] in hidden_sizes]
    if model_ids_filter:
        checkpoints = [c for c in checkpoints
                       if any(c["model_id"].startswith(prefix) for prefix in model_ids_filter)]

    print(f"Found {len(checkpoints)} checkpoints:")
    for c in checkpoints:
        print(f"  {c['model_id']}")

    print("\nLoading sequences...")
    train_seqs = load_sequences(train_path)
    valid_seqs = load_sequences(valid_path)
    all_seqs = train_seqs + valid_seqs
    # seq_ix isn't unique across the train and valid files, so reassign IDs here
    for i, seq in enumerate(all_seqs):
        seq["global_ix"] = i
    print(f"Total: {len(all_seqs)} sequences")

    engine = OnlineFeatureEngine()

    all_oof_preds = {}     # global_ix -> list of preds (one per model)
    per_model_preds = {}   # global_ix -> {model_id: preds}
    all_oof_targets = {}   # global_ix -> targets
    model_ids_seen = set()

    for ckpt in checkpoints:
        fold = ckpt["fold"]
        model_id = ckpt["model_id"]
        model_ids_seen.add(model_id)

        _, valid_split = split_sequences_kfold(
            all_seqs, n_folds=n_folds, fold=fold, fold_seed=fold_seed
        )

        checkpoint = torch.load(ckpt["path"], map_location="cpu", weights_only=False)
        model_args = checkpoint["args"]

        model = create_model(
            arch=model_args["arch"],
            input_dim=N_FEATURES,
            hidden_size=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
            dropout=0.0,
            mixer_layers=model_args["mixer_layers"],
            head_hidden=model_args["head_hidden"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        n_scored = 0
        with torch.no_grad():
            for seq in valid_split:
                engine.reset()
                features = engine.process_sequence(seq["states"])
                features_t = torch.from_numpy(features).unsqueeze(0)

                preds, _ = model(features_t)
                preds = preds.squeeze(0).numpy()

                mask = seq["mask"].astype(bool)
                gix = seq["global_ix"]

                if gix not in all_oof_preds:
                    all_oof_preds[gix] = []
                    per_model_preds[gix] = {}
                    all_oof_targets[gix] = seq["targets"][mask]

                all_oof_preds[gix].append(preds[mask])
                per_model_preds[gix][model_id] = preds[mask]
                n_scored += 1

        wpc_info = checkpoint.get("wpc", "N/A")
        print(f"  {model_id}: {n_scored} OOF seqs (train WPC: {wpc_info})")

    print(f"\n{'='*60}")
    print("OUT-OF-FOLD ENSEMBLE RESULTS (EQUAL WEIGHTS)")
    print(f"{'='*60}")

    all_pred_list = []
    all_target_list = []

    for gix in sorted(all_oof_preds.keys()):
        ensemble_pred = np.mean(all_oof_preds[gix], axis=0)
        all_pred_list.append(ensemble_pred)
        all_target_list.append(all_oof_targets[gix])

    all_preds = np.concatenate(all_pred_list, axis=0)
    all_targets = np.concatenate(all_target_list, axis=0)

    t0_wpc = weighted_pearson_correlation(all_targets[:, 0], all_preds[:, 0])
    t1_wpc = weighted_pearson_correlation(all_targets[:, 1], all_preds[:, 1])
    avg_wpc = (t0_wpc + t1_wpc) / 2

    n_seqs = len(all_oof_preds)
    n_steps = len(all_preds)
    avg_models_per_seq = np.mean([len(v) for v in all_oof_preds.values()])

    print(f"  Sequences scored: {n_seqs}")
    print(f"  Steps scored: {n_steps:,}")
    print(f"  Avg models per sequence: {avg_models_per_seq:.1f}")
    print(f"  t0 WPC:  {t0_wpc:.6f}")
    print(f"  t1 WPC:  {t1_wpc:.6f}")
    print(f"  Avg WPC: {avg_wpc:.6f} (no data leakage)")
    print(f"{'='*60}")

    results = {"t0_wpc": t0_wpc, "t1_wpc": t1_wpc, "avg_wpc": avg_wpc}

    if do_optimize and len(model_ids_seen) > 1:
        print(f"\n{'='*60}")
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print(f"{'='*60}")

        model_ids = sorted(model_ids_seen)
        opt_weights = optimize_weights(per_model_preds, all_oof_targets, model_ids)

        output_dir = output_dir or os.path.join(PROJECT_DIR, "solution")
        weights_data = {
            "model_ids": model_ids,
            "weights": opt_weights.tolist(),
            "equal_weight_wpc": avg_wpc,
        }
        weights_path = os.path.join(output_dir, "weights.json")
        with open(weights_path, "w") as f:
            json.dump(weights_data, f, indent=2)
        print(f"\n  Weights saved to: {weights_path}")

        results["optimized_wpc"] = -float("inf")

    return results


def main():
    parser = argparse.ArgumentParser(description="Out-of-fold validation")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.join(PROJECT_DIR, "checkpoints"))
    parser.add_argument("--train-path", type=str,
                        default=os.path.join(PROJECT_DIR, "datasets", "train.parquet"))
    parser.add_argument("--valid-path", type=str,
                        default=os.path.join(PROJECT_DIR, "datasets", "valid.parquet"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--archs", type=str, nargs="+", default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Filter by model-ID prefix, e.g. --models gru_h128_l2_s42")
    parser.add_argument("--optimize-weights", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_oof_validation(
        checkpoint_dir=args.checkpoint_dir,
        train_path=args.train_path,
        valid_path=args.valid_path,
        n_folds=args.n_folds,
        fold_seed=args.fold_seed,
        seeds=args.seeds,
        archs=args.archs,
        hidden_sizes=args.hidden_sizes,
        model_ids_filter=args.models,
        do_optimize=args.optimize_weights,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
