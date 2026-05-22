"""Drive multiple train.py runs to build the full ensemble in one go.

Usage:
    python src/train_ensemble.py
    python src/train_ensemble.py --quick
    python src/train_ensemble.py --configs gru_128 lstm_128
"""
import os
import sys
import subprocess
import argparse
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# (arch, hidden_size, num_layers, mixer_layers, head_hidden)
CONFIGS = {
    "gru_128":  ("gru",  128, 2, 3, 64),
    "gru_192":  ("gru",  192, 2, 3, 96),
    "gru_256":  ("gru",  256, 2, 3, 128),
    "lstm_128": ("lstm", 128, 2, 3, 64),
    "lstm_192": ("lstm", 192, 2, 3, 96),
}

DEFAULT_SEEDS = [42, 123]
DEFAULT_N_FOLDS = 5


def build_train_cmd(
    config_name: str,
    fold: int,
    seed: int,
    n_folds: int = 5,
    epochs: int = 60,
    fold_seed: int = 42,
    output_dir: str = None,
    extra_args: list = None,
) -> list:
    arch, hidden, layers, mixer, head = CONFIGS[config_name]
    output_dir = output_dir or os.path.join(PROJECT_DIR, "checkpoints")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "train.py"),
        "--arch", arch,
        "--hidden-size", str(hidden),
        "--num-layers", str(layers),
        "--mixer-layers", str(mixer),
        "--head-hidden", str(head),
        "--fold", str(fold),
        "--n-folds", str(n_folds),
        "--seed", str(seed),
        "--fold-seed", str(fold_seed),
        "--epochs", str(epochs),
        "--output-dir", output_dir,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_export_cmd(checkpoint_path: str, output_path: str = None) -> list:
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "export_onnx.py"),
        "--checkpoint", checkpoint_path,
        "--verify",
    ]
    if output_path:
        cmd.extend(["--output", output_path])
    return cmd


def get_checkpoint_name(config_name: str, seed: int, fold: int) -> str:
    arch, hidden, layers, _, _ = CONFIGS[config_name]
    return f"{arch}_h{hidden}_l{layers}_s{seed}_fold{fold}_best.pt"


def main():
    parser = argparse.ArgumentParser(description="Train an ensemble")
    parser.add_argument("--configs", nargs="+", default=None,
                        help=f"Which configs to train. Options: {list(CONFIGS.keys())}")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_DIR, "checkpoints"))
    parser.add_argument("--solution-dir", type=str,
                        default=os.path.join(PROJECT_DIR, "solution"))
    parser.add_argument("--quick", action="store_true",
                        help="1 fold, 1 seed, 10 epochs, gru_128 only. For smoke-testing.")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, only export existing checkpoints to ONNX.")
    parser.add_argument("--extra-args", nargs="*", default=None,
                        help="Forwarded to train.py, e.g. --noise 0.01")
    args = parser.parse_args()

    if args.quick:
        configs = ["gru_128"]
        seeds = [42]
        n_folds = 1
        epochs = 10
    else:
        configs = args.configs or list(CONFIGS.keys())
        seeds = args.seeds
        n_folds = args.n_folds
        epochs = args.epochs

    for c in configs:
        if c not in CONFIGS:
            print(f"Unknown config: {c}. Available: {list(CONFIGS.keys())}")
            sys.exit(1)

    total_models = len(configs) * len(seeds) * n_folds
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE TRAINING PLAN")
    print(f"  Configs: {configs}")
    print(f"  Seeds: {seeds}")
    print(f"  Folds: {n_folds}")
    print(f"  Epochs: {epochs}")
    print(f"  Total models: {total_models}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.solution_dir, exist_ok=True)

    results = []
    t_total_start = time.time()
    model_idx = 0

    for config_name in configs:
        for seed in seeds:
            for fold in range(n_folds):
                model_idx += 1
                ckpt_name = get_checkpoint_name(config_name, seed, fold)
                ckpt_path = os.path.join(args.output_dir, ckpt_name)

                if not args.export_only:
                    print(f"\n[{model_idx}/{total_models}] Training {config_name} seed={seed} fold={fold}")
                    print("-" * 60)

                    cmd = build_train_cmd(
                        config_name, fold, seed,
                        n_folds=n_folds, epochs=epochs,
                        fold_seed=args.fold_seed,
                        output_dir=args.output_dir,
                        extra_args=args.extra_args,
                    )

                    t_start = time.time()
                    result = subprocess.run(cmd, cwd=PROJECT_DIR)
                    elapsed = time.time() - t_start

                    if result.returncode != 0:
                        print(f"  FAILED (exit code {result.returncode})")
                        results.append({
                            "config": config_name, "seed": seed, "fold": fold,
                            "status": "FAILED", "time": elapsed,
                        })
                        continue

                    results.append({
                        "config": config_name, "seed": seed, "fold": fold,
                        "status": "OK", "time": elapsed,
                    })

                if not args.skip_export and os.path.exists(ckpt_path):
                    onnx_name = ckpt_name.replace("_best.pt", ".onnx")
                    onnx_path = os.path.join(args.solution_dir, onnx_name)

                    print(f"  Exporting to ONNX: {onnx_name}")
                    export_cmd = build_export_cmd(ckpt_path, onnx_path)
                    export_result = subprocess.run(export_cmd, cwd=PROJECT_DIR)
                    if export_result.returncode != 0:
                        print(f"  ONNX export FAILED for {ckpt_name}")

    total_time = time.time() - t_total_start

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE TRAINING COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    if results:
        ok = sum(1 for r in results if r["status"] == "OK")
        failed = sum(1 for r in results if r["status"] == "FAILED")
        print(f"  Succeeded: {ok}, Failed: {failed}")
    print(f"{'='*60}")

    onnx_files = sorted(f for f in os.listdir(args.solution_dir) if f.endswith(".onnx"))
    print(f"\n  ONNX models in {args.solution_dir}:")
    for f in onnx_files:
        size = os.path.getsize(os.path.join(args.solution_dir, f)) / 1024
        print(f"    {f} ({size:.1f} KB)")
    print(f"  Total: {len(onnx_files)} models")

    results_path = os.path.join(args.output_dir, "ensemble_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
