"""Run my solution against valid.parquet end-to-end using the official scorer.
This is what I used to sanity-check submissions before zipping them up."""
import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from utils import ScorerStepByStep


def main():
    parser = argparse.ArgumentParser(description="Local validation harness")
    parser.add_argument("--solution-dir", type=str,
                        default=os.path.join(PROJECT_DIR, "solution"))
    parser.add_argument("--data", type=str,
                        default=os.path.join(PROJECT_DIR, "datasets", "valid.parquet"))
    args = parser.parse_args()

    sys.path.insert(0, args.solution_dir)
    from solution import PredictionModel

    print(f"Solution dir: {args.solution_dir}")
    print(f"Data: {args.data}")

    model = PredictionModel()
    scorer = ScorerStepByStep(args.data)

    print("\nRunning validation...")
    t_start = time.time()
    results = scorer.score(model)
    elapsed = time.time() - t_start

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"  t0 WPC:  {results['t0']:.6f}")
    print(f"  t1 WPC:  {results['t1']:.6f}")
    print(f"  Avg WPC: {results['weighted_pearson']:.6f}")
    print(f"{'='*50}")

    n_seqs = len(scorer.dataset['seq_ix'].unique())
    n_steps = len(scorer.dataset)
    print(f"\nTiming:")
    print(f"  Total: {elapsed:.1f}s for {n_seqs} sequences ({n_steps:,} steps)")
    print(f"  Per step: {elapsed/n_steps*1000:.3f}ms")
    print(f"  Per sequence: {elapsed/n_seqs*1000:.1f}ms")

    test_seqs = 1500
    est_time = elapsed / n_seqs * test_seqs
    print(f"\nExtrapolated to test set ({test_seqs} sequences):")
    print(f"  Time: {est_time:.1f}s ({est_time/60:.1f} min)")
    print(f"  Budget: 60 min")
    print(f"  Status: {'OK' if est_time < 3600 else 'OVER BUDGET'}")


if __name__ == "__main__":
    main()
