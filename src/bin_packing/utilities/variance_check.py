# Written by AI, used to check stability of our results
"""
Usage:
    python -m bin_packing.utilities.variance_check --algo ga --num-runs 5 --use-limit
    python -m bin_packing.utilities.variance_check --algo anthropic-ga --num-runs 5
"""

import hashlib
import random
import csv
import argparse
from pathlib import Path
from bin_packing.utilities import load_data
from bin_packing.main import select_algorithm, TIMED_ALGORITHMS

BENCHMARK_SETS = [
    ("falkenauer-120", lambda: load_data.getFalkenauer("120")),
    ("falkenauer-250", lambda: load_data.getFalkenauer("250")),
    ("falkenauer-500", lambda: load_data.getFalkenauer("500")),
    ("falkenauer-1000", lambda: load_data.getFalkenauer("1000")),
    ("scholl-10", lambda: load_data.getSchollInstances("10")),
    ("scholl-480", lambda: load_data.getSchollInstances("480")),
    ("scholl-720", lambda: load_data.getSchollInstances("720")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--use-limit", action="store_true")
    parser.add_argument("--tpi", type=float, default=1/600)
    parser.add_argument("--instance-csv", default="variance_per_instance.csv")
    parser.add_argument("--summary-csv", default="variance_summary.csv")
    args = parser.parse_args()

    # --- NEW: Setup the results directory ---
    results_dir = Path("..") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # ----------------------------------------

    algo_func = select_algorithm(args.algo)
    num_runs = args.runs

    # Load all datasets
    print("Loading datasets...")
    all_datasets = [(name, loader()) for name, loader in BENCHMARK_SETS]
    total_instances = sum(len(insts) for _, insts in all_datasets)
    print(f"Loaded {total_instances} instances across {len(BENCHMARK_SETS)} classes\n")

    # Per-run weighted average dif% accumulators
    run_totals = [0.0] * num_runs
    run_counts = [0] * num_runs

    # --- UPDATED: Route instance CSV to the results folder ---
    instance_path = results_dir / args.instance_csv
    instance_exists = instance_path.exists()
    instance_file = instance_path.open("a", newline="", encoding="utf-8")
    inst_fields = ["algo", "dataset", "instance", "optimal"] + [f"run_{i}" for i in range(num_runs)] + ["mean", "range"]
    inst_writer = csv.DictWriter(instance_file, fieldnames=inst_fields)
    if not instance_exists:
        inst_writer.writeheader()

    print(f"{'Dataset':<16} {'Instance':<25} {'Opt':>4}  {'Runs':<30}  {'Range':>5}")
    print("-" * 90)

    for set_name, instances in all_datasets:
        for instance in instances:
            n_items = len(instance["weights"])
            time_limit = n_items * args.tpi
            opt = instance["optimal_solution"]
            instance_id = instance["file_name"]
            run_results = []

            for r in range(num_runs):
                seed_bytes = hashlib.sha256(f"{instance_id}|{r}".encode()).digest()
                cur_seed = int.from_bytes(seed_bytes[:8], "big")
                random.seed(cur_seed)

                if args.algo in TIMED_ALGORITHMS:
                    packing = algo_func(instance["bin_capacity"], instance["weights"].copy(), time_limit, args.use_limit)
                else:
                    packing = algo_func(instance["bin_capacity"], instance["weights"].copy(), time_limit)

                bins_used = len(packing["bin_weights"])
                run_results.append(bins_used)

                dif = (bins_used / opt) - 1
                run_totals[r] += dif
                run_counts[r] += 1

            mean_bins = sum(run_results) / len(run_results)
            rng = max(run_results) - min(run_results)
            runs_str = ", ".join(str(b) for b in run_results)
            print(f"{set_name:<16} {instance_id:<25} {opt:>4}  {runs_str:<30}  {rng:>5}")

            # Write per-instance row
            row = {
                "algo": args.algo,
                "dataset": set_name,
                "instance": instance_id,
                "optimal": opt,
                "mean": f"{mean_bins:.2f}",
                "range": rng,
            }
            for i, val in enumerate(run_results):
                row[f"run_{i}"] = val
            inst_writer.writerow(row)

    instance_file.close()

    # Compute per-run weighted average dif%
    run_difs = [(run_totals[r] / run_counts[r]) * 100 for r in range(num_runs)]

    print("\n" + "=" * 50)
    print(f"{'Run':<6} {'Weighted avg dif%':>18}")
    print("-" * 26)
    for r, dif in enumerate(run_difs):
        print(f"{r:<6} {dif:>17.4f}%")

    mean = sum(run_difs) / len(run_difs)
    lo, hi = min(run_difs), max(run_difs)
    spread = hi - lo
    print("-" * 26)
    print(f"Mean:  {mean:.4f}%")
    print(f"Range: {lo:.4f}% - {hi:.4f}%  (spread: {spread:.4f}%)")

    summary_path = results_dir / args.summary_csv
    summary_exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        fields = ["algo", "num_runs", "mean_dif_pct", "min_dif_pct", "max_dif_pct", "spread"] + [f"run_{i}" for i in range(num_runs)]
        writer = csv.DictWriter(f, fieldnames=fields)
        if not summary_exists:
            writer.writeheader()
        row = {
            "algo": args.algo,
            "num_runs": num_runs,
            "mean_dif_pct": f"{mean:.4f}",
            "min_dif_pct": f"{lo:.4f}",
            "max_dif_pct": f"{hi:.4f}",
            "spread": f"{spread:.4f}",
        }
        for i, val in enumerate(run_difs):
            row[f"run_{i}"] = f"{val:.4f}"
        writer.writerow(row)

    print(f"\nPer-instance results: {instance_path}")
    print(f"Summary results:     {summary_path}")


if __name__ == "__main__":
    main()