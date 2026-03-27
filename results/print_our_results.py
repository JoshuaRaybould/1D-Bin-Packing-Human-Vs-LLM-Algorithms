"""
verify_results.py
-----------------
Reads the algorithm benchmark CSV and computes the summary table
used in Table 1 of the paper.
Metrics:
  - Execution Time (s): time_sec / runs  (mean time per single run)
  - dif%: (avg_ratio - 1) * 100          (percentage above lower bound)
Rounding: raw values stored at full precision, rounded only at display.
Usage:
  python3 verify_results.py my_test_results.csv
"""
import csv
import sys
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Order matches the paper table (left to right, best to worst quality)
ALGO_ORDER = ['gga', 'vns', 'sa', 'ts', 'grasp', 'aco', 'fdo', 'rbf']
ALGO_LABELS = {
    'sa': 'SA', 'gga': 'GA', 'ts': 'TS', 'grasp': 'GRASP',
    'vns': 'VNS', 'aco': 'ACO', 'fdo': 'FDO', 'rbf': 'RBF'
}
SIZES = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
# ---------------------------------------------------------------------------
# Load data — store at full precision, round only at display
# ---------------------------------------------------------------------------
def load_csv(filepath):
    results = {}
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['dataset'].split('-')[-1])
            algo = row['algo'].strip().lower()
            runs = int(row['runs'])
            time_total = float(row['time_sec'])
            avg_ratio = float(row['avg_ratio'])
            time_per_run = time_total / runs
            dif_pct = (avg_ratio - 1) * 100
            results[(n, algo)] = (time_per_run, dif_pct)  # full precision
    return results
# ---------------------------------------------------------------------------
# Compute average dif% per algorithm across all sizes (full precision)
# ---------------------------------------------------------------------------
def compute_avg_difs(results):
    avg_difs = {}
    for algo in ALGO_ORDER:
        dif_values = [results[(n, algo)][1] for n in SIZES if (n, algo) in results]
        avg_difs[algo] = sum(dif_values) / len(dif_values) if dif_values else None
    return avg_difs
# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
def print_table(results, avg_difs):
    header = f"{'n':<6}"
    for algo in ALGO_ORDER:
        label = ALGO_LABELS[algo]
        header += f"  {label:>8}  {'dif%':>5}"
    print(header)
    separator = "-" * (6 + len(ALGO_ORDER) * 17)
    print(separator)

    for n in SIZES:
        row_str = f"{n:<6}"
        for algo in ALGO_ORDER:
            key = (n, algo)
            if key in results:
                time_val, dif_val = results[key]
                row_str += f"  {time_val:>8.2f}  {dif_val:>5.2f}"  # round here only
            else:
                row_str += f"  {'N/A':>8}  {'N/A':>5}"
        print(row_str)

    # Average dif% row — time columns left blank
    print(separator)
    avg_str = f"{'Avg dif%':<6}"
    for algo in ALGO_ORDER:
        if avg_difs[algo] is not None:
            avg_str += f"  {'':>8}  {avg_difs[algo]:>5.2f}"  # round only here
        else:
            avg_str += f"  {'':>8}  {'N/A':>5}"
    print(avg_str)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 verify_results.py <path_to_csv>")
        sys.exit(1)
    filepath = sys.argv[1]
    results = load_csv(filepath)
    avg_difs = compute_avg_difs(results)
    print_table(results, avg_difs)
