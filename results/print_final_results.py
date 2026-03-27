"""
display_llm_tables.py
---------------------
Reads one or two benchmark CSV files (falkenauer and/or scholl) and prints
the comparison tables for each algorithm, matching the paper format.

Tables are ordered by human baseline performance (best to worst) on the
Falkenauer/Scholl benchmarks: GA -> SA -> VNS -> TABU -> GRASP -> ACO -> FDO

Rounding: all weighted averages are computed at full precision and only
rounded at the point of display, avoiding accumulated rounding errors.

Usage:
  python3 display_llm_tables.py falkenauer_results.csv
  python3 display_llm_tables.py scholl_results.csv
  python3 display_llm_tables.py falkenauer_results.csv scholl_results.csv
"""

import csv
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ordered by human baseline performance on Falkenauer/Scholl (best to worst)
ALGO_TABLES = [
    ("GA",   "gga",   "anthropic-ga",    "openai-ga"),
    ("VNS",   "vns",   "anthropic-vns",   "openai-vns"),
    ("SA",    "sa",    "anthropic-sa",    "openai-sa"),
    ("TS",  "ts",  "anthropic-ts",  "openai-ts"),
    ("GRASP", "grasp", "anthropic-grasp", "openai-grasp"),
    ("ACO",   "aco",   "anthropic-aco",   "openai-aco"),
    ("FDO",   "fdo",   "anthropic-fdo",   "openai-fdo"),
]

DATASET_ORDER = [
    "falkenauer-120",
    "falkenauer-250",
    "falkenauer-500",
    "falkenauer-1000",
    "scholl-10",
    "scholl-480",
    "scholl-720",
]

# ---------------------------------------------------------------------------
# Load data — store raw avg_ratio at full precision, round only for display
# ---------------------------------------------------------------------------

def load_csvs(filepaths):
    data = {}
    for filepath in filepaths:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row['dataset'].strip()
                algo = row['algo'].strip().lower()
                opt = int(row['optimal_hits'])
                avg_ratio = float(row['avg_ratio'])   # full precision
                num_inst = int(row['num_instances'])
                data[(dataset, algo)] = (opt, avg_ratio, num_inst)
    return data

# ---------------------------------------------------------------------------
# Print a single comparison table
# ---------------------------------------------------------------------------

def print_table(table_name, human_key, anthropic_key, openai_key, data, datasets):
    present_datasets = [ds for ds in datasets if (ds, human_key) in data]
    if not present_datasets:
        print(f"\n[Skipping {table_name} -- no data found]")
        return

    ant_label = f"Anthropic-{table_name}"
    oai_label = f"OpenAI-{table_name}"

    print(f"\n{'=' * 70}")
    print(f"  {table_name}")
    print(f"{'=' * 70}")
    print(f"  {'Dataset':<20} {'Inst':>5}  "
          f"{'opt':>5} {'dif%':>6}  "
          f"{'opt':>5} {'dif%':>6}  "
          f"{'opt':>5} {'dif%':>6}")
    print(f"  {'':20} {'':>5}  "
          f"{table_name:>11}  "
          f"{ant_label:>11}  "
          f"{oai_label:>11}")
    print(f"  {'-' * 65}")

    total_inst = 0
    total_opt  = {k: 0   for k in [human_key, anthropic_key, openai_key]}
    weight_sum = {k: 0.0 for k in [human_key, anthropic_key, openai_key]}

    for ds in present_datasets:
        inst = data[(ds, human_key)][2]
        total_inst += inst
        row_str = f"  {ds:<20} {inst:>5}"
        for key in [human_key, anthropic_key, openai_key]:
            if (ds, key) in data:
                opt, avg_ratio, _ = data[(ds, key)]
                dif_display = round((avg_ratio - 1) * 100, 2)
                total_opt[key]  += opt
                weight_sum[key] += (avg_ratio - 1) * 100 * inst  # full precision
                row_str += f"  {opt:>5} {dif_display:>6.2f}"
            else:
                row_str += f"  {'N/A':>5} {'N/A':>6}"
        print(row_str)

    # Total row: sum of instances and optimal hits only, no dif% here
    print(f"  {'-' * 65}")
    total_str = f"  {'Total':<20} {total_inst:>5}"
    for key in [human_key, anthropic_key, openai_key]:
        total_str += f"  {total_opt[key]:>5} {'':>6}"
    print(total_str)

    # Weighted avg dif% row: computed at full precision, rounded only here
    wavg_str = f"  {'Weighted Avg dif%':<20} {'':>5}"
    for key in [human_key, anthropic_key, openai_key]:
        wavg = round(weight_sum[key] / total_inst, 2) if total_inst > 0 else 0
        wavg_str += f"  {'':>5} {wavg:>6.2f}"
    print(wavg_str)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 display_llm_tables.py <csv1> [csv2]")
        sys.exit(1)

    filepaths = sys.argv[1:]
    data = load_csvs(filepaths)

    present_datasets = [ds for ds in DATASET_ORDER
                        if any((ds, algo) in data
                               for _, algo, _, _ in ALGO_TABLES)]

    if not present_datasets:
        print("No recognised datasets found in the provided files.")
        sys.exit(1)

    print(f"\nDatasets found: {present_datasets}")

    for table_name, human_key, anthropic_key, openai_key in ALGO_TABLES:
        print_table(table_name, human_key, anthropic_key, openai_key,
                    data, present_datasets)
