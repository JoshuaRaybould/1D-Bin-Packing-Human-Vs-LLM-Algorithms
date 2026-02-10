#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# Prefer a local venv if present; otherwise use system python3
if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
  PY="$PROJECT_DIR/venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PY="$PROJECT_DIR/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

MAIN="main.py"
RUNS=5
CSV="my_test_results.csv"

algos=(aco sa tabu gga vns fdo rbf grasp)

# sets=(our-u-100 our-u-200 our-u-400 our-u-800 our-u-1600)
sets=(our-u-100 our-u-200 our-u-400 our-u-600 our-u-800 our-u-1000 our-u-1200 our-u-1400 our-u-1600)

for algo in "${algos[@]}"; do
  for setname in "${sets[@]}"; do
    echo "Running: algo=$algo set=$setname runs=$RUNS"
    "$PY" "$MAIN" --algo "$algo" --set "$setname" --runs "$RUNS" --csv "$CSV"
  done
done

echo "Done. CSV: $PROJECT_DIR/$CSV"

