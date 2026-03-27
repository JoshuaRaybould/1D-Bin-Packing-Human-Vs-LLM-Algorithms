#!/bin/bash
ALGOS=(
    "vns" "fdo" "rbf" "sa" "ts" "ga" "aco" "grasp"
)
DATASETS=(
    "our-u-100" "our-u-200" "our-u-400" "our-u-600" "our-u-800"
    "our-u-1000" "our-u-1200" "our-u-1400"
    "our-u-1600"
)
CSV_FILE="my_test_results.csv"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_DIR="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
cd "$SRC_DIR" || exit
echo "================================================="
echo " Starting experiments"
echo " Results will be saved to: $CSV_FILE"
echo " Working directory set to: $PWD"
echo "================================================="
for dataset in "${DATASETS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "-> Running Dataset: $dataset | Algorithm: $algo | Runs: 5"
        python3 -m bin_packing.main --algo "$algo" --set "$dataset" --csv "$CSV_FILE" --runs 5
        if [ $? -ne 0 ]; then
            echo "   [!] Error running $algo on $dataset. Moving to next..."
        fi
    done
done
echo "================================================="
echo " Experiments completed!"
