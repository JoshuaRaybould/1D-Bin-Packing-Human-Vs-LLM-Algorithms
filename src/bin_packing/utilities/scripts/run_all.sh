#!/bin/bash

# Define the algorithm list
ALGOS=(
    # Core Algorithms
    "rbf" "sa" "tabu" "gga" "aco" "grasp" "vns" "fdo"
    # Anthropic LLM Algorithms
    "anthropic-aco" "anthropic-fdo" "anthropic-ga" "anthropic-grasp" "anthropic-sa" "anthropic-tabu" "anthropic-vns"
    # OpenAI LLM Algorithms
    "openai-aco" "openai-fdo" "openai-ga" "openai-grasp" "openai-sa" "openai-tabu" "openai-vns"
)

# Define dataset groups
FALKENAUER=("falkenauer-120" "falkenauer-250" "falkenauer-500" "falkenauer-1000")
SCHOLL=("scholl-10" "scholl-480" "scholl-720")
HARD=("hard")

# --- ARGUMENT PARSING & CSV ROUTING ---
case "$1" in
    falkenauer)
        DATASETS=("${FALKENAUER[@]}")
        CSV_FILE="falkenauer_results.csv"
        ;;
    scholl)
        DATASETS=("${SCHOLL[@]}")
        CSV_FILE="scholl_results.csv"
        ;;
    hard)
        DATASETS=("${HARD[@]}")
        CSV_FILE="hard_results.csv"
        ;;
    all)
        # Recursively run the script for each group to keep CSVs separate
        echo "Running all dataset groups sequentially..."
        bash "$0" falkenauer
        bash "$0" scholl
        bash "$0" hard
        echo "================================================="
        echo " All dataset groups completed!"
        exit 0
        ;;
    *)
        echo "Usage: $0 {falkenauer|scholl|hard|all}"
        echo "Example: ./run_experiments.sh falkenauer"
        exit 1
        ;;
esac

# --- DIRECTORY RESOLUTION ---
# Find the exact directory this script is in (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Jump 3 levels up to reach the 'src' directory (scripts -> utilities -> bin_packing -> src)
SRC_DIR="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"

# Change into the src directory
cd "$SRC_DIR" || exit

# --- EXECUTION ---
echo "================================================="
echo " Starting experiments for: $1"
echo " Results will be saved to: $CSV_FILE"
echo " Working directory set to: $PWD"
echo "================================================="

for dataset in "${DATASETS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "-> Running Dataset: $dataset | Algorithm: $algo | Runs: 5"
        
        # Execute the python module matching your exact command format
        python3 -m bin_packing.main --algo "$algo" --set "$dataset" --csv "$CSV_FILE" --runs 5
        
        # Check if the command succeeded
        if [ $? -ne 0 ]; then
            echo "   [!] Error running $algo on $dataset. Moving to next..."
        fi
    done
done

echo "================================================="
echo " $1 experiments completed!"