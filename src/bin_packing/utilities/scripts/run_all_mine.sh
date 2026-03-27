#!/bin/bash

# Define the algorithm list (Only time-limited core algorithms)
ALGOS=("rbf" "sa" "ts" "ga" "aco" "grasp" "vns" "fdo")

# Define dataset groups
FALKENAUER=("falkenauer-120" "falkenauer-250" "falkenauer-500" "falkenauer-1000")
SCHOLL=("scholl-10" "scholl-480" "scholl-720")

# --- DIRECTORY RESOLUTION ---
# Find the exact directory this script is in (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Jump 3 levels up to reach the 'src' directory (scripts -> utilities -> bin_packing -> src)
SRC_DIR="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"

# Change into the src directory
cd "$SRC_DIR" || exit

echo "================================================="
echo " Starting Timed Experiments for Falkenauer & Scholl"
echo " Working directory set to: $PWD"
echo "================================================="

# --- RUN FALKENAUER DATASETS ---
CSV_FILE="falkenauer_results.csv"
echo ""
echo "================================================="
echo " Phase 1: Running Falkenauer Datasets"
echo " Results will be saved to: $CSV_FILE"
echo "================================================="

for dataset in "${FALKENAUER[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "-> Running Dataset: $dataset | Algorithm: $algo | Runs: 5"
        
        # Execute the python module with --use-time-limit
        python3 -m bin_packing.main --algo "$algo" --set "$dataset" --csv "$CSV_FILE" --runs 5 --use-limit
        
        # Check if the command succeeded
        if [ $? -ne 0 ]; then
            echo "   [!] Error running $algo on $dataset. Moving to next..."
        fi
    done
done

# --- RUN SCHOLL DATASETS ---
CSV_FILE="scholl_results.csv"
echo ""
echo "================================================="
echo " Phase 2: Running Scholl Datasets"
echo " Results will be saved to: $CSV_FILE"
echo "================================================="

for dataset in "${SCHOLL[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "-> Running Dataset: $dataset | Algorithm: $algo | Runs: 5"
        
        # Execute the python module with --use-time-limit
        python3 -m bin_packing.main --algo "$algo" --set "$dataset" --csv "$CSV_FILE" --runs 5 --use-limit
        
        # Check if the command succeeded
        if [ $? -ne 0 ]; then
            echo "   [!] Error running $algo on $dataset. Moving to next..."
        fi
    done
done

echo ""
echo "================================================="
echo " All timed experiments completed successfully!"
echo "================================================="
