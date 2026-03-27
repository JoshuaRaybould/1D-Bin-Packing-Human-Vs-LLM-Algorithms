import pandas as pd
import sys
import os

def main():
    # Check if at least one file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python analyze_algo.py <file1.csv> [file2.csv ...]")
        return

    # Load and combine all provided CSV files
    files = sys.argv[1:]
    dataframes = []

    for file in files:
        if os.path.isfile(file):
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"File not found: {file}")

    if not dataframes:
        print("No valid CSV files loaded. Exiting.")
        return

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Successfully loaded {len(combined_df)} rows from {len(dataframes)} file(s).")

    # Interactive loop
    while True:
        algo = input("\nEnter the algorithm name (e.g., openai-sa) or 'exit' to quit: ").strip()

        if algo.lower() == 'exit':
            break

        if algo not in combined_df['algo'].values:
            print(f"Algorithm '{algo}' not found. Available algorithms are:")
            print(", ".join(sorted(combined_df['algo'].unique())))
            continue

        # Filter data for the requested algorithm and create a copy to safely add new columns
        algo_data = combined_df[combined_df['algo'] == algo].copy()

        # Calculate the percentage difference for each individual run
        algo_data['pct_diff'] = (algo_data['avg_ratio'] - 1) * 100

        print(f"\n" + "="*50)
        print(f" RESULTS FOR: {algo}")
        print("="*50)

        # 1. Print optimal_hits, avg_ratio, and the new pct_diff values
        print("\n--- Individual Run Values ---")
        display_cols = ['dataset', 'num_instances', 'optimal_hits', 'avg_ratio', 'pct_diff']
        print(algo_data[display_cols].to_string(index=False))

        # 2. Calculate summaries
        total_instances = algo_data['num_instances'].sum()
        total_optimal_hits = algo_data['optimal_hits'].sum()

        if total_instances > 0:
            # Weighted average like normal
            weighted_avg_ratio = (algo_data['avg_ratio'] * algo_data['num_instances']).sum() / total_instances

            # The new metric: weighted average - 1 * 100
            pct_diff = (weighted_avg_ratio - 1) * 100

            print("\n--- Summary ---")
            print(f"Total instances across all runs: {total_instances}")
            print(f"Total optimal hits:              {total_optimal_hits}")
            print(f"Weighted Average of avg_ratio:   {weighted_avg_ratio:.6f}")
            print(f"Pct Difference:     {pct_diff:.4f}%")
        else:
            print("\n--- Summary ---")
            print(f"Total optimal hits:              {total_optimal_hits}")
            print("Cannot calculate weighted averages: Total num_instances is 0.")

if __name__ == "__main__":
    main()
