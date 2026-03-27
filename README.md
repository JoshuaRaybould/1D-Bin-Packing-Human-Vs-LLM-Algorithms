# Evaluating Randomised Algorithms and Their LLM Generated Counterparts on the Bin Packing Problem

Final Year Project — Joshua Raybould

## Overview

This project implements eight randomised algorithms for the one-dimensional offline Bin Packing Problem (BPP) and compares them against implementations generated from scratch by two Large Language Models: OpenAI's GPT-5.2 and Anthropic's Claude Opus 4.6. An automated iterative refinement pipeline was used to generate LLM counterparts for each algorithm.

The human algorithm implementations — the primary (human) contribution of this project — are located in `src/bin_packing/algorithms/`. A longer form report is available at `docs/main.pdf`.

Note that in existing results files our Genetic Algorithm (GA) is referred to as gga rather than ga, we have updated the name to ga for consistency.
Also the Falkenauer U and Scholl datasets are not included in our repository, so algorithms can not be ran on those.


## Requirements

- Python 3.11+
- NumPy (for instance generation only)
- OpenAI and Anthropic Python SDKs (for LLM algorithm generation only)

## Usage

All commands should be run from the `src/` directory.

### Running an algorithm on a dataset

```bash
python -m bin_packing.main --algo ga --set our-u-200 --runs 5 --use-limit --csv results.csv
```

- `--algo`: Algorithm to run. Options: `rbf`, `sa`, `ts`, `ga`, `aco`, `grasp`, `vns`, `fdo`, and LLM variants prefixed with `anthropic-` or `openai-` (e.g., `anthropic-sa`, `openai-aco`).
- `--set`: Instance set. Options include `falkenauer-120`, `falkenauer-250`, `falkenauer-500`, `falkenauer-1000`, `scholl-10`, `scholl-480`, `scholl-720`, `hard`, `test-u`, and generated uniform sets (`our-u-100` through `our-u-1600`).
- `--runs`: Number of independent runs per instance (default: 1).
- `--use-limit`: Tells human algorithms to use only the time limit (required for fair comparison with LLM algorithms).
- `--tpi`: Time per item in seconds (default: 1/600).
- `--csv`: Optional CSV filename to append summary results.

### Generating instances

```bash
python -m bin_packing.utilities.generate_instances --count 20 --cap 150 --items 500
```

### Stability verification

```bash
python -m bin_packing.utilities.variance_check --algo ga --runs 5 --use-limit
```

Runs the specified algorithm across the full BPPLIB benchmark suite and records per-instance and aggregate stability data.

### Printing results tables

From the `results/` directory:
```bash
python print_final_results.py falkenauer_results.csv scholl_results.csv
python print_our_results.py my_test_results.csv
```

## Project Structure

```
src/bin_packing/
  main.py                   Entry point for running algorithms

  algorithms/               Human-implemented algorithms (primary contribution)
    helpers.py              Shared components (FFD, L2 lower bound)
    simulated_annealing.py
    tabu_search.py
    variable_neighbourhood_search.py
    GRASP.py
    grouping_genetic_algorithm.py
    ant_colony_optimisation.py
    FDO.py
    randomised_best_fit.py

  llm_algorithms/           Selected best LLM implementations
    anthropic_*.py          Anthropic's implementations (7 algorithms)
    openai_*.py             OpenAI's implementations (7 algorithms)

  llm_generation/           LLM generation pipeline
    generate_algorithms.py  Main generation loop
    prompts.py              Prompt templates (shared across both models)
    providers.py            API wrappers for OpenAI and Anthropic
    algorithms_tests.py     Correctness validation suite

  utilities/
    load_data.py            Instance loading
    test_correctness.py     Packing validation
    generate_instances.py   Standalone instance generator
    variance_check.py       Stability verification script
    scripts/                Bash scripts for batch runs

datasets/                   
  my_instances/             Our generated instances

docs/                       Reports and figures
  main.pdf                  Longer form report

llm_artifacts/              All LLM responses during generation
  anthropic/                Anthropic responses, organised by algorithm
  openai/                   OpenAI responses, organised by algorithm

notebooks/                  Jupyter notebook with baseline evaluation graphs

results/                    CSV result files and display scripts
  falkenauer_results.csv    LLM comparison results (Falkenauer)
  scholl_results.csv        LLM comparison results (Scholl)
  my_test_results.csv       Baseline evaluation results (generated instances)
  print_final_results.py    Display LLM comparison tables
  print_our_results.py      Display baseline evaluation tables
  stabilitycheck/           Stability verification results
    summary.csv             Per-algorithm aggregate stability
    per_instance.csv        Per-instance bin counts across runs
```

## Algorithms Implemented

**Human implementations:** Randomised Best Fit, Simulated Annealing, Tabu Search, Variable Neighbourhood Search, GRASP, Grouping Genetic Algorithm (GGA-CGT), Ant Colony Optimisation (MMAS), Fitness Dependent Optimizer.

**LLM implementations:** Each of the above (excluding RBF) was independently generated by both GPT-5.2 and Claude Opus 4.6 through the iterative refinement pipeline.

## Datasets

- **Baseline evaluation (Table 1):** Generated uniform instances with C=150, sizes 100-1600, 20 instances per size class.
- **LLM comparison (Table 2):** Standard BPPLIB benchmarks — Falkenauer U (120, 250, 500, 1000 items) and Scholl (10, 480, 720 instances) — totalling 1290 instances with known optima.
