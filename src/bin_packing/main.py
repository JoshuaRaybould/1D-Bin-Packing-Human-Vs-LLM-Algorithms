from bin_packing.utilities import load_data
from bin_packing.utilities import test_correctness
import argparse
import csv
from pathlib import Path
import hashlib
import random
import time
from bin_packing.algorithms import randomised_best_fit, simulated_annealing, grouping_genetic_algorithm, tabu_search, ant_colony_optimisation, GRASP, variable_neighbourhood_search, FDO
from bin_packing.llm_algorithms import (
    # Anthropic
    anthropic_ant_colony_optimisation,
    anthropic_fitness_dependent_optimiser,
    anthropic_genetic_algorithm,
    anthropic_GRASP,
    anthropic_simulated_annealing,
    anthropic_tabu_search,
    anthropic_variable_neighbourhood_search,
    
    # OpenAI
    openai_ant_colony_optimisation,
    openai_fitness_dependent_optimiser,
    openai_genetic_algorithm,
    openai_GRASP,
    openai_simulated_annealing,
    openai_tabu_search,
    openai_variable_neighbourhood_search
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

def build_parser():
   parser = argparse.ArgumentParser(
      description="Run bin packing algorithms on selectable instance sets."
   )

   # dataset selection 
   parser.add_argument(
      "--set",
      # required=True,
      choices=[
         # Well known datasets, for thorough testing
         "falkenauer-120",
         "falkenauer-250",
         "falkenauer-500",
         "falkenauer-1000",
         "scholl-10",
         "scholl-480",
         "scholl-720",
         "hard",

         # Some uniform instances we generated to check our algorithms perform reasonably
         # Contains 10 400 item istances and 10 800 item instances, each 150 bin capacity
         "test-u",

         # Uniform sets to test our algorithms
         "our-u-100",
         "our-u-200",
         "our-u-400",
         "our-u-600",
         "our-u-800",
         "our-u-1000",
         "our-u-1200",
         "our-u-1400",
         "our-u-1600"
      ],
      help="Which instance set to load."
   )

    # algorithm selection
   parser.add_argument(
      "--algo",
      # required=True,
      choices=["rbf", "sa", "ts", "ga", "aco", "grasp", "vns", "fdo", 
               "anthropic-aco", "anthropic-fdo", "anthropic-ga", "anthropic-grasp", "anthropic-sa", "anthropic-ts", "anthropic-vns", 
               "openai-aco", "openai-fdo", "openai-ga", "openai-grasp", "openai-sa", "openai-ts", "openai-vns"
               ],
      help="Which algorithm to run."
   )

   parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs on the same instances.")

    # mode selection 
   parser.add_argument(
      "--mode",
      default="default",
      choices=["default", "test", "choose"],
      help="How to run the selected algorithm."
   )

   # mode-specific options
   parser.add_argument("--num", type=int, default=10, help="For mode=choose: number of instances.")
   # parser.add_argument("--print-sols", action="store_true", help="Print per-instance solutions.")

   parser.add_argument(
   "--csv",
   default=None,
   help="Optional path to append one summary row (per dataset run) as CSV."
   )

   parser.add_argument(
      "--tpi",
      type=float,
      default=(1/600),
      help="Time per item in seconds (time limit per instance = num_items * tpi). Default=1/600."
   )

   parser.add_argument(
      "--use-limit",
      action="store_true",
      help="Flag to tell human written algorithms to use just the time limit."
   )

   return parser

TIMED_ALGORITHMS = {"sa", "ts", "ga", "aco", "grasp", "vns", "fdo"}

def select_algorithm(name: str):
    algorithm_map = {
         "rbf": randomised_best_fit.randomisedBestFit,
         "sa": simulated_annealing.simulatedAnnealingFFD,
         "ts": tabu_search.tabuSearchFFD,
         "ga": grouping_genetic_algorithm.groupingGeneticAlgorithm,
         "aco": ant_colony_optimisation.antColonyOptimisation,
         "grasp": GRASP.reactiveGRASP,
         "vns": variable_neighbourhood_search.variableNeighbourhoodSearchFFD,
         "fdo": FDO.FDO,

         # LLM Generated Algotihms
         # Anthropic
         "anthropic-aco": anthropic_ant_colony_optimisation.solve,
         "anthropic-fdo": anthropic_fitness_dependent_optimiser.solve,
         "anthropic-ga": anthropic_genetic_algorithm.solve,
         "anthropic-grasp": anthropic_GRASP.solve,
         "anthropic-sa": anthropic_simulated_annealing.solve,
         "anthropic-ts": anthropic_tabu_search.solve,
         "anthropic-vns": anthropic_variable_neighbourhood_search.solve,
               
         # OpenAI
         "openai-aco": openai_ant_colony_optimisation.solve,
         "openai-fdo": openai_fitness_dependent_optimiser.solve,
         "openai-ga": openai_genetic_algorithm.solve,
         "openai-grasp": openai_GRASP.solve,
         "openai-sa": openai_simulated_annealing.solve,
         "openai-ts": openai_tabu_search.solve,
         "openai-vns": openai_variable_neighbourhood_search.solve,

    }
    
    return algorithm_map[name]


def load_instances(args):
   set_name = args.set

   if set_name.startswith("falkenauer-"):
      n = set_name.split("-", 1)[1]
      return load_data.getFalkenauer(n)   

   if set_name.startswith("scholl-"):
      n = set_name.split("-", 1)[1]
      return load_data.getSchollInstances(n)

   if set_name == "hard":
      return load_data.getHardInstances()

   if set_name == "test-u":
      return load_data.getTestInstances()
   
   if set_name.startswith("our-u-"):
      n = set_name[len("our-u-"):]
      return load_data.getOurUniformInstances(n)
   
   raise ValueError(f"Unknown set: {set_name}")

def applyAlgorithm(instances, chosenAlgorithm, algo_name, runs, timePerItem=(1/600), use_limit=False):
   # print(timePerItem)
   print("Started")
   ratioScore = 0
   totalBins = 0
   totalOpt = 0
   totalTime = 0
   optimalHits = 0

   for instance in instances:
      timeLimit = len(instance["weights"]) * timePerItem
      bestBins = float("inf")
      optBins = instance["optimal_solution"]
      instanceId = instance["file_name"]

      for x in range(0, runs):
         seedBytes = hashlib.sha256(f"{instanceId}|{x}".encode()).digest()
         curSeed = int.from_bytes(seedBytes[:8], "big")
         random.seed(curSeed)

         startTime = time.perf_counter()
         
         if algo_name in TIMED_ALGORITHMS:
             packing = chosenAlgorithm(instance["bin_capacity"], instance["weights"].copy(), timeLimit, use_limit)
         else:
             packing = chosenAlgorithm(instance["bin_capacity"], instance["weights"].copy(), timeLimit)
         
         endTime = time.perf_counter()
         totalTime += (endTime - startTime)

         algBins = len(packing["bin_weights"])

         if algBins < bestBins:
            bestBins = algBins

         totalBins += algBins

         test_correctness.quickValidatePacking(instance, packing, algBins, optBins)
         
         totalOpt += optBins
         ratioScore += (algBins/optBins)

      if optBins == bestBins:
         optimalHits += 1

   extraBins = totalBins - totalOpt
   overall_ratio = (extraBins + totalOpt) / totalOpt
   avg_ratio = ratioScore / (len(instances) * runs)

   print("Time: " + str(totalTime))
   print("Bins used: " + str(totalBins))
   print("Optimal number of bins: " + str(totalOpt))
   print("Ratio of bins used by algorithm to bins used in optimal: " +  str(overall_ratio))
   print("Extra bins (compared to optimal or lb): " + str(extraBins))
   print("Average ratio of bins used by algorithm to bins used in optimal case: " + str(avg_ratio))
   print("Number of instances used: " + str(len(instances)))
   print("optimal was reached for " + str(optimalHits) + " of the instances")

   return {
      "time_sec": totalTime,
      "bins_used": totalBins,
      "optimal_bins": totalOpt,
      "extra_bins": extraBins,
      "overall_ratio": overall_ratio,
      "num_instances": len(instances),
      "optimal_hits": optimalHits,
      "avg_ratio": avg_ratio,
      "time_per_item": timePerItem
   }


def append_summary_to_csv(filename: str, row: dict, results_dir: str | Path = DEFAULT_RESULTS_DIR):
    # Create Results/ if it doesn't exist
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Build full path: Results/<filename>
    csv_path = results_path / filename

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(): 
   parser = build_parser()
   args = parser.parse_args()

   mode = args.mode

   chosenAlgorithm = select_algorithm(args.algo)
   instances = load_instances(args)

   if mode == "test":
      if test_correctness.testAlgorithmCorrectness(chosenAlgorithm, instances):
         print("Seems to be correct")
      return

   if mode == "choose":
      numInstances = args.num
      chosenInstances = []
      for instance in instances:
         chosenInstances.append(instance)
         if len(chosenInstances) == numInstances:
            break

      summary = applyAlgorithm(chosenInstances, chosenAlgorithm, args.algo, args.runs, args.tpi, args.use_limit)

      if args.csv:
         row = {"dataset": args.set, "algo": args.algo, "runs": args.runs, **summary}
         append_summary_to_csv(args.csv, row)
      return

   # default
   summary = applyAlgorithm(instances, chosenAlgorithm, args.algo, args.runs, args.tpi, args.use_limit)
   if args.csv:
      row = {"dataset": args.set, "algo": args.algo, "runs": args.runs, **summary}
      append_summary_to_csv(args.csv, row)

if __name__ == "__main__":
   main()


