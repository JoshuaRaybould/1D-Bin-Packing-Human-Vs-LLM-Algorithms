from utilities import load_data
from utilities import test_correctness
import argparse
import csv
from pathlib import Path
import hashlib
import random
import time
from algorithms import randomised_best_fit, simulated_annealing, grouping_genetic_algorithm, tabu_search, ant_colony_optimisation, GRASP, variable_neighbourhood_search, FDO

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
      choices=["rbf", "sa", "tabu", "gga", "aco", "grasp", "vns", "fdo"],
      help="Which algorithm to run."
   )

   parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs on the same instances.")

    # mode selection 
   parser.add_argument(
      "--mode",
      default="default",
      choices=["default", "test", "choose", "generate"],
      help="How to run the selected algorithm."
   )

   # generation / writing options
   parser.add_argument("--outdir", default="instances", help="Output folder for generated instance .txt files.")

   # mode-specific options
   parser.add_argument("--num", type=int, default=10, help="For mode=choose: number of instances.")
   # parser.add_argument("--print-sols", action="store_true", help="Print per-instance solutions.")

   parser.add_argument("--count", type=int, default=20, help="For set=/our-uniform: number of instances.")
   parser.add_argument("--cap", type=int, default=100, help="For set=/our-uniform: bin capacity.")
   parser.add_argument("--items", type=int, default=100, help="For set=/our-uniform: number of items.")

   parser.add_argument(
   "--csv",
   default=None,
   help="Optional path to append one summary row (per dataset run) as CSV."
   )


   return parser


def select_algorithm(name: str):
    algorithm_map = {
        "rbf": randomised_best_fit.randomisedBestFit,
        "sa": simulated_annealing.simulatedAnnealingFFD,
        "tabu": tabu_search.tabuSearchFFD,
        "gga": grouping_genetic_algorithm.groupingGeneticAlgorithm,
        "aco": ant_colony_optimisation.antColonyOptimisation,
        "grasp": GRASP.reactiveGRASP,
        "vns": variable_neighbourhood_search.variableNeighbourhoodSearchFFD,
        "fdo": FDO.adaptiveFDO,
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

def applyAlgorithm(instances, chosenAlgorithm, runs):
   ratioScore = 0
   totalBins = 0
   totalOpt = 0
   totalTime = 0
   optimalHits = 0

   for instance in instances:
      bestBins = float("inf")
      optBins = instance["optimal_solution"]
      instanceId = instance["file_name"]

      # Decision to use time.perf_counter() was due to https://builtin.com/articles/timing-functions-python
      # time.time() is apparently not as precise and timeit is generally for small bits of code
      for x in range(0, runs):
         
         # Derive a unique, repeatable seed from (instanceId, runIndex) to make each trial reproducible and not affected by other runs.
         seedBytes = hashlib.sha256(f"{instanceId}|{x}".encode()).digest()
         curSeed = int.from_bytes(seedBytes[:8], "big")
         random.seed(curSeed)

         startTime = time.perf_counter()
         packing = chosenAlgorithm(instance["bin_capacity"], instance["weights"].copy())
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
      "num_items": len(instance["weights"])
   }


def append_summary_to_csv(filename: str, row: dict, results_dir: str = "Results"):
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

   if mode == "generate":

      instances = load_data.getOurRandomInstances(
         numInstances=args.count,
         capacity=args.cap,
         numItems=args.items,
         distribution="u"
      )

      # Put num items into the filename prefix
      prefix = f"our_u_{args.items}"

      out_dir = load_data.saveInstancesAsTxt(instances, prefix)

      print(f"Wrote {len(instances)} instances to: {out_dir.resolve()}")
      return

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

      summary = applyAlgorithm(chosenInstances, chosenAlgorithm, args.runs)

      if args.csv:
         row = {"dataset": args.set, "algo": args.algo, "runs": args.runs, **summary}
         append_summary_to_csv(args.csv, row)
      return

   # default
   summary = applyAlgorithm(instances, chosenAlgorithm, args.runs)
   if args.csv:
      row = {"dataset": args.set, "algo": args.algo, "runs": args.runs, **summary}
      append_summary_to_csv(args.csv, row)

if __name__ == "__main__":
   main()


