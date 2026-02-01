from utilities import load_data
from utilities import test_correctness
import argparse
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
         "our-u-800",
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
      choices=["default", "test", "choose", "small", "generate"],
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
   
   if set_name.startswith("our-uniform-"):
      n = set_name.split("-", 1)[1]
      return load_data.getOurUniformInstances(n)
   
   raise ValueError(f"Unknown set: {set_name}")


def applyAlgorithm(instances, chosenAlgorithm):
   ratioScore = 0
   totalBins = 0
   totalOpt = 0
   totalTime = 0

   for instance in instances:
      # Decision to use time.perf_counter() was due to https://builtin.com/articles/timing-functions-python
      # time.time() is apparently not as precise and timeit is generally for small bits of code
      startTime = time.perf_counter()
      packing = chosenAlgorithm(instance["bin_capacity"], instance["weights"].copy())
      endTime = time.perf_counter()

      totalWeight = sum(packing["bin_weights"])
      if totalWeight - sum(instance["weights"]) != 0:
         raise Exception("Error: weight of instance differs to sum of bin weights")
      
      seen = set()
      for bin_pack in packing["packing"]:
            for idx in bin_pack:
               if idx in seen:
                   raise Exception("Item packed twice")
               seen.add(idx)
      if len(seen) != len(instance["weights"]):
            raise Exception("Missing items in packing")
      
      for binPack in packing["packing"]:
         binWeight = 0
         for index in binPack:
            binWeight += instance["weights"][index]
         if binWeight > instance["bin_capacity"]:
            print(binWeight)
            for index in binPack:
               print(instance["weights"][index])
            print(instance["bin_capacity"])
            raise Exception("bin capacity exceeded")

      totalTime += (endTime - startTime)

      opt = instance["optimal_solution"]
      alg = len(packing["bin_weights"])
      totalBins += alg
      if alg < opt:
         print(instance["bin_capacity"])
         print(packing["bin_weights"])
         print(packing["packing"])
         print(instance["optimal_solution"])
         raise Exception("Error: negative waste, our algorithm is cheating")
      totalOpt += opt

      ratioScore += (alg/opt)

   waste = totalBins - totalOpt
   print("Time: " + str(totalTime))
   print("Bins used: " + str(totalBins))
   print("Optimal number of bins: " + str(totalOpt))
   print("Waste (Excess bins): " + str(waste))
   print("Ratio of bins used by algorithm to bins used in optimal: " +  str((waste + totalOpt)/totalOpt))
   print("Number of instances used: " + str(len(instances)))
   print("Average ratio of bins used by algorithm to bins used in optimal case: " + str(ratioScore/len(instances)))



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
      prefix = f"{"our_u"}_{args.items}"

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
      applyAlgorithm(chosenInstances, chosenAlgorithm)
      return

   # default
   applyAlgorithm(instances, chosenAlgorithm)

if __name__ == "__main__":
   main()


