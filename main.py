import load_data
from algorithms import randomised_best_fit

instances = load_data.getRandomInstances()

ratioScore = 0
for instance in instances:
   packing = randomised_best_fit.randomisedBestFit(instance["bin_capacity"], instance["weights"].copy())

    opt = instance["optimal_solution"]
    alg = len(packing)

    ratioScore += (alg/opt)

print(ratioScore)
print(ratioScore/len(instances))

