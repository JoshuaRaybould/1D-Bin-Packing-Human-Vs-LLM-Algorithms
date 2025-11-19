from utilities import load_data
from utilities import test_correctness
import time
from algorithms import randomised_best_fit, simulated_annealing, grouping_genetic_algorithm

algorithmOptions = [randomised_best_fit.randomisedBestFit, simulated_annealing.simulatedAnnealingFF, simulated_annealing.simulatedAnnealingFFD, grouping_genetic_algorithm.groupingGeneticAlgorithm]
chosenAlgorithm = algorithmOptions[3]

test = False
done = False
while not done:
    done = True
    selection = int(input("Type a number 1-7: "))

    if selection == 1:
        instances = load_data.getRandomInstances()
    elif selection == 2:
        instances = load_data.getFalkenauer()
    elif selection == 3:
        instances = load_data.getHardInstances(True, True)
    elif selection == 4:
        instances = load_data.getHardInstances(False, False)
    elif selection == 5:
        # arguments here correspond to: number of instances, capacity, number of items, distribution type
        instances = load_data.getOurRandomInstances(20, 100, 100, "n")
    elif selection == 6:
        instances = load_data.getOurRandomInstances(15, 100, 100, "u")
    elif selection == 7:
        instances = load_data.getRandomInstances()
        test = True
    else:
        done = False


if test:
    if test_correctness.testAlgorithmCorrectness(chosenAlgorithm, instances):
        print("Seems to be correct")
else:
    ratioScore = 0
    waste = 0
    totalOpt = 0
    totalTime = 0

    for instance in instances:
        # Decision to use time.perf_counter() was due to https://builtin.com/articles/timing-functions-python
        # time.time() is apparently not as precise and timeit is generally for small bits of code
        startTime = time.perf_counter()
        packing = chosenAlgorithm(instance["bin_capacity"], instance["weights"].copy())
        endTime = time.perf_counter()

        # print(instance["weights"])
        # print(packing)
        # print()

        totalWeight = sum(packing["bin_weights"])
        if totalWeight - sum(instance["weights"]) != 0:
            raise Exception("Error: weight of instance differs to that of packing")

        totalTime += (endTime - startTime)

        opt = instance["optimal_solution"]
        alg = len(packing["bin_weights"])
        waste += alg - opt
        if waste < 0:
            print(instance["bin_capacity"])
            print(packing["bin_weights"])
            print(packing["packing"])
            print(instance["optimal_solution"])
            #print(instance[])
            raise Exception("Error: negative waste, our algorithm is cheating")
        totalOpt += opt

        ratioScore += (alg/opt)

    print("Time: " + str(totalTime))
    print("Total Bins used in optimal case: " + str(totalOpt))
    print("Waste (Excess bins): " + str(waste))
    print("Ratio of bins used by algorithm to bins used in optimal: " +  str((waste + totalOpt)/totalOpt))
    print("Number of instances used: " + str(len(instances)))
    print("Average ratio of bins used by algorithm to bins used in optimal case: " + str(ratioScore/len(instances)))


