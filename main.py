import load_data
import time
from algorithms import randomised_best_fit

done = False
while not done:
    done = True
    selection = int(input("Type a number 1-4 "))

    if selection == 1:
        instances = load_data.getRandomInstances()
    elif selection == 2:
        instances = load_data.getFalkenauer()
    elif selection == 3:
        instances = load_data.getHardInstances(True, True)
    elif selection == 4:
        instances = load_data.getHardInstances(False, False)
    else:
        done = False

ratioScore = 0
waste = 0
totalOpt = 0
totalTime = 0

for instance in instances:
    # Decision to use time.perf_counter() was due to https://builtin.com/articles/timing-functions-python
    # time.time() is apparently not as precise and timeit is generally for small bits of code
    startTime = time.perf_counter()
    packing = randomised_best_fit.randomisedBestFit(instance["bin_capacity"], instance["weights"].copy())
    endTime = time.perf_counter()

    totalTime += (endTime - startTime)

    opt = instance["optimal_solution"]
    alg = len(packing)
    waste += alg - opt
    totalOpt += opt

    ratioScore += (alg/opt)

print("Time: " + str(totalTime))
print("Total Bins used in optimal case: " + str(totalOpt))
print("Waste: " + str(waste))
print("Ratio of bins used by algorithm to bins used in optimal: " +  str((waste + totalOpt)/totalOpt))
print("Number of instances used: " + str(len(instances)))
print("Average ratio of bins used by algorithm to bins used in optimal case: " + str(ratioScore/len(instances)))


