import load_data
from algorithms import randomised_best_fit

done = False
while not done:
    done = True
    selection = int(input("Type a number 1-4 "))

    if selection == 1:
        instances = load_data.getRandomInstances(True)
    elif selection == 2:
        instances = load_data.getFalkenauer(True)
    elif selection == 3:
        instances = load_data.getHardInstances(True, False)
    elif selection == 4:
        instances = load_data.getHardInstances(False, False)
    else:
        done = False

ratioScore = 0
waste = 0
totalOpt = 0
for instance in instances:
    packing = randomised_best_fit.randomisedBestFit(instance["bin_capacity"], instance["weights"].copy())

    opt = instance["optimal_solution"]
    alg = len(packing)
    waste += alg - opt
    totalOpt += opt

    ratioScore += (alg/opt)

print(totalOpt)
print(waste)
print((waste + totalOpt)/totalOpt)
print(len(instances))
print(ratioScore)
print(ratioScore/len(instances))

