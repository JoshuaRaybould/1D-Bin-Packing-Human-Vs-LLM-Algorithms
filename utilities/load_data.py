from pathlib import Path
from pathlib import PurePosixPath
from algorithms.helpers import getLowerBound
import numpy as np
import csv

def loadNoSolInstances(instancesDir):

    # Instances will include the number of items, bin capacity, solution and the weight for each item
    instances = []
    for instanceFile in instancesDir.iterdir():

        if instanceFile.is_file():
            fileName = PurePosixPath(instanceFile).name
            instanceInfo = {}
            instanceInfo["weights"] = []
            instanceInfo["file_name"] = fileName

            with instanceFile.open() as f:
                i = 0
                line = f.readline()
                while line:
                    val = int(line)
                    if i == 0:
                        instanceInfo["number_of_items"] = val
                    elif i == 1:
                        instanceInfo["bin_capacity"] = val
                    else:
                        instanceInfo["weights"].append(val)

                    line = f.readline()
                    i += 1
            instanceInfo["optimal_solution"] = getLowerBound(instanceInfo["weights"], instanceInfo["bin_capacity"])
            instances.append(instanceInfo)
    
    return instances

# Load instances from a specific given dataset
def loadInstances(instancesDir, solutions, solvedOnly):

    sols = {}
    with solutions.open() as f:
        csvReader = csv.DictReader(f)
        for row in csvReader:
            if solvedOnly:
                if row["Status"] == "Solved":
                    sols[row["Name"]] = row["Best LB"]
            else:
                sols[row["Name"]] = row["Best LB"]

    # Instances will include the number of items, bin capacity, solution and the weight for each item
    instances = []
    for instanceFile in instancesDir.iterdir():

        if instanceFile.is_file():
            fileName = PurePosixPath(instanceFile).name
            if fileName not in sols:
                continue
            instanceInfo = {}
            instanceInfo["weights"] = []
            instanceInfo["file_name"] = fileName

            instanceInfo["optimal_solution"] = int(sols[fileName])


            with instanceFile.open() as f:
                i = 0
                line = f.readline()
                while line:
                    val = int(line)
                    if i == 0:
                        instanceInfo["number_of_items"] = val
                    elif i == 1:
                        instanceInfo["bin_capacity"] = val
                    else:
                        instanceInfo["weights"].append(val)

                    line = f.readline()
                    i += 1
            instances.append(instanceInfo)
    
    return instances


# Each dataset being loaded in here is from BPPLIB
# The first 2 datasets here are completely solved, but the ANI and AI datasets are not
# An option has been included to choose to use just solved instances
# If we include all instances then in cases they aren't solved we simply use the lower bound on the number of bins

# Load the dataset of Random Generated instances
def getRandomInstances():
    instancesDir = Path("./Instances/Randomly_Generated")
    randSolutions = Path("./Instances/Solutions/RandomInstanceSolutions.csv")

    return loadInstances(instancesDir, randSolutions, True)

# Load Falkenauer U dataset of 80 instances
# The item sizes are uniformly distributed here
def getFalkenauer():
    instancesDir = Path("./Instances/Falkenauer_U")
    falkSolutions = Path("./Instances/Solutions/FalkenauerSolutions.csv")

    return loadInstances(instancesDir, falkSolutions, True)

# Load ANI or AI datasets. These are intentionally difficult instances.
def getHardInstances(version, solvedOnly):
    # True for AI, False for ANI
    if version:
        instancesDir = Path("./Instances/Difficult_Instances/AI")
        AISolutions = Path("./Instances/Solutions/AISolutions.csv")
        return loadInstances(instancesDir, AISolutions, solvedOnly)
    else:
        instancesDir = Path("./Instances/Difficult_Instances/ANI")
        ANISolutions = Path("./Instances/Solutions/ANISolutions.csv")
        return loadInstances(instancesDir, ANISolutions, solvedOnly)

def particleSwarmTest():
    instancesDir = Path("./Instances/BinPacking_OMP")
    return loadNoSolInstances(instancesDir)




# Below are functions for generating a specific number instances with bins set to some capacity and number of items
# The downside is the optimal won't be known for any
# At least we can calculate a simple lower bound by doing ceil(total_weight/capacity)
# The minimum possible weight takes inpiration from the randomly generated dataset from BPPLIB


def createNormalDistribution(minimum, maximum, numItems, rng):
    mean = (maximum + minimum)/2

    normalDist = rng.standard_normal(numItems)

    # We need to find the value with greatest magnitude so we can scale all values into our range accordingly
    maxVal = 0
    for val in normalDist:
        maxVal = max(maxVal, abs(val))

    # The value beyond the mean that 1 in the normalDist corresponds to
    scale = (maximum - mean)/maxVal

    for x in range(0, len(normalDist)):
        normalDist[x] = (normalDist[x] * scale) + mean

    return normalDist


# Random function information: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
def getOurRandomInstances(numInstances, capacity, numItems, distribution):

    instances = []

    rng = np.random.default_rng()
    for x in range(0, numInstances):
        instanceInfo = {}

        instanceInfo["number_of_items"] = numItems
        instanceInfo["bin_capacity"] = capacity

        minMultiplier = rng.integers(10, 31)
        minimumWeight = (minMultiplier*capacity)/100

        weights = []
        if distribution == "n":
            weights = createNormalDistribution(minimumWeight, capacity, numItems, rng) # Normal distribution
        else:
            weights = rng.integers(low=minimumWeight, high=(capacity+1), size=numItems) # Uniform distribution
        weights.sort()
        weights = (np.round(weights)).astype(int)
        instanceInfo["weights"] = weights.tolist()
        #totalWeight = sum(instanceInfo["weights"])

        instanceInfo["optimal_solution"] = getLowerBound(instanceInfo["weights"], instanceInfo["bin_capacity"]) #math.ceil(totalWeight/capacity)

        instances.append(instanceInfo)

    return instances


