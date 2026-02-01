from pathlib import Path
from pathlib import PurePosixPath
from algorithms.helpers import getLowerBound
import numpy as np
import csv
import math

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

# Load Falkenauer U dataset of 80 instances total
# The item sizes are uniformly distributed here
def getFalkenauer(n):
    # n is number of items
    if n == 120:
        instancesDir = Path("./Instances/Falkenauer_U/Falkenauer_" + n)
    elif n == 250:
        instancesDir = Path("./Instances/Falkenauer_U/Falkenauer_" + n)
    elif n == 500:
        instancesDir = Path("./Instances/Falkenauer_U/Falkenauer_" + n)
    else:
        instancesDir = Path("./Instances/Falkenauer_U/Falkenauer_" + n)

    falkSolutions = Path("./Instances/Solutions/FalkenauerSolutions.csv")

    return loadInstances(instancesDir, falkSolutions, True)

# Load the dataset of Scholl instances
def getSchollInstances(n):
    # n is number of instances (10, 480, 720), the set of 10 being the hardest
    instancesDir = Path("./Instances/Scholl/Scholl_" + n) 
        
    schollSolutions = Path("./Instances/Solutions/SchollSolutions.csv")
    return loadInstances(instancesDir, schollSolutions, True)

# Load the Hard28 dataset
def getHardInstances():
    instancesDir = Path("./Instances/Hard28")
    hardSolutions = Path("./Instances/Solutions/Hard28Solutions.csv")
    return loadInstances(instancesDir, hardSolutions, True)

# Get some test instances we generated
def getTestInstances():
    instancesDir = Path("./my_instances/test_u")
    return loadNoSolInstances(instancesDir)

# Get uniform instances we generated
def getOurUniformInstances(n):
    instancesDir = Path("./my_instances/our_u_" + n)
    return loadNoSolInstances(instancesDir)

# Below are functions for generating a specific number instances with bins set to some capacity and number of items
# The downside is the optimal won't be known for any
# At least we can calculate a lower bound on the optimal number (given by Martello, no worse than 3/4 the optimal)
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
        minimumWeight = math.ceil((minMultiplier*capacity)/100)
        maxMultiplier = rng.integers(50, 71)
        maximumWeight = math.ceil((maxMultiplier*capacity)/100)

        weights = []
        if distribution == "n":
            weights = createNormalDistribution(minimumWeight, capacity, numItems, rng) # Normal distribution
        else:
            weights = rng.integers(low=minimumWeight, high=(maximumWeight+1), size=numItems) # Uniform distribution
        weights.sort()
        weights = (np.round(weights)).astype(int)
        instanceInfo["weights"] = weights.tolist()
        #totalWeight = sum(instanceInfo["weights"])

        instanceInfo["optimal_solution"] = getLowerBound(instanceInfo["weights"], instanceInfo["bin_capacity"]) #math.ceil(totalWeight/capacity)

        instances.append(instanceInfo)

    return instances


def writeInstanceTxt(instanceInfo, outFile: Path):
    """
    Write instances to file, in format:
      line 1: number_of_items
      line 2: bin_capacity
      remaining lines: item weights
    """
    outFile.parent.mkdir(parents=True, exist_ok=True)

    n = instanceInfo["number_of_items"]
    capacity = instanceInfo["bin_capacity"]
    weights = instanceInfo["weights"]

    if len(weights) != n:
        raise ValueError(f"number_of_items={n} but len(weights)={len(weights)}")

    with outFile.open("w") as f:
        f.write(str(n) + "\n")
        f.write(str(capacity) + "\n")
        for w in weights:
            f.write(str(int(w)) + "\n")


def saveInstancesAsTxt(instances, prefix):
    """
    Save instances as separate text files in outDir.
    Returns the output directory as a Path.
    """
    outDir = "my_instances/" + str(prefix)
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)

    for i, inst in enumerate(instances, start=1):

        filename = f"{prefix}_{i:03d}.txt"
        writeInstanceTxt(inst, outDir / filename)

    return outDir



