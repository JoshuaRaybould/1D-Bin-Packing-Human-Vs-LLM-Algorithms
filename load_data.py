from pathlib import Path
from pathlib import PurePosixPath
import csv

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
                # print("Skipping")
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
    instancesDir = Path("./Datasets/Randomly_Generated")
    randSolutions = Path("./Datasets/Solutions/RandomInstanceSolutions.csv")

    return loadInstances(instancesDir, randSolutions, True)

# Load Falkenauer U dataset of 80 instances
# The item sizes are uniformly distributed here
def getFalkenauer():
    instancesDir = Path("./Datasets/Falkenauer_U")
    falkSolutions = Path("./Datasets/Solutions/FalkenauerSolutions.csv")

    return loadInstances(instancesDir, falkSolutions, True)

# Load ANI or AI datasets. These are intentionally difficult instances.
def getHardInstances(version, solvedOnly):
    # True for AI, False for ANI
    if version:
        instancesDir = Path("./Datasets/Difficult_Instances/AI")
        AISolutions = Path("./Datasets/Solutions/AISolutions.csv")
        return loadInstances(instancesDir, AISolutions, solvedOnly)
    else:
        instancesDir = Path("./Datasets/Difficult_Instances/ANI")
        ANISolutions = Path("./Datasets/Solutions/ANISolutions.csv")
        return loadInstances(instancesDir, ANISolutions, solvedOnly)


