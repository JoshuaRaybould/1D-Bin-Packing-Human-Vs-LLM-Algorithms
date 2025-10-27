from pathlib import Path
from pathlib import PurePosixPath
import csv

# Load the dataset of Random Generated instances downloaded from BPPLIB
def getRandomInstances():
    instancesDir = Path("./Datasets/Randomly_Generated")
    randSolutions = Path("./Datasets/Solutions/Solutions.csv")

    sols = {}
    with randSolutions.open() as f:
        csvReader = csv.DictReader(f)
        for row in csvReader:
            if row["Status"] == "Solved":
                sols[row["Name"]] = row["Best LB"]



    # Instances will include the number of items, bin capacity, solution and the weight for each item
    instances = []
    for instanceFile in instancesDir.iterdir():
        if instanceFile.is_file():
            instanceInfo = {}
            instanceInfo["weights"] = []
            fileName = PurePosixPath(instanceFile).name
            instanceInfo["optimal_solution"] = sols[fileName]

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

