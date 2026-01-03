import math
import random
from . import first_fit

def shake(incumbentSolution, unmoved, binCapacity):
    moves = {}
    moves["single"], moves["swap"] = [], []

    binPos = -1
    while not moves["single"] and not moves["swap"]:
        selectedIndex = random.randint(0, len(unmoved) - 1)
        weight = unmoved[selectedIndex]
        
        # Check and add any possible single moves
        for x in range(0, len(incumbentSolution["bin_weights"])):
            if incumbentSolution["bin_weights"][x] + weight <= binCapacity:
                moves["single"].append(x)

        # Do the same for swaps
        # First locate the bin our value is in
        spaceWhenRemoved = -1
        for b in range(0, len(incumbentSolution["packing"])):
            if selectedIndex in incumbentSolution["packing"][b]:
                binPos = b
                spaceWhenRemoved = binCapacity - (incumbentSolution["bin_weights"][b] - weight)
                break

        # for x in range(len(unmoved), -1, -1):
            # if :


def variableNeighbourhoodSearch(binCapacity, weights, candidateSolution):
    incumbentSolution = candidateSolution
    
    kMax = 20
    iteration = 0
    totalIterations = 20
    while iteration < totalIterations:
        iteration += 1

        unmoved = weights.copy()
        for k in range(0, kMax + 1):
            newSol = shake(incumbentSolution, unmoved, binCapacity)
        



    return candidateSolution

def variableNeighbourhoodSearchFFD(binCapacity, weights):
    candidateSolution = first_fit.firstFit(binCapacity, weights, False)
    return variableNeighbourhoodSearch(binCapacity, weights, candidateSolution)


