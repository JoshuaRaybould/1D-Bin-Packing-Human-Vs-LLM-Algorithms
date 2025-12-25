import math
import random
from . import tabu_search

def everyAlphaUsed(averageCosts):
    for averageCost in averageCosts:
        if averageCost[1] == 0:
            return False
    return True

def reactiveGRASP(binCapacity, weights):

    alphaVals = [0.2, 0.4, 0.6, 0.8]
    probabilities = []
    averageCosts = []
    initialProbability = 1/len(alphaVals)
    for _ in alphaVals:
        probabilities.append(initialProbability)
        # 0 average quality, 0 uses of this alpha value
        averageCosts.append([0, 0]) 

    # We can use the lower bound as a way to check if we have arrived at the ideal solution (though it may not be achievable)
    lowerBound = math.ceil(sum(weights)/binCapacity)
    iteration = 0

    bestSolution = []
    bestSolutionBins = float("infinity") 

    weights.sort(reverse=True) # Saves time finding best and worst quality items
    while iteration < 500:
        iteration += 1
        
        toPack = weights.copy()
        candidateSolution = {}
        candidateSolution["packing"], candidateSolution["bin_weights"] = [], []

        alphaIndex = -1
        curProbability = random.random()
        for x in range(0, len(probabilities)):
            if curProbability <= probabilities[x]:
                alphaIndex = x
                break
            curProbability -= probabilities[x]

        while toPack:

            # Quality of an item is its weight, we first find the best and worst item based on this
            # Note the best item must actually fit in the bin
            bestItem = 0
            worstItem = -1

            if candidateSolution["packing"] and candidateSolution["bin_weights"][-1] + toPack[-1] <= binCapacity:
                # There is still space in the last opened bin
                while candidateSolution["bin_weights"][-1] + toPack[bestItem] > binCapacity:
                    bestItem += 1
            else:
                # Open new bin
                candidateSolution["packing"].append([])
                candidateSolution["bin_weights"].append(0)

            minQuality = toPack[worstItem]
            maxQuality = toPack[bestItem]

            alpha = alphaVals[alphaIndex]
            thresholdVal = minQuality + (1 - alpha) * (maxQuality - minQuality)

            worstAllowedIndex = bestItem
            for x in range(bestItem, len(toPack)):
                if toPack[x] >= thresholdVal:
                    worstAllowedIndex = x
                else:
                    break
            
            # We uniformly select from indexes between 0 and worstAllowedIndex
            packIndex = random.randint(bestItem, worstAllowedIndex)

            candidateSolution["packing"][-1].append(toPack[packIndex])
            candidateSolution["bin_weights"][-1] += toPack[packIndex]
            toPack.pop(packIndex)

        solCost = len(candidateSolution["packing"])
        # Get the sum of the qualities of the solutions given by the specific alpha index
        sumOfScores = averageCosts[alphaIndex][0] * averageCosts[alphaIndex][1] + solCost
        averageCosts[alphaIndex][1] += 1
        averageCosts[alphaIndex][0] = sumOfScores/ (averageCosts[alphaIndex][1])
        
        if iteration % 5  == 0 and everyAlphaUsed(averageCosts):
            qualitities = []
            totalQuality = 0
            
            for x in range(0, len(averageCosts)):
                quality = len(weights) / averageCosts[x][0]
                qualitities.append(quality)
                totalQuality += quality
            print(averageCosts)
            
            for x in range(0, len(probabilities)):
                probabilities[x] = qualitities[x] / totalQuality

        if not bestSolution or solCost < bestSolutionBins:
            bestSolution = candidateSolution
            bestSolutionBins = solCost
            
            if bestSolutionBins == lowerBound:
                return bestSolution
    print(probabilities)
    return bestSolution
