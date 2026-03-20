import random
from . import tabu_search
from . import helpers
import time

def everyAlphaUsed(averageCosts):
    for averageCost in averageCosts:
        if averageCost[1] == 0:
            return False
    return True

def reactiveGRASP(binCapacity, weights, timeLimit, useTimeLimit=False):
    if not useTimeLimit:
        timeLimit = 1000 # effectively unlimited
        maxIterations = 10
    else:
       maxIterations = float("inf") 

    start_time = time.time()
    timeBudget = 0.98 * timeLimit

    alphaVals = [0.05, 0.1, 0.15]
    probabilities = []
    learningRate = 0.995
    averageCosts = []
    initialProbability = 1/len(alphaVals)
    for _ in alphaVals:
        probabilities.append(initialProbability)
        # 0 average quality, 0 uses of this alpha value
        averageCosts.append([0, 0]) 

    # We can use the lower bound as a way to check if we have arrived at the ideal solution (though it may not be achievable)
    lowerBound = helpers.getLowerBound(weights, binCapacity)
    iteration = 0

    bestSolution = []
    bestSolutionBins = float("infinity") 

    indexes = []
    for x in range(0, len(weights)):
        indexes.append(x)
    indexes.sort(key=lambda itemIndex: weights[itemIndex], reverse=True) # Saves time finding best and worst quality items
    while iteration < maxIterations:
        elapsed = time.time() - start_time
        if elapsed >= timeBudget:
            break
        iteration += 1
        
        toPack = indexes.copy()
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

            smallestItem = toPack[-1]
            if candidateSolution["packing"] and candidateSolution["bin_weights"][-1] + weights[smallestItem] <= binCapacity:
                # There is still space in the last opened bin
                curCandidate = toPack[bestItem]
                while candidateSolution["bin_weights"][-1] + weights[curCandidate] > binCapacity:
                    bestItem += 1
                    curCandidate = toPack[bestItem]
            else:
                # Open new bin
                candidateSolution["packing"].append([])
                candidateSolution["bin_weights"].append(0)

            worstItemIndex = toPack[worstItem] # Its index in the weights array
            minQuality = weights[worstItemIndex]
            bestItemIndex = toPack[bestItem]
            maxQuality = weights[bestItemIndex]

            alpha = alphaVals[alphaIndex]
            thresholdVal = minQuality + (1 - alpha) * (maxQuality - minQuality)

            worstAllowedIndex = bestItem
            for x in range(bestItem, len(toPack)):
                itemx = toPack[x]
                if weights[itemx] >= thresholdVal:
                    worstAllowedIndex = x
                else:
                    break
            
            # We uniformly select from indexes between 0 and worstAllowedIndex
            packIndex = random.randint(bestItem, worstAllowedIndex)

            candidateSolution["packing"][-1].append(toPack[packIndex])
            candidateSolution["bin_weights"][-1] += weights[toPack[packIndex]]
            toPack.pop(packIndex)
        
        elapsed = time.time() - start_time
        if elapsed >= timeBudget:
            break

        candidateSolution = tabu_search.tabuSearch(binCapacity, weights, candidateSolution, True, timeBudget - elapsed, True)

        solCost = len(candidateSolution["packing"])
        
        # Get the sum of the qualities of the solutions given by the specific alpha index
        sumOfScores = averageCosts[alphaIndex][0] * averageCosts[alphaIndex][1] + solCost
        averageCosts[alphaIndex][1] += 1
        averageCosts[alphaIndex][0] = sumOfScores/ (averageCosts[alphaIndex][1])
        
        if iteration % 2  == 0 and everyAlphaUsed(averageCosts):
            qualitities = []
            totalQuality = 0

            smallestCost = float("infinity")
            for cost in averageCosts:
                if cost[0] < smallestCost:
                    smallestCost = cost[0]
            
            learningParameter = learningRate * smallestCost

            for x in range(0, len(averageCosts)):
                quality = len(weights) / (averageCosts[x][0] - learningParameter)
                qualitities.append(quality)
                totalQuality += quality
            
            for x in range(0, len(probabilities)):
                probabilities[x] = qualitities[x] / totalQuality

        if not bestSolution or solCost < bestSolutionBins:
            bestSolution = candidateSolution
            bestSolutionBins = solCost
            
            if bestSolutionBins == lowerBound:
                return bestSolution
    
    return bestSolution
