from . import helpers
import random
import math
import pickle
import time

def applyPace(currentSolution, weights, binCapacity, fitness, solIndex, pace):
    for move in pace:
        itemIndex = move[0]
        itemWeight = weights[itemIndex]
        # Weight of the bin the item is moving to
        curBinIndex = move[1]
        otherBinIndex = move[2]
        if otherBinIndex >= len(currentSolution["packing"]):
            return
        otherBinWeight = currentSolution["bin_weights"][otherBinIndex] 

        if otherBinWeight + itemWeight <= binCapacity:
 

            oldCurBinWeight = currentSolution["bin_weights"][curBinIndex]
            currentSolution["packing"][curBinIndex].remove(itemIndex)
            currentSolution["bin_weights"][curBinIndex] -= itemWeight
            newCurBinWeight = currentSolution["bin_weights"][curBinIndex]

            oldOtherBinWeight = currentSolution["bin_weights"][otherBinIndex]
            currentSolution["packing"][otherBinIndex].append(itemIndex)
            currentSolution["bin_weights"][otherBinIndex] += itemWeight
            currentSolution["containing_bin"][itemIndex] = otherBinIndex
            newOtherBinWeight = currentSolution["bin_weights"][otherBinIndex]

            prevFitnessTerm = oldCurBinWeight * oldCurBinWeight + oldOtherBinWeight * oldOtherBinWeight
            newFitnessTerm = newCurBinWeight * newCurBinWeight + newOtherBinWeight * newOtherBinWeight

            fitness[solIndex] += (newFitnessTerm - prevFitnessTerm)
    
    curBin = 0
    while curBin < len(currentSolution["packing"]):
        # If bin empty remove it
        if not currentSolution["packing"][curBin]:
            currentSolution["packing"].pop(curBin)
            currentSolution["bin_weights"].pop(curBin)
            for containingBin in currentSolution["containing_bin"]:
                if currentSolution["containing_bin"][containingBin] > curBin:
                    currentSolution["containing_bin"][containingBin] -= 1
        else:
            curBin += 1

def calcFitness(population, fitness):
    maxFitnessIndex = 0
    minFitnessIndex = 0

    for x in range(0, len(population)):
        curFitness = 0
        for weight in population[x]["bin_weights"]:
            curFitness += (weight * weight)
        fitness.append(curFitness)
        if curFitness >= fitness[maxFitnessIndex]:
            maxFitnessIndex = x
        if curFitness <= fitness[minFitnessIndex]:
            minFitnessIndex = x
    return (maxFitnessIndex, minFitnessIndex)

def adaptiveFDO(binCapacity, weights, timeLimit):
    start_time = time.time()
    timeBudget = 0.95 * timeLimit

    populationSize = 10
    population = []

    lowerBound = helpers.getLowerBound(weights, binCapacity)

    numItems = len(weights)
    for _ in range(0, populationSize):
        population.append(helpers.firstFitWithContainingBin(binCapacity, weights, False))

    wf = 0
    fitness = []
    (maxFitnessIndex, minFitnessIndex) = calcFitness(population, fitness)
    maxFitness = fitness[maxFitnessIndex]
    minFitness = fitness[minFitnessIndex]
    
    bestSolution = pickle.loads(pickle.dumps(population[maxFitnessIndex], -1)) 
    bestSolByLen = bestSolution
    bestLen = len(bestSolByLen["packing"])
    newBestSolution = []
    iteration = 0
    maxIterations = 150
    while iteration < maxIterations and len(bestSolution["packing"]) != lowerBound:
        elapsed = time.time() - start_time
        if elapsed >= timeBudget:
            break
 
        # Set worst solution to global best
        population[minFitnessIndex] = pickle.loads(pickle.dumps(bestSolution, -1))
        fitness[minFitnessIndex] = maxFitness
        minFitnessIndex = -1
        minFitness = float("infinity") # We will need to determine which is the minimum in our population after the iteration

        fws = []
        for x in range(0, len(population)):
            fw = (fitness[x]/maxFitness) - wf
            fws.append(fw)

        # Now move each bee
        for x in range(0, len(fws)):
            if fws[x] == 1 or fws[x] == 0:
                currentSolution = population[x]

                r = random.random()
                tasks = math.floor(r * numItems)
                # Determine pace
                pace = []
                items = list(range(0, numItems))

                for y in range(0, tasks):
                    if items:
                        itemIndex = random.choice(items)
                        items.remove(itemIndex)
                
                        # Select a random bin to move each to (different to current bin)
                        curBinIndex = currentSolution["containing_bin"][itemIndex]
                        chosenBinIndex = random.randint(0, len(currentSolution["bin_weights"]) - 1)
                        while chosenBinIndex == curBinIndex:
                            chosenBinIndex = random.randint(0, len(currentSolution["bin_weights"]) - 1)

                        pace.append((itemIndex, curBinIndex, chosenBinIndex))
                
                applyPace(currentSolution, weights, binCapacity, fitness, x, pace)

            else:
                currentSolution = population[x]

                tasks = numItems * fw
                # Determine pace
                pace = []
                items = list(range(0, numItems))
                i = 0
                while i < tasks:
                    if items:
                        itemIndex = random.choice(items)
                        items.remove(itemIndex)

                        # Form the vectors describing the moves using the item's bins
                        curItemBin = currentSolution["containing_bin"][itemIndex]
                        itemBinInbest = bestSolution["containing_bin"][itemIndex]
                        if curItemBin != itemBinInbest:
                            pace.append((itemIndex, curItemBin, itemBinInbest))
                            i += 1
                    else:
                        break
                
                applyPace(currentSolution, weights, binCapacity, fitness, x, pace)

            if len(currentSolution["packing"]) < bestLen:
                bestSolByLen = pickle.loads(pickle.dumps(currentSolution, -1)) 
                bestLen = len(bestSolByLen["packing"])
            if fitness[x] > maxFitness:
                maxFitness = fitness[x]
                newBestSolution = currentSolution
            elif fitness[x] < minFitness:
                minFitness = fitness[x]
                minFitnessIndex = x

        if newBestSolution:
            bestSolution = pickle.loads(pickle.dumps(newBestSolution, -1)) 
            newBestSolution = []
        iteration += 1
    
    return bestSolByLen
