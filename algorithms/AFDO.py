from . import helpers
import random
import math
import pickle

def calcFitness(population, fitness):
    maxFitnessIndex = 0

    for x in range(0, len(population)):
        curFitness = 0
        for weight in population[x]["bin_weights"]:
            curFitness += (weight * weight)
        fitness.append(curFitness)
        if curFitness >= fitness[maxFitnessIndex]:
            maxFitnessIndex = x
    return maxFitnessIndex

def adaptiveFDO(binCapacity, weights):
    populationSize = 20
    population = []

    numItems = len(weights)
    for _ in range(0, populationSize):
        population.append(helpers.firstFitWithContainingBin(binCapacity, weights, False))

    wf = 0
    fitness = []
    maxFitnessIndex = calcFitness(population, fitness)
    maxFitness = fitness[maxFitnessIndex]
    
    bestSolution = pickle.loads(pickle.dumps(population[maxFitnessIndex], -1)) 
    newBestSolution = []
 
    iteration = 0
    maxIterations = 20
    while iteration < maxIterations:
        
        newMaxIndex = maxFitnessIndex

        fws = []
        for x in range(0, len(population)):
            fw = (fitness[x]/fitness[maxFitnessIndex]) - wf
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

                for x in range(0, tasks):
                    if items:
                        itemIndex = random.choice(items)
                        items.remove(itemIndex)
                
                        # Select a random bin to move each to (different to current bin)
                        curBinIndex = currentSolution["containing_bin"][itemIndex]
                        chosenBinIndex = random.randint(0, len(currentSolution["bin_weights"] - 1))
                        while chosenBinIndex == curBinIndex:
                            chosenBinIndex = random.randint(0, len(currentSolution["bin_weights"] - 1))

                        pace.append((itemIndex, curBinIndex, chosenBinIndex))
                
                # This is seperate for clarity sake, to make it clear what pace is here
                # It could easily be put in the above for loop
                for move in pace:
                    itemIndex = move[0]
                    itemWeight = weights[itemIndex]
                    # Weight of the bin the item is moving to
                    curBinIndex = move[1]
                    otherBinIndex = move[2]
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

                        fitness[x] += (newFitnessTerm - prevFitnessTerm)

                if fitness[x] > maxFitness:
                    maxFitness = fitness[x]
                    newBestSolution = currentSolution

            else:
                currentSolution = population[x]

                tasks = numItems * fw
                # Determine pace
                pace = []
                items = list(range(0, numItems))
                itemIndexesToMove = []
                x = 0
                while x < tasks:
                    if items:
                        itemIndex = random.choice(items)
                        items.remove(itemIndex)
                        itemIndexesToMove.append(itemIndex)

                        # Form the vectors describing the moves using the item's bins
                        curItemBin = currentSolution["containing_bin"][itemIndex]
                        itemBinInbest = bestSolution["containing_bin"][itemIndex]
                        if curItemBin != itemBinInbest:
                            pace.append((itemIndex, curItemBin, itemBinInbest))
                            x += 1
                    else:
                        break

                

                # First determine difference between current solution and best
                #pace=(Xi,t⊖X∗i,t)⊗fw
                # #Xi,t+1=Xi,t⊕pace
                # As in the if case, this could be done in the above loop
                # But this allows the idea of pace to be seen more clearly
                #for move in pace:


        #bestSolution = newBestSolution
