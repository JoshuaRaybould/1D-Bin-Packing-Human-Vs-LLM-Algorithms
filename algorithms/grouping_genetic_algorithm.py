import random
from . import helpers
import random
import pickle
import time

totalTimeFitting = 0

def doesEncodingMakeSense(encoding, groupIndex):
    for x in range(0, len(encoding["encoding"])):
        itemGroup = encoding["encoding"][x]
        if itemGroup != -1 and x not in encoding["bin_groups"][itemGroup]:
            print("TROUBLE")
            print(groupIndex)
            print(encoding)
            break

def preAllocateItems(binCapacity, weights, binGroups, binWeights, encoding):
    binGroups[0] = [0]
    binWeights[0] = weights[0]
    curBin = 0
    encoding[0] = 0

    for x in range(1, len(weights)):
        if weights[x] + binWeights[curBin] > binCapacity:
            curBin += 1
            binGroups[curBin] = [x]
            binWeights[curBin] = weights[x]
            encoding[x] = curBin
        else:
            return x
        
    return len(weights)

def randomisedFirstFitEncoded(binCapacity, weights, binGroups, binWeights, encoding, numPreAllocated):
    # We don't directly shuffle the weights since they are used for every instance
    itemOrder = []
    # Only deal with items not already allocated
    for x in range(numPreAllocated, len(weights)):
        itemOrder.append(x)
    random.shuffle(itemOrder)

    for index in itemOrder:
        packed = False
        for b in range(0, len(binGroups)):

            if binWeights[b] + weights[index] <= binCapacity:
                binWeights[b] += weights[index]
                encoding[index] = b
                binGroups[b].append(index)
                packed = True
                break

        if not packed:
            groupIndex = len(binGroups)
            encoding[index] = groupIndex
            binGroups[groupIndex] = [index]
            binWeights[groupIndex] = weights[index]

    fullEncoding = {}
    fullEncoding["encoding"] = encoding
    fullEncoding["bin_groups"] = binGroups
    fullEncoding["bin_weights"] = binWeights
    fullEncoding["bin_order"] = []
    return fullEncoding

def tournamentSelect(population, fitness):
    tournamentSize = 2
    c1 = random.randint(0, len(population) - 1)
    for _ in range(1, tournamentSize):
        c2 = random.randint(0, len(population) - 1)
        if fitness[c2] > fitness[c1]:
            c1 = c2
    return c1

def getFirstUnusedGroupIndex(child):
    i = 0
    while True:
        if i not in child["bin_groups"]:
            return i
        i += 1

def rearrangeByPairs(child, unassignedItems, weights, binCapacity):
    # print("Rearrange start")

    # Randomise bin order
    binGroupsIndex = list(range(0, len(child["bin_groups"])))

    #print(len(child["bin_groups"]))
    #print(len(child["bin_order"]))
    #print(child["bin_groups"])
    #print(child["bin_order"])
    #print(binGroupsIndex)
    random.shuffle(binGroupsIndex)
    newUnassignedItems = []
    numNewBins = 0
    foundMove = False 
    
    j = 0
    while j < len(unassignedItems):
        foundMove = False
        for x in range(j + 1, len(unassignedItems)):

            if foundMove:
                break
            
            itemj = unassignedItems[j]
            itemx = unassignedItems[x]
            itemjWeight = weights[itemj]
            itemxWeight = weights[itemx]
            sumWeightsToSwap = itemjWeight + itemxWeight

            for binIndex in binGroupsIndex:

                if foundMove:
                    break

                actualBinIndex = child["bin_order"][binIndex]
                curBin = child["bin_groups"][actualBinIndex]
                binWeights = child["bin_weights"]
                binWeight = binWeights[actualBinIndex]
                if binWeight == binCapacity:
                    continue

                # Go through each pair in the bins until we can swap or have went through all pairs
                for i in range(0, len(curBin)):

                    if foundMove:
                        break

                    for k in range(i + 1, len(curBin)):
                        itemi = curBin[i]
                        itemk = curBin[k]
                        itemiWeight = weights[itemi]
                        itemkWeight = weights[itemk]

                        if itemjWeight >= itemiWeight + itemkWeight and (binWeight - (itemiWeight + itemkWeight)) + itemjWeight <= binCapacity:
                            # print("Enter if")
                            # print(unassignedItems)
                            # print(j)
                            newUnassignedItems.append(itemi)
                            newUnassignedItems.append(itemk)
                            curBin.remove(itemi)
                            curBin.remove(itemk)
                            curBin.append(itemj)
                            child["encoding"][itemj] = actualBinIndex

                           # print("weight and changes")
                            #print(binWeights[actualBinIndex])
        
                           # print(mutateStartWeight)
                            child["bin_weights"][actualBinIndex] = (binWeight - (itemiWeight + itemkWeight)) + itemjWeight
                            #print(binWeights[actualBinIndex])
                            #print(mutateStartWeight)
                            unassignedItems.pop(j)
                            foundMove = True
                            break                          
                        elif itemxWeight >= itemiWeight + itemkWeight and (binWeight - (itemiWeight + itemkWeight)) + itemxWeight <= binCapacity:
                            newUnassignedItems.append(itemi)
                            newUnassignedItems.append(itemk)
                            curBin.remove(itemi)
                            curBin.remove(itemk)
                            curBin.append(itemx)
                            child["encoding"][itemx] = actualBinIndex
                            binWeights[actualBinIndex] = (binWeight - (itemiWeight + itemkWeight)) + itemxWeight
                            unassignedItems.pop(x)
                            foundMove = True
                            break
                        elif sumWeightsToSwap >= itemiWeight + itemkWeight and (binWeight - (itemiWeight + itemkWeight)) + sumWeightsToSwap <= binCapacity:
                            newUnassignedItems.append(itemi)
                            newUnassignedItems.append(itemk)
                            curBin.remove(itemi)
                            curBin.remove(itemk)
                            curBin.append(itemj)
                            curBin.append(itemx)
                            child["encoding"][itemj] = actualBinIndex
                            child["encoding"][itemx] = actualBinIndex
                            binWeights[actualBinIndex] = (binWeight - (itemiWeight + itemkWeight)) + sumWeightsToSwap
                            unassignedItems.pop(x)
                            unassignedItems.pop(j)
                            foundMove = True
                            break
                    
        if not foundMove:
            j += 1
    
    # print("mid method")

    """print(mutateStartWeight)
    print(unassignedItems)
    print(newUnassignedItems)
    print("end mid method")"""
    for unassignedItem in unassignedItems:
        newUnassignedItems.append(unassignedItem)
    unassignedItems = newUnassignedItems

    for unassignedItem in unassignedItems:
        if weights[unassignedItem] >= binCapacity/2:
            unassignedItems.sort(key=lambda itemIndex: weights[itemIndex], reverse=True)
            break

    """print(unassignedItems)
    sumOfUnassigned = 0
    for item in unassignedItems:
        sumOfUnassigned += weights[item]
    print(sumOfUnassigned)
    print("If we add back unassigned we get")
    print(sumOfUnassigned + mutateStartWeight)"""
    
    for itemIndex in unassignedItems:
        itemWeight = weights[itemIndex]
        packed = False

        for groupIndex in child["bin_groups"]:
            binWeight = child["bin_weights"][groupIndex]

            if itemWeight + binWeight <= binCapacity:
                child["encoding"][itemIndex] = groupIndex
                child["bin_groups"][groupIndex].append(itemIndex)
                child["bin_weights"][groupIndex] += weights[itemIndex]
                packed = True
                break

        if not packed:
            label = len(child["bin_groups"])
            while label in child["bin_groups"]:
                label += 1
      
            child["encoding"][itemIndex] = label
            child["bin_groups"][label] = [itemIndex]
            child["bin_weights"][label] = weights[itemIndex]
            child["bin_order"].append(label)

            numNewBins += 1
                        
    return child

def fillGroup(parent1, parent2Groups, unassignedItems1):

    for groupIndex in parent2Groups:
        # We need to go through every single value in the groups and assign if unassigned, or change bin
        # doesEncodingMakeSense(parent1, -1)
        for item in parent2Groups[groupIndex]:
            itemCurrentGroup = parent1["encoding"][item]

            if itemCurrentGroup == -1:
                unassignedItems1.remove(item)
                parent1["encoding"][item] = groupIndex
            else:
                #if item not in parent1["bin_groups"][itemCurrentGroup]:
                   #print("The parent")
                   #print(parent1)

                parent1["bin_groups"][itemCurrentGroup].remove(item)
                if not parent1["bin_groups"][itemCurrentGroup]:
                    parent1["bin_groups"].pop(itemCurrentGroup)
                parent1["encoding"][item] = groupIndex

        parent1["bin_groups"][groupIndex] = parent2Groups[groupIndex]
        # doesEncodingMakeSense(parent1, groupIndex)

def addBinToChildSol(child, parentOrders, binNum):
    parent = parentOrders[0]
    parentBinOrder = parentOrders[1]
    parentBinLabel = parentBinOrder[binNum]

    # First check no items in the bin are already in our solution
    curBin = parent["bin_groups"][parentBinLabel]
    for item in curBin:
        if child["encoding"][item] != -1:
            return
    
    for item in curBin:
        child["encoding"][item] = binNum
    
    numBinGroups = len(child["bin_groups"])
    child["bin_groups"][numBinGroups] = curBin.copy()
    binWeight = parent["bin_weights"][parentBinLabel]
    child["bin_weights"][numBinGroups] = binWeight
    child["bin_order"].append(numBinGroups)


def crossover(parent1, parent2, weights, binCapacity):
    # ordered bins, gene level crossover
    # doesEncodingMakeSense(parent1, -10)
    # doesEncodingMakeSense(parent2, -10)

    child = {}
    child["encoding"] = [-1] * len(weights)
    child["bin_groups"] = {}
    child["bin_weights"] = {}
    child["bin_order"] = []

    p1BinOrder = parent1["bin_order"]
    p2BinOrder = parent2["bin_order"]

    p1BinLen = len(p1BinOrder)
    p2BinLen = len(p2BinOrder)

    iterations = min(p1BinLen, p2BinLen)

    for x in range(0, iterations):
        # We need to order the current bins by fullness
        orderedParents = [(parent2, p2BinOrder), (parent1, p1BinOrder)]

        # Determine which bin is more full
        if parent1["bin_weights"][p1BinOrder[x]] >= parent2["bin_weights"][p2BinOrder[x]]:
            orderedParents[0] = (parent1, p1BinOrder)
            orderedParents[1] = (parent2, p2BinOrder)
        
        for parentOrders in orderedParents:
            addBinToChildSol(child, parentOrders, x)
    
    while p1BinLen > iterations:
        addBinToChildSol(child, [parent1, p1BinOrder], iterations)
        iterations += 1
    
    while p2BinLen > iterations:
        addBinToChildSol(child, [parent2, p2BinOrder], iterations)
        iterations += 1

    # It is possible some items are not yet assigned, determine these and assign them
    unassignedItems = []
    for x in range(0, len(child["encoding"])):
        if child["encoding"][x] == -1:
            unassignedItems.append(x)
    
    if unassignedItems:
        rearrangeByPairs(child, unassignedItems, weights, binCapacity)

    return child

def mutate(child, weights, binCapacity):

    unassignedItems = []

    childBinOrder = child["bin_order"]
    #print(childBinOrder)

    emptiestBin = childBinOrder[-1]
    childBinOrder.pop(-1)
    # Always delete the emptiest bin
    binsToDelete = [emptiestBin] 

    numBinsToDelete = min(3, len(childBinOrder))

    for _ in range(1, numBinsToDelete):
        indexOfBinToDelete = random.randint(0, len(childBinOrder) - 1)
        binToDelete = childBinOrder.pop(indexOfBinToDelete)
        binsToDelete.append(binToDelete)
    
    unassignedItems = []

    # Remove the bins, we then run FFD to put the items back into groups
    for binToDelete in binsToDelete:
        curRemovedItems = child["bin_groups"].pop(binToDelete)
        child["bin_weights"].pop(binToDelete)
        for removedItem in curRemovedItems:
            unassignedItems.append(removedItem) 
    #print(unassignedItems)
    """    
    for groupIndex in child["bin_groups"]:
        if i == b:
            unassignedItems = child["bin_groups"].pop(groupIndex)
            break
        i += 1"""

    """print("START weight " + str(mutateStartWeight))
    preRearrangeWeight = 0
    for binIndex in child["bin_weights"]:
        preRearrangeWeight += child["bin_weights"][binIndex]
    print("pre weight " + str(preRearrangeWeight))"""
    rearrangeByPairs(child, unassignedItems, weights, binCapacity)

    return child

def scoreFitnesses(population, fitness):
    # We use the same fitness function as in our simulated annealing implementation
    # The sum of the squares of each bin's weight
    best = 0
    bestFitness = float("-inf")
    for x in range(0, len(population)):
        fitness.append(0)
        # Iterate through groups
        groups = population[x]["bin_groups"]
        binWeights = population[x]["bin_weights"]
        for groupIndex in groups:
            binWeight = binWeights[groupIndex]

            fitness[x] += binWeight * binWeight
        if fitness[x] > bestFitness:
            best = x
            bestFitness = fitness[x]
    """print("BEST")
    print(best)
    print(bestFitness)
    print("END")"""
    return best


def groupingGeneticAlgorithm(binCapacity, weights):

    populationSize = 20
    elitistSize = 6
    population = []

    weights.sort(reverse=True)

    binGroups = {}
    binWeights = {}
    # The value stored at the index corresponds to the bin the item is in
    encoding = [0] * len(weights)

    numPreAllocated = preAllocateItems(binCapacity, weights, binGroups, binWeights, encoding)

    # To generate our initial population we will apply first with with pre-allocated items
    # We first pack the large items, then have the remaining weights in random order and apply first fit
    for _ in range(0, populationSize):
        curBinGroups = pickle.loads(pickle.dumps(binGroups, -1)) 
        curBinWeights = binWeights.copy()
        curEncoding = encoding.copy()

        population.append(randomisedFirstFitEncoded(binCapacity, weights, curBinGroups, curBinWeights, curEncoding, numPreAllocated))

        # Get the bins in descending order - necessary for mutation and crossover
        descendingBinGroups = list(range(0, len(curBinGroups)))
        descendingBinGroups.sort(key=lambda group: curBinWeights[group], reverse=True)
        population[-1]["bin_order"] = descendingBinGroups

    # We use the same fitness function as in our simulated annealing implementation
    # The sum of the squares of each bin's weight
    fitness = []
    best = scoreFitnesses(population, fitness)
    bestFitness = fitness[best]
    #print(population)
    populationAndFitness = sorted(zip(population, fitness), key=lambda popFit: popFit[1], reverse=True)
    population = [x for x, _ in populationAndFitness]
    fitness = [x for _, x in populationAndFitness]
    best = 0
    bestBins = len(population[best]["bin_groups"])
    bestFitness = fitness[best]
    i = 0
    lowerBound = helpers.getLowerBound(weights, binCapacity)
    
    while i < 50 and bestBins > lowerBound:

        i += 1

        newPopulation = []
        # Save the best packings we've found so far
        for x in range(0, elitistSize):
            newPopulation.append(population[x]) 
            
        while len(newPopulation) < populationSize:
            # Select 2 from our population by tournament selection
            parent1Index = tournamentSelect(population, fitness)
            parent2Index = tournamentSelect(population, fitness)
            # Crossover
            # Crossover is expensive so we are going to do it 10% of the time
            prob = random.random()

            children = [0, 0]            

            if parent1Index != parent2Index and prob > 0.5:
                children[0] = crossover(population[parent1Index], population[parent2Index], weights, binCapacity)
                children[1] = crossover(population[parent2Index], population[parent1Index], weights, binCapacity)

            else:
                children = [population[parent1Index], population[parent2Index]]

            # Mutation
            child1 = mutate(children[0], weights, binCapacity)
            newPopulation.append(child1)

            if len(newPopulation) >= populationSize:
                break

            child2 = mutate(children[1], weights, binCapacity)
            newPopulation.append(child2)

        population = newPopulation
        
        #
        fitness = []
        best = scoreFitnesses(population, fitness)
        if bestFitness != fitness[best]:
            print("progress")
        bestFitness = fitness[best]
        #print(population)
        populationAndFitness = sorted(zip(population, fitness), key=lambda popFit: popFit[1], reverse=True)
        population = [x for x, _ in populationAndFitness]
        fitness = [x for _, x in populationAndFitness]
        best = 0
        bestBins = len(population[best]["bin_groups"])
        bestFitness = fitness[best]
        #
  
    # Convert from encoding back to normal
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []
    # Iterate through all groups/bins in the best solution
    #best = scoreFitnesses(weights, population, fitness)
    bestSolGroups = population[best]["bin_groups"]
    for groupIndex in bestSolGroups:
        bins["packing"].append([])
        bins["bin_weights"].append(0)
        for i in bestSolGroups[groupIndex]:
            bins["packing"][-1].append(weights[i])
            bins["bin_weights"][-1] += weights[i]
    
    #print(totalTimeFitting)

    return bins
