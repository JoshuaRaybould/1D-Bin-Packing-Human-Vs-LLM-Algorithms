import random
from . import helpers
import random
import pickle

totalTimeFitting = 0

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

def buildParentSets(population, elististSize):
    parentIndexes = {}
    parentIndexes["good"] = []
    parentIndexes["random"] = []
    parentsSelected = 0
    popSize = len(population)
    
    while parentsSelected < elististSize:
        goodParentIndex = random.randint(0, elististSize - 1)
        randomParentIndex = random.randint(elististSize, popSize - 1)
        while goodParentIndex == randomParentIndex:
            randomParentIndex = random.randint(elististSize, popSize - 1)
        
        parentIndexes["good"].append(goodParentIndex)
        parentIndexes["random"].append(randomParentIndex)
        parentsSelected += 1

    return parentIndexes

def rearrangeByPairs(child, unassignedItems, weights, binCapacity):
    # Randomise bin order
    binGroupsIndex = list(range(0, len(child["bin_groups"])))

    random.shuffle(binGroupsIndex)
    newUnassignedItems = []
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

                curBinLen = len(curBin)
                # Go through each pair in the bins until we can swap or have went through all pairs
                for i in range(0, curBinLen):
                    if foundMove:
                        break
                   
                    itemi = curBin[i]
                    itemiWeight = weights[itemi]

                    for k in range(i + 1, curBinLen):
                        itemk = curBin[k]
                        itemkWeight = weights[itemk]
                        sumWeightik = itemiWeight + itemkWeight
                        weightOnRemoval = (binWeight - sumWeightik)

                        if itemjWeight >= sumWeightik and weightOnRemoval + itemjWeight <= binCapacity:
                            newUnassignedItems.append(itemi)
                            newUnassignedItems.append(itemk)
                            curBin.remove(itemi)
                            curBin.remove(itemk)
                            curBin.append(itemj)
                            child["encoding"][itemj] = actualBinIndex
                            child["bin_weights"][actualBinIndex] = (weightOnRemoval) + itemjWeight
                            unassignedItems.pop(j)
                            foundMove = True
                            break      
                        elif sumWeightsToSwap >= sumWeightik and (weightOnRemoval) + sumWeightsToSwap <= binCapacity:
                            newUnassignedItems.append(itemi)
                            newUnassignedItems.append(itemk)
                            curBin.remove(itemi)
                            curBin.remove(itemk)
                            curBin.append(itemj)
                            curBin.append(itemx)
                            child["encoding"][itemj] = actualBinIndex
                            child["encoding"][itemx] = actualBinIndex
                            binWeights[actualBinIndex] = (weightOnRemoval) + sumWeightsToSwap
                            unassignedItems.pop(x)
                            unassignedItems.pop(j)
                            foundMove = True
                            break

        if not foundMove:
            j += 1

    for unassignedItem in unassignedItems:
        newUnassignedItems.append(unassignedItem)
    unassignedItems = newUnassignedItems

    for unassignedItem in unassignedItems:
        if weights[unassignedItem] >= binCapacity/2:
            unassignedItems.sort(key=lambda itemIndex: weights[itemIndex], reverse=True)
            break

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

                        
    return child

def fillGroup(parent1, parent2Groups, unassignedItems1):

    for groupIndex in parent2Groups:
        # We need to go through every single value in the groups and assign if unassigned, or change bin
        for item in parent2Groups[groupIndex]:
            itemCurrentGroup = parent1["encoding"][item]

            if itemCurrentGroup == -1:
                unassignedItems1.remove(item)
                parent1["encoding"][item] = groupIndex
            else:
                parent1["bin_groups"][itemCurrentGroup].remove(item)
                if not parent1["bin_groups"][itemCurrentGroup]:
                    parent1["bin_groups"].pop(itemCurrentGroup)
                parent1["encoding"][item] = groupIndex

        parent1["bin_groups"][groupIndex] = parent2Groups[groupIndex]

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

    return best


def groupingGeneticAlgorithm(binCapacity, weights):

    populationSize = 19
    elitistSize = 5
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
    populationAndFitness = sorted(zip(population, fitness), key=lambda popFit: popFit[1], reverse=True)
    population = [x for x, _ in populationAndFitness]
    fitness = [x for _, x in populationAndFitness]
    best = 0
    bestBins = len(population[best]["bin_groups"])

    i = 0
    lowerBound = helpers.getLowerBound(weights, binCapacity)
    
    while i < 45 and bestBins > lowerBound:

        i += 1
        parentIndexes = buildParentSets(population, elitistSize)
        curParentsIndex = 0

        newPopulation = []
        # Save the best packings we've found so far
        for x in range(0, elitistSize):
            newPopulation.append(population[x]) 
            
        while len(newPopulation) < elitistSize * 3:
            goodParentIndex = parentIndexes["good"][curParentsIndex]
            randomParentIndex = parentIndexes["random"][curParentsIndex]
            curParentsIndex += 1

            # Crossover
            # Crossover is expensive so we are going to do it 50% of the time
            prob = random.random()

            children = [0, 0]            

            if prob > 0.5:
                children[0] = crossover(population[goodParentIndex], population[randomParentIndex], weights, binCapacity)
                children[1] = crossover(population[randomParentIndex], population[goodParentIndex], weights, binCapacity)
            else:
                children = [population[goodParentIndex], population[randomParentIndex]]

            # Mutation
            child1 = mutate(children[0], weights, binCapacity)
            newPopulation.append(child1)

            if len(newPopulation) >= populationSize:
                break

            child2 = mutate(children[1], weights, binCapacity)
            newPopulation.append(child2)

        position = elitistSize
        while len(newPopulation) < populationSize:
            child = mutate(population[position], weights, binCapacity)
            newPopulation.append(child)
            position += 1

        population = newPopulation
        
        #
        fitness = []
        best = scoreFitnesses(population, fitness)

        #print(population)
        populationAndFitness = sorted(zip(population, fitness), key=lambda popFit: popFit[1], reverse=True)
        population = [x for x, _ in populationAndFitness]
        fitness = [x for _, x in populationAndFitness]
        best = 0
        bestBins = len(population[best]["bin_groups"])
  
    # Convert from encoding back to normal
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []
    # Iterate through all groups/bins in the best solution

    bestSolGroups = population[best]["bin_groups"]
    for groupIndex in bestSolGroups:
        bins["packing"].append([])
        bins["bin_weights"].append(0)
        for i in bestSolGroups[groupIndex]:
            bins["packing"][-1].append(i)
            bins["bin_weights"][-1] += weights[i]
 
    return bins
