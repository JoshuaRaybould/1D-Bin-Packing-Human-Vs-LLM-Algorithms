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
    global totalTimeFitting
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

def crossover(parent1, parent2, weights, binCapacity):
    # 2 point crossover
    # doesEncodingMakeSense(parent1, -10)
    # doesEncodingMakeSense(parent2, -10)

    s = random.randint(0, len(parent1["bin_groups"]) - 1)
    e = random.randint(0, len(parent1["bin_groups"]) - 1)
    if e < s:
        tmp = s
        s = e
        e = tmp
    prefE = s + random.randint(0, 4)
    e = min(e, prefE)

    # We want to take the groups from index s to e in parent 1 and swap them with those groups in parent2
    # Note, doesn't necessarily correspond to groups labelled s, s+1 ... e
    parent1Groups = {}
    parent2Groups = {}
    groupsToCheck = []

    # We also need to unassign every item which are in these groups for each
    unassignedItems1 = []
    unassignedItems2 = []
    i = 0

    for groupIndex in parent1["bin_groups"]:
        if i >=s and i <= e:
            groupsToCheck.append(groupIndex)
            parent1Groups[groupIndex] = parent1["bin_groups"][groupIndex].copy()

            for item in parent1Groups[groupIndex]:
                unassignedItems1.append(item)
                parent1["encoding"][item] = -1 # Stands for unassigned
        i += 1

    for groupIndex in groupsToCheck:
        parent1["bin_groups"].pop(groupIndex)
        if groupIndex in parent2["bin_groups"]:
            parent2Groups[groupIndex] = parent2["bin_groups"][groupIndex].copy()

            for item in parent2Groups[groupIndex]:
                unassignedItems2.append(item)
                parent2["encoding"][item] = -1 # Stands for unassigned
            parent2["bin_groups"].pop(groupIndex)


    # Fill in parent 1 with the groups from parent 2, then assign unassigned items
    fillGroup(parent1, parent2Groups, unassignedItems1)

    firstFitDecreasing(parent1, unassignedItems1, weights, binCapacity)

    # Do the same for parent2
    fillGroup(parent2, parent1Groups, unassignedItems2)
    firstFitDecreasing(parent2, unassignedItems2, weights, binCapacity)


    return [parent1, parent2]

def mutate(child, weights, binCapacity):

    unassignedItems = []

    mutateStartWeight = 0
    for binIndex in child["bin_weights"]:
        mutateStartWeight += child["bin_weights"][binIndex]

    """for groupIndex in child["bin_groups"]:
        for j in child["bin_groups"][groupIndex]:
            mutateStartWeight += weights[j]"""
    
    childBinOrder = child["bin_order"]
    #print(childBinOrder)

    emptiestBin = childBinOrder[-1]
    childBinOrder.pop(-1)
    # Always delete the emptiest bin
    binsToDelete = [emptiestBin] 

    numBinsToDelete = min(2, len(childBinOrder))

    for _ in range(1, numBinsToDelete):
        indexOfBinToDelete = random.randint(0, len(childBinOrder) - 1)
        binToDelete = childBinOrder.pop(indexOfBinToDelete)
        binsToDelete.append(binToDelete)
    
    unassignedItems = []
    descending = False
    # Remove the bins, we then run FFD to put the items back into groups
    for binToDelete in binsToDelete:
        curRemovedItems = child["bin_groups"].pop(binToDelete)
        child["bin_weights"].pop(binToDelete)
        for removedItem in curRemovedItems:
            if weights[removedItem] >= binCapacity/2:
                descending = True
            unassignedItems.append(removedItem) 
    #print(unassignedItems)
    """    
    for groupIndex in child["bin_groups"]:
        if i == b:
            unassignedItems = child["bin_groups"].pop(groupIndex)
            break
        i += 1"""
    
    if descending:
        unassignedItems.sort(key=lambda itemIndex: weights[itemIndex], reverse=True)
    else:
        random.shuffle(unassignedItems)

    """print("START weight " + str(mutateStartWeight))
    preRearrangeWeight = 0
    for binIndex in child["bin_weights"]:
        preRearrangeWeight += child["bin_weights"][binIndex]
    print("pre weight " + str(preRearrangeWeight))"""
    rearrangeByPairs(child, unassignedItems, weights, binCapacity)

    mutateEndWeight = 0
    for binIndex in child["bin_weights"]:
        mutateEndWeight += child["bin_weights"][binIndex]

    """for groupIndex in child["bin_groups"]:
        for j in child["bin_groups"][groupIndex]:
            mutateEndWeight += weights[j]"""

    if mutateStartWeight != mutateEndWeight:
        print("ERROR: weight changed during mutation")
        print("START weight " + str(mutateStartWeight))
        print("END weight " + str(mutateEndWeight))

    return child

def scoreFitnesses(population, fitness):
    # We use the same fitness function as in our simulated annealing implementation
    # The sum of the squares of each bin's weight
    best = 0
    bestFitness = float("-inf")
    for x in range(0, len(population)):
        fitness[x] = 0
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
    elitistSize = 2
    population = []
    groupOrderings = []

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
    fitness = {}
    best = scoreFitnesses(population, fitness)
    print(fitness)
    bestFitness = fitness[best]
    print(fitness[best])
    bestBins = len(population[best]["bin_groups"])

    i = 0
    lowerBound = helpers.getLowerBound(weights, binCapacity)
    
    while i < 90 and bestBins > lowerBound:

        #print(bestFitness)
        i += 1

        newPopulation = []
        newPopulation.append(population[best]) # Save the best packing we've found so far
        while len(newPopulation) < populationSize:
            # Select 2 from our population by tournament selection
            parent1Index = tournamentSelect(population, fitness)
            parent2Index = tournamentSelect(population, fitness)
            # Crossover
            # Crossover is expensive so we are going to do it 10% of the time
            prob = random.random()

            
            parentCopy1 = pickle.loads(pickle.dumps(population[parent1Index], -1))
            parentCopy2 = pickle.loads(pickle.dumps(population[parent2Index], -1))

            #if parent1Index != parent2Index and prob > 0.9:
                # https://stackoverflow.com/questions/24756712/deepcopy-is-extremely-slow
                #children = crossover(parentCopy1, parentCopy2, child1BinOrder, child2BinOrder, weights, binCapacity)
            #else:
            children = [parentCopy1, parentCopy2]

            # Mutation
            child1 = mutate(children[0], weights, binCapacity)
            newPopulation.append(child1)

            if len(newPopulation) >= populationSize:
                break

            child2 = mutate(children[1], weights, binCapacity)
            newPopulation.append(child2)

        population = newPopulation
        
        fitness = {}
        best = scoreFitnesses(population, fitness)
        if bestFitness != fitness[best]:
            print("progress")
        #print(fitness)
        bestFitness = fitness[best]
        #print(fitness[best])
  
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
