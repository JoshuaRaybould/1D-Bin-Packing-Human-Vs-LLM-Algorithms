import random
import math
import random
import copy

def doesEncodingMakeSense(encoding, groupIndex):
    for x in range(0, len(encoding["encoding"])):
        itemGroup = encoding["encoding"][x]
        if itemGroup != -1 and x not in encoding["bin_groups"][itemGroup]:
            print("TROUBLE")
            print(groupIndex)
            print(encoding)
            break

def randomisedFirstFitEncoded(binCapacity, weights):
    binGroups = {}
    # First len(weights) indexes correspond to items
    # The value stored at the index corresponds to the bin the item is in
    encoding = [0] * len(weights)
    # We can't directly shuffle the weights since they are used for every instance
    itemOrder = []
    for x in range(0, len(weights)):
        itemOrder.append(x)
    random.shuffle(itemOrder)

    for index in itemOrder:
        packed = False
        for b in range(0, len(binGroups)):
            binWeight = 0
            for j in binGroups[b]:
                binWeight += weights[j]

            if binWeight + weights[index] <= binCapacity:
                encoding[index] = b
                binGroups[b].append(index)
                packed = True
                break

        if not packed:
            encoding[index] = len(binGroups)
            binGroups[len(binGroups)] = [index]
    fullEncoding = {}
    fullEncoding["encoding"] = encoding
    fullEncoding["bin_groups"] = binGroups
    return fullEncoding

def tournamentSelect(population, fitness):
    tournamentSize = 2
    c1 = random.randint(0, len(population) - 1)
    for x in range(1, tournamentSize):
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

def firstFitDecreasing(child, unassignedItems, weights, binCapacity):
    itemWeightPairs = []
    for unassignedItem in unassignedItems:
        pair = [weights[unassignedItem], unassignedItem]
        itemWeightPairs.append(pair)

    # https://stackoverflow.com/questions/25432182/sorting-an-array-of-arrays-in-python
    itemWeightPairs = sorted(itemWeightPairs, key=lambda x: x[0], reverse=True)

    for i in range(0, len(itemWeightPairs)):
        packed = False
        itemIndex = itemWeightPairs[i][1]

        for groupIndex in child["bin_groups"]:
            binWeight = 0
            for j in child["bin_groups"][groupIndex]:
                binWeight += weights[j]

            if binWeight + weights[itemIndex] <= binCapacity:
                child["encoding"][itemIndex] = groupIndex
                child["bin_groups"][groupIndex].append(itemIndex)
                packed = True
                break

        if not packed:
            label = getFirstUnusedGroupIndex(child)
            child["encoding"][itemIndex] = label
            child["bin_groups"][label] = [itemIndex]

    return child

def fillGroup(parent1, parent2Groups, unassignedItems1):

    for groupIndex in parent2Groups:
        # We need to go through every single value in the groups and assign if unassigned, or change bin
        doesEncodingMakeSense(parent1, -1)
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
        doesEncodingMakeSense(parent1, groupIndex)

def crossover(parent1, parent2, weights, binCapacity):
    # 2 point crossover
    doesEncodingMakeSense(parent1, -10)
    doesEncodingMakeSense(parent2, -10)

    s = random.randint(0, len(parent1["bin_groups"]))
    e = random.randint(0, len(parent1["bin_groups"]))
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
    for groupIndex in child["bin_groups"]:
        for j in child["bin_groups"][groupIndex]:
            mutateStartWeight += weights[j]


    # Pick a bin at random to remove, we then run FFD to put the items back into groups
    b = random.randint(0, len(child["bin_groups"]))

    # We eliminate the bth group, not necessarily corresponding to a group labelled b
    i = 0
    for groupIndex in child["bin_groups"]:
        if i == b:
            unassignedItems = child["bin_groups"].pop(groupIndex)
            break
        i += 1

    child = firstFitDecreasing(child, unassignedItems, weights, binCapacity)

    mutateEndWeight = 0
    for groupIndex in child["bin_groups"]:
        for j in child["bin_groups"][groupIndex]:
            mutateEndWeight += weights[j]

    return child

def scoreFitnesses(weights, population, fitness):
   # We use the same fitness function as in our simulated annealing implementation
   # The sum of the squares of each bin's weight
   best = 0
   bestFitness = float("inf")
   for x in range(0, len(population)):
      fitness[x] = 0
      # Iterate through groups
      groups = population[x]["bin_groups"]
      for groupIndex in groups:
            binWeight = 0

            for item in groups[groupIndex]:
               binWeight += weights[item]

            fitness[x] += binWeight * binWeight
      if fitness[x] > bestFitness:
            best = x
            bestFitness = fitness[x]

   return best


def groupingGeneticAlgorithm(binCapacity, weights):
    populationSize = 20
    population = []

    # To generate our initial population we will apply first fit to a random order of weights
    for x in range(0, populationSize):
        population.append(randomisedFirstFitEncoded(binCapacity, weights))

    best = 0
    bestFitness = float("inf")
    i = 0
    lowerBound = math.ceil(sum(weights)/binCapacity)
    while i < 25 and bestFitness > lowerBound:
        i += 1

        # We use the same fitness function as in our simulated annealing implementation
        # The sum of the squares of each bin's weight
        fitness = {}
        best = scoreFitnesses(weights, population, fitness)
        bestFitness = fitness[best]

        newPopulation = []
        newPopulation.append(population[best]) # Save the best packing we've found so far

        while len(newPopulation) < populationSize:
            # Select 2 from our population by tournament selection
            parent1Index = tournamentSelect(population, fitness)
            parent2Index = tournamentSelect(population, fitness)

            # Crossover
            # Crossover is expensive so we are going to do it 10% of the time
            prob = random.random()
            if parent1Index != parent2Index and prob > 0.9:
               children = crossover(copy.deepcopy(population[parent1Index]), copy.deepcopy(population[parent2Index]), weights, binCapacity)
            else:
                children = [copy.deepcopy(population[parent1Index]), copy.deepcopy(population[parent2Index])]
            # children = [population[parent1Index], population[parent2Index]]

            # Mutation
            child1 = mutate(children[0], weights, binCapacity)
            child2 = mutate(children[1], weights, binCapacity)

            newPopulation.append(child1)
            newPopulation.append(child2)

        population = newPopulation

    # Convert from encoding back to normal
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []
    # Iterate through all groups/bins in the best solution
    best = scoreFitnesses(weights, population, fitness)
    bestSolGroups = population[best]["bin_groups"]
    for groupIndex in bestSolGroups:
        bins["packing"].append([])
        bins["bin_weights"].append(0)
        for i in bestSolGroups[groupIndex]:
            bins["packing"][-1].append(weights[i])
            bins["bin_weights"][-1] += weights[i]

    return bins
