import random
import math

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
        #print(binGroups)
        #print(encoding)
        if not packed:
            encoding[index] = len(binGroups)
            binGroups[len(binGroups)] = [index]

    #print("COOL")
    #for x in range(0, len(binGroups)):
        #print(binGroups[x])
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

def crossover(parent1, parent2):
    # 2 point crossover
    s = random.randint(0, len(parent1["bin_groups"]))
    e = random.randint(0, len(parent1["bin_groups"]))
    if e < s:
        tmp = s
        s = e
        e = tmp

    # We want to take the groups from index s to e in parent 1 and swap them with those groups in parent2

    return [parent1, parent2]

def firstFitDecreasing(child, unassignedItems, weights, binCapacity, removedLabel):
    itemWeightPairs = []
    for unassignedItem in unassignedItems:
        pair = [weights[unassignedItem], unassignedItem]
        itemWeightPairs.append(pair)
    print(itemWeightPairs)
    #print("unassigned items: ")
    #print(unassignedItems)
    #print("itemWeightPairs ")
    #print(itemWeightPairs)
    # https://stackoverflow.com/questions/25432182/sorting-an-array-of-arrays-in-python
    itemWeightPairs = sorted(itemWeightPairs, key=lambda x: x[0], reverse=True)
    #print(itemWeightPairs)

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
                print("We packed")
                packed = True
                break

        if not packed:
            print("We opened new bin")
            child["encoding"][itemIndex] = removedLabel
            child["bin_groups"][removedLabel] = [itemIndex]

    return child



def mutate(child, weights, binCapacity):
    unassignedItems = []

    mutateStartWeight = 0
    for groupIndex in child["bin_groups"]:
        for j in child["bin_groups"][groupIndex]:
            mutateStartWeight += weights[j]


    # Pick a bin at random to remove, we then run FFD to put the items back into groups
    b = random.randint(0, len(child["bin_groups"]))
    # print(b)
    # We eliminate the bth group, not necessarily corresponding to a group labelled b
    i = 0
    removedLabel = -1
    for groupIndex in child["bin_groups"]:
        if i == b:
            removedLabel = groupIndex
            #print("bingroups")
            #print(b)
            #print(child["bin_groups"])
            unassignedItems = child["bin_groups"].pop(groupIndex)
            #print(child["bin_groups"])
            #print("end")
            break
        i += 1

    child = firstFitDecreasing(child, unassignedItems, weights, binCapacity, removedLabel)

    mutateEndWeight = 0
    for groupIndex in child["bin_groups"]:
        for j in child["bin_groups"][groupIndex]:
            mutateEndWeight += weights[j]

    if mutateStartWeight != mutateEndWeight:
        print("WELLLLLL")
        print(unassignedItems)
        print(mutateStartWeight)
        print(mutateEndWeight)
        print(child)
        for item in unassignedItems:
            print(weights[item])

    return child



def groupingGeneticAlgorithm(binCapacity, weights):
    itemChormosomeLen = len(weights)
    populationSize = 20
    population = []

    # To generate our initial population we will apply first fit to a random order of weights
    for x in range(0, populationSize):
        population.append(randomisedFirstFitEncoded(binCapacity, weights))

    best = 0
    bestFitness = float("inf")
    i = 0
    lowerBound = math.ceil(sum(weights)/binCapacity)
    while i < 100 and bestFitness > lowerBound:
        i += 1
        #print("ok")
        # We use the same fitness function as in our simulated annealing implementation
        # The sum of the squares of each bin's weight
        fitness = {}
        for x in range(0, len(population)):
            fitness[x] = 0
            # Iterate through groups
            groups = population[x]["bin_groups"]
            for groupIndex in groups:
                binWeight = 0

                for item in groups[groupIndex]:
                    binWeight += weights[item]
                #print(population[x])
                #print(groups)
                #print("A Group")
                #print(g)
                #print()
                fitness[x] += binWeight * binWeight
            if fitness[x] > bestFitness:
                best = x
                bestFitness = fitness[x]

        newPopulation = []
        newPopulation.append(population[best]) # Save the best packing we've found so far

        while len(newPopulation) < populationSize:
            # Select 2 from our population by tournament selection
            parent1Index = tournamentSelect(population, fitness)
            parent2Index = tournamentSelect(population, fitness)

            # Crossover
            children = crossover(population[parent1Index].copy(), population[parent2Index].copy())

            # Mutation
            child1 = mutate(children[0], weights, binCapacity)
            child2 = mutate(children[1], weights, binCapacity)

            newPopulation.append(child1)
            newPopulation.append(child2)

    # Convert from encoding back to normal
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []
    # Iterate through all groups/bins in the best solution
    bestSolGroups = population[0]["bin_groups"]
    for groupIndex in bestSolGroups:
        bins["packing"].append([])
        bins["bin_weights"].append(0)
        for i in bestSolGroups[groupIndex]:
            bins["packing"][-1].append(weights[i])
            bins["bin_weights"][-1] += weights[i]
    """print(population[best]["bin_groups"])
    print("packing")
    print(bins["packing"])
    print("bin weights")
    print(bins["bin_weights"])
    print("capacity" +  str(binCapacity))"""

    return bins
