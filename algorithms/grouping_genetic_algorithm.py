import random
import math

def randomisedFirstFitEncoded(binCapacity, weights):
    binGroups = {}
    # First len(weights) indexes correspond to items
    # The value stored at the index corresponds to the bin the item is in
    encoding = []
    random.shuffle(weights)

    for i in range(0, len(weights)):
        packed = False

        for b in range(0, len(binGroups)):
            if sum(binGroups[b]) + weights[i] <= binCapacity:
                encoding.append(b)
                binGroups[b].append(i)
                packed = True
                break

        if not packed:
            encoding.append(len(binGroups))
            binGroups[len(binGroups)] = [i]


    return [encoding, binGroups]

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
    while i < 100 and best > lowerBound:
        i += 1

        # We use the same fitness function as in our simulated annealing implementation
        # The sum of the squares of each bin's weight
        fitness = {}
        for x in range(0, len(population)):
            fitness[x] = 0
            for g in range(itemChormosomeLen, len(population[x])):
                fitness[x] += binWeight * binWeight
            if fitness[x] < bestFitness:
                best = x
                bestFitness = fitness[x]

        newPopulation = []
        newPopulation.append(population[x]) # Save the best packing we've found so far

        #while len(newPopulation) < populationSize:
            # Select 2 from our population by

    print(population)
    return population[0]
