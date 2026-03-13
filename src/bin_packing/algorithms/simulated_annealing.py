import math
import random
from . import helpers
import time

# Decreasing is to decide whether to use first fit with or without putting the data in decreasing order first
def simulatedAnnealing(binCapacity, weights, decreasing, timeLimit):
    start_time = time.time()
    timeBudget = 0.95 * timeLimit

    temperature = 50000
    cooling = 0.99998
    candidateSolution = helpers.firstFit(binCapacity, weights, decreasing)

    # We can use the lower bound as a way to check if we have arrived at the ideal solution (though it may not be achievable)
    lowerBound = helpers.getLowerBound(weights, binCapacity)
    iteration = 0
    while len(candidateSolution["bin_weights"]) > lowerBound and temperature > 0.01 and iteration < 1500000:
        elapsed = time.time() - start_time
        if elapsed >= timeBudget:
            break

        # It is only possible to improve the solution if there are at least 2 bins. This is guaranteed by the lowerBound condition in the loop (so no explicit check here)
        # Select at random 2 bins
        numBins = len(candidateSolution["bin_weights"])
        i = random.randint(0, numBins-1)
        j = random.randint(0, numBins-1)
        while i == j:
            j = random.randint(0, numBins-1)

        # Randomly decide the type of tweak to apply
        choice = random.randint(1,2)

      
        tweaked = False
        if choice == 1:
            # If possible, swap a pair of items in bin i and j
            iValToSwap = random.choice(candidateSolution["packing"][i])
            jValToSwap = random.choice(candidateSolution["packing"][j])
            iWeight = weights[iValToSwap]
            jWeight = weights[jValToSwap]

            jBinWeight = candidateSolution["bin_weights"][j]
            newJBinWeight = jBinWeight - jWeight + iWeight
            iBinWeight = candidateSolution["bin_weights"][i]
            newIBinWeight = iBinWeight - iWeight + jWeight

            if newJBinWeight <= binCapacity and newIBinWeight <= binCapacity:
                tweaked = True
                new = newJBinWeight*newJBinWeight + newIBinWeight*newIBinWeight
                original = jBinWeight*jBinWeight + iBinWeight*iBinWeight
                scoreChange = new - original
                scoreChange = scoreChange/(binCapacity)
                

                # If for example we get probability 0.6, we want a 60% chance of accepting, so we generate a random number from 0 to 1 and compare them.
                if (scoreChange < 0):
                    probability = math.exp(scoreChange/ temperature)
                    generatedVal = random.random()
                if scoreChange >= 0 or generatedVal < probability:
                    # Swap the items
                    candidateSolution["packing"][i].remove(iValToSwap)
                    candidateSolution["bin_weights"][i] -= iWeight
                    candidateSolution["packing"][i].append(jValToSwap)
                    candidateSolution["bin_weights"][i] += jWeight

                    candidateSolution["packing"][j].remove(jValToSwap)
                    candidateSolution["bin_weights"][j] -= jWeight
                    candidateSolution["packing"][j].append(iValToSwap)
                    candidateSolution["bin_weights"][j] += iWeight

        elif choice == 2:
            # If possible, move an item from bin i to bin j
            valToMove = random.choice(candidateSolution["packing"][i])
            itemWeight = weights[valToMove]

            iBinWeight = candidateSolution["bin_weights"][i]
            newIBinWeight = iBinWeight - itemWeight
            jBinWeight = candidateSolution["bin_weights"][j]
            newJBinWeight = jBinWeight + itemWeight

            if newJBinWeight <= binCapacity:
                tweaked = True
                new = newJBinWeight*newJBinWeight + newIBinWeight*newIBinWeight
                original = jBinWeight*jBinWeight + iBinWeight*iBinWeight
                scoreChange = new - original
                scoreChange = scoreChange/(binCapacity)

                # If for example we get probability 0.6, we want a 60% chance of accepting, so we generate a random number from 0 to 1 and compare them.
                if (scoreChange < 0):
                    probability = math.exp(scoreChange/ temperature)
                    generatedVal = random.random()
                if scoreChange >= 0 or generatedVal < probability:
                    # Move the item
                    candidateSolution["packing"][i].remove(valToMove)
                    candidateSolution["bin_weights"][i] -= itemWeight
                    candidateSolution["packing"][j].append(valToMove)
                    candidateSolution["bin_weights"][j] += itemWeight

                    if candidateSolution["bin_weights"][i] == 0:
                        candidateSolution["bin_weights"].pop(i)
                        candidateSolution["packing"].pop(i)

       
        # Decrease temperature
        temperature = temperature * cooling

    return candidateSolution

def simulatedAnnealingFFD(binCapacity, weights, timeLimit):
    return simulatedAnnealing(binCapacity, weights, True, timeLimit)

#def simulatedAnnealingFF(binCapacity, weights, timeLimit):
#    return simulatedAnnealing(binCapacity, weights, False, timeLimit)
