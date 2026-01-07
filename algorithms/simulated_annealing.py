import math
import random
from . import helpers

# Decreasing is to decide whether to use first fit with or without putting the data in decreasing order first
def simulatedAnnealing(binCapacity, weights, decreasing):

    temperature = 100000
    candidateSolution = helpers.firstFit(binCapacity, weights, decreasing, False)

    #print(len(candidateSolution["bin_weights"]))
    """num1s = 0
    num2s = 0
    acc1s = 0
    acc2s = 0"""

    # We can use the lower bound as a way to check if we have arrived at the ideal solution (though it may not be achievable)
    lowerBound = math.ceil(sum(weights)/binCapacity)
    iteration = 0
    while len(candidateSolution["bin_weights"]) > lowerBound and temperature > 0.01 and iteration < 100000:
        iteration += 1

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
            #num1s += 1
            # If possible, swap a pair of items in bin i and j
            iValToSwap = random.choice(candidateSolution["packing"][i])
            jValToSwap = random.choice(candidateSolution["packing"][j])

            jBinWeight = candidateSolution["bin_weights"][j]
            newJBinWeight = jBinWeight - jValToSwap + iValToSwap
            iBinWeight = candidateSolution["bin_weights"][i]
            newIBinWeight = iBinWeight - iValToSwap + jValToSwap

            if newJBinWeight <= binCapacity and newIBinWeight <= binCapacity:
                #acc1s += 1
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
                    candidateSolution["bin_weights"][i] -= iValToSwap
                    candidateSolution["packing"][i].append(jValToSwap)
                    candidateSolution["bin_weights"][i] += jValToSwap

                    candidateSolution["packing"][j].remove(jValToSwap)
                    candidateSolution["bin_weights"][j] -= jValToSwap
                    candidateSolution["packing"][j].append(iValToSwap)
                    candidateSolution["bin_weights"][j] += iValToSwap

        elif choice == 2:
            #num2s += 1
            # If possible, move an item from bin i to bin j
            valToMove = random.choice(candidateSolution["packing"][i])

            iBinWeight = candidateSolution["bin_weights"][i]
            newIBinWeight = iBinWeight - valToMove
            jBinWeight = candidateSolution["bin_weights"][j]
            newJBinWeight = jBinWeight + valToMove

            if newJBinWeight <= binCapacity:
                #acc2s += 1
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
                    #print("We did move")
                    # Move the item
                    candidateSolution["packing"][i].remove(valToMove)
                    candidateSolution["bin_weights"][i] -= valToMove
                    candidateSolution["packing"][j].append(valToMove)
                    candidateSolution["bin_weights"][j] += valToMove

                    if candidateSolution["bin_weights"][i] == 0:
                        #print("We removed a whole bin!!!")
                        candidateSolution["bin_weights"].pop(i)
                        candidateSolution["packing"].pop(i)

        if tweaked:
            # Decrease temperature
            temperature = temperature * 0.995

    #print(len(candidateSolution["bin_weights"]))
    """print("start")
    print(num1s)
    print(num2s)
    print(acc1s)
    print(acc2s)
    print("end")"""
    return candidateSolution

def simulatedAnnealingFFD(binCapacity, weights):
    return simulatedAnnealing(binCapacity, weights, True)

def simulatedAnnealingFF(binCapacity, weights):
    return simulatedAnnealing(binCapacity, weights, False)
