import random

def randomisedBestFit(binCapacity, weights, timeLimit):
    # First randomise the order of the items. We will use the Fisher-Yates shuffle as it produces an unbiased permutation
    indexes = []
    for x in range(0, len(weights)):
        indexes.append(x)

    for x in range(0, len(weights)):
        num = random.randint(x,len(weights) - 1)
        tmp = indexes[x]
        indexes[x] = indexes[num]
        indexes[num] = tmp

    # Then do best fit
    # Each bin is of the form [[values], total_weight]
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []
    for index in indexes:
        minimalSufficientInd = -1
        for b in range(0, len(bins["bin_weights"])):
            binSpace = binCapacity - bins["bin_weights"][b]
            minSpaceSoFar = binCapacity - bins["bin_weights"][minimalSufficientInd]
            if binSpace > weights[index] and (minimalSufficientInd == -1 or binSpace < minSpaceSoFar):
                minimalSufficientInd = b
            elif binSpace == weights[index]:
                minimalSufficientInd = b
                break

        if minimalSufficientInd == -1:
            bins["packing"].append([index])
            bins["bin_weights"].append(weights[index])
        else:
            bins["packing"][minimalSufficientInd].append(index)
            bins["bin_weights"][minimalSufficientInd] += weights[index]
   
    # We will return the packing
    return bins




