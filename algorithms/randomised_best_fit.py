import random

def randomisedBestFit(binCapacity, weights):
    # First randomise the order of the items/weights. We will use the Fisher-Yates shuffle as it produces an unbiased permutation
    for x in range(0, len(weights)):
        num = random.randint(x,len(weights) - 1)
        tmp = weights[x]
        weights[x] = weights[num]
        weights[num] = tmp

    # Then do best fit
    # Each bin is of the form [[values], total_weight]
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []
    for weight in weights:
        minimalSufficientInd = -1
        for b in range(0, len(bins["bin_weights"])):
            binSpace = binCapacity - bins["bin_weights"][b]
            minSpaceSoFar = binCapacity - bins["bin_weights"][minimalSufficientInd]
            if binSpace > weight and (minimalSufficientInd == -1 or binSpace < minSpaceSoFar):
                minimalSufficientInd = b
            elif binSpace == weight:
                minimalSufficientInd = b
                break

        if minimalSufficientInd == -1:
            bins["packing"].append([weight])
            bins["bin_weights"].append(weight)
        else:
            bins["packing"][minimalSufficientInd].append(weight)
            bins["bin_weights"][minimalSufficientInd] += weight

    # We will return the packing
    return bins




