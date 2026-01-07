import random

# if index is true we put the index of the item in the packing rather than its weight
def firstFit(binCapacity, weights, decreasing, index):
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []

    if decreasing:
        weights.sort(reverse=True)
    else:
        random.shuffle(weights)
        
    for x in range(0, len(weights)):
        weight = weights[x]

        valueToPutInPacking = weight
        if index:
            valueToPutInPacking = x
        packed = False

        for b in range(0, len(bins["bin_weights"])):
            if bins["bin_weights"][b] + weight <= binCapacity:
                bins["packing"][b].append(valueToPutInPacking)
                bins["bin_weights"][b] += weight
                packed = True
                break

        if not packed:
            bins["packing"].append([valueToPutInPacking])
            bins["bin_weights"].append(weight)


    return bins
