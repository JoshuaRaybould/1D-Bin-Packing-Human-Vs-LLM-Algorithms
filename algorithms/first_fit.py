import random

def firstFit(binCapacity, weights, decreasing):
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []

    if decreasing:
        weights.sort(reverse=True)
    else:
        random.shuffle(weights)

    for weight in weights:
        packed = False

        for b in range(0, len(bins["bin_weights"])):
            if bins["bin_weights"][b] + weight <= binCapacity:
                bins["packing"][b].append(weight)
                bins["bin_weights"][b] += weight
                packed = True
                break

        if not packed:
            bins["packing"].append([weight])
            bins["bin_weights"].append(weight)


    return bins
