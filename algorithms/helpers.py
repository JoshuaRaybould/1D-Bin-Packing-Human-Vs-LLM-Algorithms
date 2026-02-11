import random
import math

def firstFit(binCapacity, weights, decreasing):
    bins = {}
    bins["packing"], bins["bin_weights"] = [], []

    indexes = list(range(0, len(weights)))

    if decreasing:
        indexes.sort(key=lambda itemIndex: weights[itemIndex], reverse=True)
    else:
        random.shuffle(indexes)
        
    for x in range(0, len(weights)):
        weight = weights[indexes[x]]

        valueToPutInPacking = indexes[x]
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

# As the above implementation but gives us a way to quickly determine the bin an item is in 
# This is necessary to distinguish items of same weight in some cases
def firstFitWithContainingBin(binCapacity, weights, decreasing):
    bins = {}
    bins["packing"], bins["bin_weights"], bins["containing_bin"] = [], [], {}

    # items are labelled 0 up to len(weights) - 1
    # From https://stackoverflow.com/questions/18265935/how-do-i-create-a-list-with-numbers-between-two-values
    indexes = list(range(0, len(weights)))

    if decreasing:
        indexes.sort(key=lambda itemIndex: weights[itemIndex], reverse=True)
    else:
        random.shuffle(indexes)

    for x in range(0, len(indexes)):
        weight = weights[indexes[x]]

        valueToPutInPacking = indexes[x]
        packed = False

        for b in range(0, len(bins["bin_weights"])):
            if bins["bin_weights"][b] + weight <= binCapacity:
                bins["packing"][b].append(valueToPutInPacking)
                bins["bin_weights"][b] += weight
                bins["containing_bin"][indexes[x]] = b
                packed = True
                break

        if not packed:
            bins["packing"].append([valueToPutInPacking])
            bins["bin_weights"].append(weight)
            bins["containing_bin"][indexes[x]] = len(bins["bin_weights"]) - 1

    return bins


def getStoppingSum(weights, capacity):
    stoppingVal = 0
    for x in range(len(weights) - 1, -1, -1):
        if weights[x] > capacity/2:
            break
        else:
            stoppingVal += weights[x]
    return stoppingVal

# Martello and Toth's lower bound
def getLowerBound(weights, binCapacity):
    # We sort in case
    weightsCopy = weights.copy()
    weightsCopy.sort(reverse=True)

    k = 0
    curkIndex = -1

    I1andI2Const = 0 
    I2Size = 0
    I2Sum = 0
    I3Sum = 0

    bestLowerBound = 0

    lowerBound = 0

    I2Pos = -1

    index = 0
    #worseLowerBound = math.ceil((sum(weights))/binCapacity)
    stoppingSum = getStoppingSum(weightsCopy, binCapacity)
    while index < len(weightsCopy):
        if weightsCopy[index] <= binCapacity/2:
            curkIndex = index
            index += 1
            while index < len(weightsCopy) and weightsCopy[index] == weightsCopy[curkIndex]:
                curkIndex = index
                index += 1

            # Determine I1 and I2 Size
            I1Size = 0
            for x in range(0, len(weightsCopy)):
                if weightsCopy[x] > binCapacity - weightsCopy[curkIndex]:
                    I1Size += 1
                elif weightsCopy[x] > binCapacity/2:
                    if I2Pos == -1:
                        I2Pos = x
                    I2Size += 1
                    I2Sum += weightsCopy[x]
                elif weightsCopy[x] >= weightsCopy[curkIndex]:
                    if I2Pos == -1:
                        I2Pos = x
                    I3Sum += weightsCopy[x]
                else:
                    break
            
            I1andI2Const = I1Size + I2Size
            break
        index += 1

    smallItemTerm = (I3Sum - (I2Size * binCapacity - I2Sum))/binCapacity
    lowerBound = I1andI2Const + max(0, math.ceil(smallItemTerm))
    bestLowerBound = lowerBound

    stoppingSmallTerm = (stoppingSum - (I2Size * binCapacity - I2Sum))/binCapacity
    stoppingPoint = I1andI2Const + max(0, math.ceil(stoppingSmallTerm))

    if stoppingPoint <= lowerBound:
        return lowerBound

    startPoint = curkIndex + 1
    #print("I3 SUM " + str(I3Sum))
    # In this case all items are bigger than 1/2 capacity so each needs their own bin
    # (this should not be the case for our instances)
    if I2Pos == -1:
        print("Does your instance have very large items?")
        return len(weightsCopy)
    else:
        while startPoint < len(weightsCopy):
            curkIndex = startPoint
            k = weightsCopy[curkIndex]
            I3Sum += k
            curkIndex += 1

            while curkIndex < len(weightsCopy) and weightsCopy[curkIndex] == k:
                I3Sum += k
                startPoint = curkIndex
                curkIndex += 1

            I2Max = binCapacity - k

            while I2Pos > 0 and weightsCopy[I2Pos - 1] <= I2Max:
                I2Size += 1
                I2Sum += weightsCopy[I2Pos - 1]
                I2Pos -= 1

            smallItemTerm = (I3Sum - (I2Size * binCapacity - I2Sum))/binCapacity
            lowerBound = I1andI2Const + max(0, math.ceil(smallItemTerm))
                
            bestLowerBound = max(lowerBound, bestLowerBound)
            
            stoppingSmallTerm = (stoppingSum - (I2Size * binCapacity - I2Sum))/binCapacity
            stoppingPoint = I1andI2Const + math.ceil(stoppingSmallTerm)

            if stoppingPoint <= lowerBound:
                return lowerBound
            
            startPoint = curkIndex
        
        return bestLowerBound
