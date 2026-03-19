import pickle
import random
from . import helpers
import time

def calcFitness(solution):
    fitness = 0
    for binWeight in solution["bin_weights"]:
        fitness += (binWeight * binWeight)
    return fitness

def bestImprovement(solution, weights, binCapacity):
    while True:
        # First iterate through all bins, selecting items from just those which aren't full
        items = []
        bins  = []
        for x in range(0, len(solution["bin_weights"])):
            if solution["bin_weights"][x] != binCapacity:
                bins.append(x)
                for item in solution["packing"][x]:
                    items.append([item, x])
        items = sorted(items, key=lambda x: weights[x[0]])

        # First value in tuple is index in items of the item to move. Second is bin to move to.
        bestTransfer = (0, 0) 
        bestTransferScore = 0

        # Values in tuple are indexes in items, corresponding to those items we will swap.
        bestSwap = (0, 0) 
        bestSwapScore = 0

        # Find all possible transfers, for each bin check all items until they no longer fit
        for bin in bins:
            for item in items:
                if solution["bin_weights"][bin] + weights[item[0]] <= binCapacity:
                    if item[1] != bin:
                        bin1Score, bin2Score = solution["bin_weights"][bin], solution["bin_weights"][item[1]]
                        originalBinsScore = (bin1Score * bin1Score) + (bin2Score * bin2Score)

                        newBin1Score = bin1Score + weights[item[0]]
                        newBin2Score = bin2Score - weights[item[0]]
                        newBinsScore = (newBin1Score * newBin1Score) + (newBin2Score * newBin2Score)

                        transferScore = newBinsScore - originalBinsScore
                        if transferScore > bestTransferScore:
                            bestTransfer = (item, bin)
                            bestTransferScore = transferScore
                else:
                    break

        # Find all possible swaps
        for item1 in items:
            for item2 in items:
                if item1[0] != item2[0] and item1[1] != item2[1]:
                    item1Weight, item2Weight = weights[item1[0]], weights[item2[0]]
                    item1Bin, item2Bin = item1[1], item2[1]

                    # If swapping is feasible
                    if solution["bin_weights"][item2Bin] + item1Weight - item2Weight <= binCapacity and solution["bin_weights"][item1Bin] + item2Weight - item1Weight <= binCapacity:
                        bin1Score, bin2Score = solution["bin_weights"][item1Bin], solution["bin_weights"][item2Bin]
                        originalBinsScore = (bin1Score * bin1Score) + (bin2Score * bin2Score)

                        newBin1Score = bin1Score - weights[item1[0]] + weights[item2[0]]
                        newBin2Score = bin2Score + weights[item1[0]] - weights[item2[0]]
                        newBinsScore = (newBin1Score * newBin1Score) + (newBin2Score * newBin2Score)

                        swapScore = newBinsScore - originalBinsScore
                        if swapScore > bestSwapScore:
                            bestSwap = (item1, item2)
                            bestSwapScore = swapScore

        if bestTransferScore == 0 and bestSwapScore == 0:
            return solution
        elif bestTransferScore >= bestSwapScore:
            itemToTransfer = bestTransfer[0][0]
            binToTransferFrom = bestTransfer[0][1]
            binToTransferTo = bestTransfer[1]

            solution["packing"][binToTransferTo].append(itemToTransfer)
            solution["bin_weights"][binToTransferTo] += weights[itemToTransfer]
            solution["containing_bin"][itemToTransfer] = binToTransferTo
      
            solution["packing"][binToTransferFrom].remove(itemToTransfer)
            solution["bin_weights"][binToTransferFrom] -= weights[itemToTransfer]

            if not solution["packing"][binToTransferFrom]:
                solution["packing"].pop(binToTransferFrom)
                solution["bin_weights"].pop(binToTransferFrom)
                bins.remove(binToTransferFrom) 
                # We need to update all of our bin indexes now
                for containingBin in solution["containing_bin"]:
                    if solution["containing_bin"][containingBin] > binToTransferFrom:
                        solution["containing_bin"][containingBin] -= 1
        else:
            item1ToSwap = bestSwap[0][0]
            item1Bin = bestSwap[0][1]
            item2ToSwap = bestSwap[1][0]
            item2Bin = bestSwap[1][1]
            
            # Deal with bin 1
            solution["packing"][item1Bin].remove(item1ToSwap)
            solution["bin_weights"][item1Bin] -= weights[item1ToSwap]
            solution["packing"][item1Bin].append(item2ToSwap)
            solution["bin_weights"][item1Bin] += weights[item2ToSwap]
            solution["containing_bin"][item2ToSwap] = item1Bin

            # Bin 2
            solution["packing"][item2Bin].remove(item2ToSwap)
            solution["bin_weights"][item2Bin] -= weights[item2ToSwap]
            solution["packing"][item2Bin].append(item1ToSwap)
            solution["bin_weights"][item2Bin] += weights[item1ToSwap]
            solution["containing_bin"][item1ToSwap] = item2Bin

def shake(incumbentSolution, unmoved, weights, binCapacity):
    moves = {}
    moves["single"], moves["swap"] = [], []
    selectedIndex = -1

    while not moves["single"] and not moves["swap"] and unmoved:
        selectedIndex = random.randint(0, len(unmoved) - 1)
        curItemIndex = unmoved[selectedIndex]
        curWeight = weights[curItemIndex]
        curItemBin = incumbentSolution["containing_bin"][curItemIndex]

        # Check and add any possible single moves
        for x in range(0, len(incumbentSolution["bin_weights"])):
            if incumbentSolution["bin_weights"][x] + curWeight <= binCapacity and x != curItemBin:
                moves["single"].append(x)

        # Do the same for swaps
        spaceWhenRemoved = binCapacity - (incumbentSolution["bin_weights"][curItemBin] - curWeight)

        for x in range(len(unmoved) - 1, -1, -1):
            swappingItemWeight = weights[unmoved[x]]
            if x != selectedIndex and swappingItemWeight != curWeight:
                if swappingItemWeight > spaceWhenRemoved:
                    continue
                elif swappingItemWeight >= curWeight and unmoved[x] not in incumbentSolution["packing"][curItemBin]:
                    # -1 meaning the bin unmoved[x] is in has not yet been determined (though it is in a different bin to our item)
                    moves["swap"].append([x, -1]) 
                    continue
                
                swappingItemBin = incumbentSolution["containing_bin"][unmoved[x]]

                otherBinSpace = binCapacity - (incumbentSolution["bin_weights"][swappingItemBin] - swappingItemWeight)
                if otherBinSpace >= curWeight:
                    moves["swap"].append([x, swappingItemBin])
        
        # if no moves can be done with our selected item, remove it
        if not moves["single"] and not moves["swap"]:
            unmoved.pop(selectedIndex)
    
    # Either we found a value with possible moves or ran out of options
    if unmoved:
        selectedMove = random.randint(0, len(moves["single"]) + len(moves["swap"]) - 1)
       
        if selectedMove < len(moves["single"]):
            binToMoveTo = moves["single"][selectedMove]


            incumbentSolution["packing"][curItemBin].remove(curItemIndex)
            incumbentSolution["bin_weights"][curItemBin] -= curWeight
            incumbentSolution["packing"][binToMoveTo].append(curItemIndex)
            incumbentSolution["bin_weights"][binToMoveTo] += curWeight
            incumbentSolution["containing_bin"][curItemIndex] = binToMoveTo

            if not incumbentSolution["packing"][curItemBin]:
                incumbentSolution["packing"].pop(curItemBin)
                incumbentSolution["bin_weights"].pop(curItemBin)
               
                # We need to update all of our bin indexes now
                for containingBin in incumbentSolution["containing_bin"]:
                    if incumbentSolution["containing_bin"][containingBin] > curItemBin:
                        incumbentSolution["containing_bin"][containingBin] -= 1

            unmoved.pop(selectedIndex)
        else:
            swappingMove = selectedMove - len(moves["single"])
            itemToSwapWith = moves["swap"][swappingMove][0]
            itemIndex = unmoved[itemToSwapWith]

            binToMoveTo = moves["swap"][swappingMove][1]

            if binToMoveTo == -1:
                binToMoveTo = incumbentSolution["containing_bin"][itemIndex]

            incumbentSolution["packing"][curItemBin].remove(curItemIndex)
            incumbentSolution["bin_weights"][curItemBin] -= curWeight
            incumbentSolution["packing"][curItemBin].append(itemIndex)
            incumbentSolution["bin_weights"][curItemBin] += weights[itemIndex]
            incumbentSolution["containing_bin"][itemIndex] = curItemBin 

            incumbentSolution["packing"][binToMoveTo].remove(itemIndex)
            incumbentSolution["bin_weights"][binToMoveTo] -= weights[itemIndex]
            incumbentSolution["packing"][binToMoveTo].append(curItemIndex)
            incumbentSolution["bin_weights"][binToMoveTo] += curWeight
            incumbentSolution["containing_bin"][curItemIndex] = binToMoveTo

            unmoved.pop(selectedIndex)
            unmoved.remove(itemIndex)

    return incumbentSolution
        

def variableNeighbourhoodSearch(binCapacity, weights, candidateSolution, timeLimit, useTimeLimit=False):
    if not useTimeLimit:
        timeLimit = 1000 # effectively unlimited
        maxIterations = 4
    else:
        maxIterations = float("inf") 
    start_time = time.time()
    timeBudget = 0.98 * timeLimit

    incumbentSolution = pickle.loads(pickle.dumps(candidateSolution, -1))
    incumbentFitness = calcFitness(incumbentSolution)

    kMax = 3
    iteration = 0

    lowerBound = helpers.getLowerBound(weights, binCapacity)

    indexArr = []

    for x in range(0, len(weights)):
        indexArr.append(x)

    while iteration < maxIterations and len(incumbentSolution["packing"]) > lowerBound: 
        iteration += 1
        k = 1

        while k < kMax + 1:
            elapsed = time.time() - start_time
            if elapsed >= timeBudget:
                return incumbentSolution

            if len(incumbentSolution["packing"]) == lowerBound:
                return incumbentSolution
            unmoved = indexArr.copy()
            newSolution = pickle.loads(pickle.dumps(incumbentSolution, -1))
            for x in range(0, k):
                newSolution = shake(newSolution, unmoved, weights, binCapacity)
            newSolution = bestImprovement(newSolution, weights, binCapacity)
            newSolFitness = calcFitness(newSolution)
            if newSolFitness > incumbentFitness:
                incumbentSolution = newSolution
                incumbentFitness = newSolFitness
                k = 1
            else:
                k += 1

    return incumbentSolution

def variableNeighbourhoodSearchFFD(binCapacity, weights, timeLimit, useTimeLimit):
    candidateSolution = helpers.firstFitWithContainingBin(binCapacity, weights, True)
    return variableNeighbourhoodSearch(binCapacity, weights, candidateSolution, timeLimit, useTimeLimit)


