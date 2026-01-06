import pickle
import random
from . import first_fit

def findContainingBin(itemIndex, candidateSolution):
    for b in range(0, len(candidateSolution["packing"])):
            if itemIndex in candidateSolution["packing"][b]:
                return b
            
def bestImprovement(solution, weights, binCapacity):
    print("THE BEST")
    return solution
        

def shake(incumbentSolution, unmoved, weights, binCapacity):
    moves = {}
    moves["single"], moves["swap"] = [], []
    selectedIndex = -1

    while not moves["single"] and not moves["swap"] and unmoved:
        selectedIndex = random.randint(0, len(unmoved) - 1)
        curItemIndex = unmoved[selectedIndex]
        curWeight = weights[curItemIndex]
        curItemBin = findContainingBin(curItemIndex, incumbentSolution)

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
                    break
                elif swappingItemWeight >= curWeight and unmoved[x] not in incumbentSolution["packing"][curItemBin]:
                    # -1 meaning the bin unmoved[x] is in has not yet been determined (though it is in a different bin to our item)
                    moves["swap"].append([x, -1]) 
                    continue

                swappingItemBin = findContainingBin(unmoved[x], incumbentSolution)
                #print("item should be in the below bin")
                #print("item is: " + str(unmoved[x]))
                #print(incumbentSolution["packing"][swappingItemBin])
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

            unmoved.pop(selectedIndex)
        else:
            swappingMove = selectedMove - len(moves["single"])
            itemToSwapWith = moves["swap"][swappingMove][0]
            itemIndex = unmoved[itemToSwapWith]

            binToMoveTo = moves["swap"][swappingMove][1]

            if binToMoveTo == -1:
                binToMoveTo = findContainingBin(itemIndex, incumbentSolution)

            incumbentSolution["packing"][curItemBin].remove(curItemIndex)
            incumbentSolution["bin_weights"][curItemBin] -= curWeight
            incumbentSolution["packing"][curItemBin].append(itemIndex)
            incumbentSolution["bin_weights"][curItemBin] += weights[itemIndex]

            incumbentSolution["packing"][binToMoveTo].remove(itemIndex)
            incumbentSolution["bin_weights"][binToMoveTo] -= weights[itemIndex]
            incumbentSolution["packing"][binToMoveTo].append(curItemIndex)
            incumbentSolution["bin_weights"][binToMoveTo] += curWeight

            unmoved.pop(selectedIndex)
            unmoved.remove(itemIndex)

    return incumbentSolution
        

def variableNeighbourhoodSearch(binCapacity, weights, candidateSolution):
    incumbentSolution = candidateSolution
    
    kMax = 20
    iteration = 0
    totalIterations = 20

    indexArr = []
    #print(weights)
    #print(incumbentSolution)
    for x in range(0, len(weights)):
        indexArr.append(x)

    while iteration < totalIterations:
        iteration += 1

        unmoved = indexArr.copy()

        for k in range(0, kMax + 1):     
            solCopy = pickle.loads(pickle.dumps(incumbentSolution, -1))
            newSolution = shake(solCopy, unmoved, weights, binCapacity)
            newSolution = bestImprovement(newSolution)

            #return newSol
        

    return candidateSolution

def variableNeighbourhoodSearchFFD(binCapacity, weights):
    candidateSolution = first_fit.firstFit(binCapacity, weights, True, True)
    return variableNeighbourhoodSearch(binCapacity, weights, candidateSolution)


