import pickle
import random
from . import helpers

def determineBest(bestSolution, candidateSolution):
   bestSolScore = 0
   candidateScore = 0
   for weight in bestSolution["bin_weights"]:
      bestSolScore += (weight * weight)

   for weight in candidateSolution["bin_weights"]:
      candidateScore += (weight * weight)
   
   if bestSolScore >= candidateScore:
      return bestSolution
   else:
      return pickle.loads(pickle.dumps(candidateSolution, -1))

def addToTabuList(candidateSolution, i, j, tabuList, tabuPos, tabuMaxLen):
   # Handle the tabu list
   binICopy = candidateSolution["packing"][i].copy()
   binICopy.sort()
   tabuI = [i, binICopy]
   binJCopy = candidateSolution["packing"][j].copy()
   binJCopy.sort()
   tabuJ = [j, binJCopy]
   if len(tabuList) < tabuMaxLen:
      tabuList.append(tabuI)
      tabuList.append(tabuJ)
   else:
      tabuList[tabuPos % tabuMaxLen] = tabuI
      tabuPos += 1
      tabuList[tabuPos % tabuMaxLen] = tabuJ
      tabuPos += 1
   return tabuPos

def getEmptiestBin(candidateSolution):
   theEmptiest = -1
   smallestWeight = float("inf")
   for x in range(0, len(candidateSolution["bin_weights"])):
      curWeight = candidateSolution["bin_weights"][x]
      if smallestWeight > curWeight:
         theEmptiest = x
         smallestWeight = curWeight
   return theEmptiest

# Perform search on given candidate solution
# Set fastSearch true to reduce iterations
def tabuSearch(binCapacity, weights, candidateSolution, fastSearch):
   movement = 0
   swapping = 0
   num2s = 0
   num1s = 0

   tabuList = []
   tabuMaxLen = 8
   tabuPos = 0

   bestSolution = candidateSolution

   # We can use the lower bound as a way to check if we have arrived at the ideal solution (though it may not be achievable)
   lowerBound = helpers.getLowerBound(weights, binCapacity)
   totalIterations = 21500
   if fastSearch:
      totalIterations = 1500
   iteration = 0
   while len(candidateSolution["bin_weights"]) > lowerBound and iteration < totalIterations:
      iteration += 1

      # We select the best of these neighbours provided none are in the tabu list
      numNeighbours = 30
      bestScoreSoFar = 0
      emptiestBin = getEmptiestBin(candidateSolution)
      for _ in range(0, numNeighbours):
         selectedIandVal = [-1, -1]
         selectedJandVal = [-1, -1]

         # Select at random 2 bins
         numBins = len(candidateSolution["bin_weights"])
         i = random.randint(0, numBins-1)
         j = random.randint(0, numBins-1)
         while i == j:
            j = random.randint(0, numBins-1)

         # Randomly decide the type of tweak to apply
         choice = random.randint(1,2)

         if choice == 1:
            num1s += 1
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
               new = newJBinWeight*newJBinWeight + newIBinWeight*newIBinWeight
               old = jBinWeight*jBinWeight + iBinWeight*iBinWeight
               scoreChange = new - old

               # If the score isn't better than what we've seen we aren't picking it
               if bestScoreSoFar == 0 or scoreChange > bestScoreSoFar:
                  # If not taboo select it
                  newjBin = candidateSolution["packing"][j].copy()
                  newjBin.remove(jValToSwap)
                  newjBin.append(iValToSwap)
                  newiBin = candidateSolution["packing"][i].copy()
                  newiBin.remove(iValToSwap)
                  newiBin.append(jValToSwap)

                  newjBin.sort()
                  newiBin.sort()
                  if [j, newjBin] not in tabuList and [i, newiBin] not in tabuList:
                     bestScoreSoFar = scoreChange
                     selectedIandVal = [i, iValToSwap]
                     selectedJandVal = [j, jValToSwap]

         elif choice == 2:
            num2s += 1
            # If possible, move an item from bin i to bin j
            # Half the time intentionally target the emptiest bin instead of i
            useEmptiest = random.random()
            if useEmptiest > 0.4:
               if j != emptiestBin:
                  i = emptiestBin 
               else:
                  tmp = i
                  i = j
                  j = tmp

            valToMove = random.choice(candidateSolution["packing"][i])
            valWeight = weights[valToMove]

            iBinWeight = candidateSolution["bin_weights"][i]
            newIBinWeight = iBinWeight - valWeight
            jBinWeight = candidateSolution["bin_weights"][j]
            newJBinWeight = jBinWeight + valWeight

            if newJBinWeight <= binCapacity:
               new = newJBinWeight*newJBinWeight + newIBinWeight*newIBinWeight
               old = jBinWeight*jBinWeight + iBinWeight*iBinWeight
               scoreChange = new - old

               # If the score isn't better than what we've seen we  aren't picking it
               if bestScoreSoFar == 0 or scoreChange > bestScoreSoFar:
                  # If not taboo select it
                  newiBin = candidateSolution["packing"][i].copy()
                  newiBin.remove(valToMove)
                  newjBin = candidateSolution["packing"][j].copy()
                  newjBin.append(valToMove)

                  newiBin.sort()
                  newjBin.sort()

                  if newiBin == []:
                     # Guarantee we delete the bin
                     bestScoreSoFar = scoreChange
                     selectedIandVal = [i, valToMove]
                     selectedJandVal = [j, -1]
                     break
                  if [i, newiBin] not in tabuList and [j, newjBin] not in tabuList:
                     bestScoreSoFar = scoreChange
                     selectedIandVal = [i, valToMove]
                     selectedJandVal = [j, -1]

      if bestScoreSoFar < 0:
         bestSolution = determineBest(bestSolution, candidateSolution)

      if selectedIandVal[1] >= 0 and selectedJandVal[1] >= 0:
         swapping += 1
         # Swap the items
         i, iValToSwap = selectedIandVal[0], selectedIandVal[1]
         j, jValToSwap = selectedJandVal[0], selectedJandVal[1]

         # Handle the tabu list
         tabuPos = addToTabuList(candidateSolution, i, j, tabuList, tabuPos, tabuMaxLen)

         candidateSolution["packing"][i].remove(iValToSwap)
         candidateSolution["bin_weights"][i] -= weights[iValToSwap]
         candidateSolution["packing"][i].append(jValToSwap)
         candidateSolution["bin_weights"][i] += weights[jValToSwap]

         candidateSolution["packing"][j].remove(jValToSwap)
         candidateSolution["bin_weights"][j] -= weights[jValToSwap]
         candidateSolution["packing"][j].append(iValToSwap)
         candidateSolution["bin_weights"][j] += weights[iValToSwap]

      elif selectedIandVal[1] >= 0:
         movement += 1

         i, valToMove = selectedIandVal[0], selectedIandVal[1]
         j = selectedJandVal[0]
         # Handle the tabu list
         tabuPos = addToTabuList(candidateSolution, i, j, tabuList, tabuPos, tabuMaxLen)

         candidateSolution["packing"][i].remove(valToMove)
         candidateSolution["bin_weights"][i] -= weights[valToMove]
         candidateSolution["packing"][j].append(valToMove)
         candidateSolution["bin_weights"][j] += weights[valToMove]

         if candidateSolution["bin_weights"][i] == 0:
            candidateSolution["bin_weights"].pop(i)
            candidateSolution["packing"].pop(i)

   return determineBest(bestSolution, candidateSolution)

def tabuSearchFFD(binCapacity, weights):
    candidateSolution = helpers.firstFit(binCapacity, weights, True)
    return tabuSearch(binCapacity, weights, candidateSolution, False)

#def tabuSearchFF(binCapacity, weights):
#    candidateSolution = helpers.firstFit(binCapacity, weights, False)
#    return tabuSearch(binCapacity, weights, candidateSolution, False)
