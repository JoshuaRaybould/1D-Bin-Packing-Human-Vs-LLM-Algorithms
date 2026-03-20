import pickle
import random
from . import helpers
import time

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
def tabuSearch(binCapacity, weights, candidateSolution, fastSearch, timeLimit, useTimeLimit=False):
   if not useTimeLimit:
      timeLimit = 1000 # effectively unlimited
      maxIterations = 18500
   else:
      maxIterations = float("inf") 
   start_time = time.time()
   timeBudget = 0.98 * timeLimit

   tabuList = []
   tabuMaxLen = 8
   tabuPos = 0
   emptyProb = 0.6
   emptyMinPercent = 1 - emptyProb

   bestSolution = pickle.loads(pickle.dumps(candidateSolution, -1))

   # We can use the lower bound as a way to check if we have arrived at the ideal solution (though it may not be achievable)
   lowerBound = helpers.getLowerBound(weights, binCapacity)

   if fastSearch:
      maxIterations = 1800
   iteration = 0
   while len(candidateSolution["bin_weights"]) > lowerBound and iteration < maxIterations:
      elapsed = time.time() - start_time
      if elapsed >= timeBudget:
            break
      iteration += 1

      # We select the best of these neighbours provided none are in the tabu list
      numNeighbours = 35
      bestScoreSoFar = float("-inf")
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
               if scoreChange > bestScoreSoFar:
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
            # If possible, move an item from bin i to bin j
            # Target the empty bin (1 - emptyMinPercent) of the time
            useEmptiest = random.random()
            if useEmptiest > emptyMinPercent:
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
               if scoreChange > bestScoreSoFar:
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
            tabuList = []
            tabuPos = 0

   return determineBest(bestSolution, candidateSolution)

def tabuSearchFFD(binCapacity, weights, timeLimit, useTimeLimit=False):
    candidateSolution = helpers.firstFit(binCapacity, weights, True)
    return tabuSearch(binCapacity, weights, candidateSolution, False, timeLimit, useTimeLimit)

