import random
import math
from . import helpers
import time

def convertToBins(solution, weights):
   bins = {}
   bins["packing"], bins["bin_weights"] = [], []
   for group in solution:
      bins["packing"].append(group)
      binWeight = 0
      for item in group:
         binWeight += weights[item]
      bins["bin_weights"].append(binWeight)
   return bins

def pickWeightIndex(curAntSol, indexes, pheromoneScores, pheromoneImportance, heuristicImportance, binCapacity, weights):
   lightestItem = indexes[-1]

   curBinWeight = 0
   if curAntSol:
      for item in curAntSol[-1]:
         curBinWeight += weights[item]

   # If we haven't started building the solution or there is not enough space in the last bin, open a new one
   if not curAntSol or curBinWeight + weights[lightestItem] > binCapacity:
      curAntSol.append([])
      curBinWeight = 0
   
   probabilities = []
   total = 0

   for x in range(len(indexes)-1, -1, -1):
      item = indexes[x]

      if curBinWeight + weights[item] > binCapacity:
         break

      curWeight = weights[item]
      pheromonePart = 0
      if not curAntSol[-1]:
         pheromonePart = 1
      for j in curAntSol[-1]:
         pheromonePart += pheromoneScores[(curWeight, weights[j])]

      pheromonePart = math.pow(pheromonePart, pheromoneImportance)
      heuristicScore = math.pow(curWeight, heuristicImportance)

      # These need to be scaled using the total later
      probability = pheromonePart * heuristicScore
      probabilities.append(probability)
      total +=  probability
   
   thisProbability = random.random()
   for x in range(0, len(probabilities)):
      thisProbability -= probabilities[x]/total
      if thisProbability <= 0:
         indexesIndex = len(indexes) - 1 - x
         weightIndex = indexes[indexesIndex]
         curAntSol[-1].append(weightIndex)
         indexes.pop(indexesIndex)
         return

def updatePheromones(solution, fitness, pheromoneScores, weights):
   # Note we already carried out evaporation
   # Here we are adding to the pheromones based on the solution and its fitness
   for bin in solution:
      for x in range(0, len(bin)):
         for y in range(0, len(bin)):
            if x != y:
               pheromoneScores[(weights[bin[x]], weights[bin[y]])] += fitness

def antColonyOptimisation(binCapacity, weights, timeLimit, useTimeLimit=False):
   if not useTimeLimit:
      timeLimit = 1000 # effectively unlimited
      maxIterations = 7
   else:
      maxIterations = float("inf")

   start_time = time.time()
   timeBudget = 0.98 * timeLimit

   populationSize = 3 # Number of "ant trails"
   evaporationParameter = 0.2
   heuristicImportance = 10
   pheromoneImportance = 2
   numItems = len(weights)
   iteration = 0
   
   iterationsBetweenGlobalReinforcement = math.ceil(500/numItems)
   # Approximate τmax based on work by Stutzle and Hoos
   tmax = 1/(1- evaporationParameter)
   
   # Define a lower limit τmax also based on work by Stutzle and Hoos
   avg = numItems/2
   pbest = 0.05
   tmin = (tmax * (1 - math.pow(pbest, 1/numItems))) / ((avg - 1) * math.pow(pbest, 1/numItems))

   indexes = []
   for x in range(0, len(weights)):
      indexes.append(x)
   indexes.sort(key=lambda itemIndex: weights[itemIndex], reverse=True)

   # Scores are between item weights
   # we set each possible weight pairing to have optimistic score tmax
   pheromoneScores = {}
   bestSolutionSoFar = []
   bestSolutionSoFarFitness = 0
   bestSolutionThisIteration = []
   totalWeight = sum(weights)

   for x in range(0, numItems):
      for y in range(0, numItems):
         if x != y:
            pheromoneScores[(weights[x], weights[y])] = tmax
   
   lowerBound = helpers.getLowerBound(weights, binCapacity)

   # Have each ant build a solution
   while iteration < maxIterations:
      iteration += 1

      elapsed = time.time() - start_time
      if elapsed >= timeBudget:
         return convertToBins(bestSolutionSoFar, weights)

      if bestSolutionSoFar and len(bestSolutionSoFar) == lowerBound:
         bins = convertToBins(bestSolutionSoFar, weights)
         return bins
      bestSolutionThisIterationFitness = 0
      bestSolutionThisIteration = []
      for x in range(0, populationSize):
         antIndexSet = indexes.copy()
         curAntSol = []
         while antIndexSet:
            elapsed = time.time() - start_time
            if elapsed >= timeBudget:
                  return convertToBins(bestSolutionSoFar, weights)
            pickWeightIndex(curAntSol, antIndexSet, pheromoneScores, pheromoneImportance, heuristicImportance, binCapacity, weights)
         
         fitness = 0
         for group in curAntSol:
            groupWeight = 0
            for index in group:
               groupWeight += weights[index]
            fitness += groupWeight * groupWeight
         fitness = fitness/totalWeight

         if fitness > bestSolutionSoFarFitness:
            bestSolutionSoFar = curAntSol
            bestSolutionSoFarFitness = fitness
         if fitness > bestSolutionThisIterationFitness:
            bestSolutionThisIteration = curAntSol
            bestSolutionThisIterationFitness = fitness
      
      for pair in pheromoneScores:
         pheromoneScores[pair] *= evaporationParameter
      
      elapsed = time.time() - start_time
      if elapsed >= timeBudget:
         return convertToBins(bestSolutionSoFar, weights)

      if iteration % iterationsBetweenGlobalReinforcement == 0:
         # update pheromone using best
         updatePheromones(bestSolutionSoFar, bestSolutionSoFarFitness, pheromoneScores, weights)
      else:
         updatePheromones(bestSolutionThisIteration, bestSolutionThisIterationFitness, pheromoneScores, weights)

      for pair in pheromoneScores:
         pheromoneScores[pair] = max(tmin, pheromoneScores[pair])

   bins = convertToBins(bestSolutionSoFar, weights)
   return bins
   




