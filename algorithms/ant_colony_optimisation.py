import random
import math

def pickWeightIndex(curAntSol, weights, pheromoneScores, pheromoneImportance, heuristicImportance, binCapacity):
   if not curAntSol or sum(curAntSol[-1]) + weights[-1] > binCapacity:
      curAntSol.append([])
   
   #print(pheromoneScores)

   probabilities = []
   pheromoneParts = []
   total = 0
   for x in range(len(weights)-1, -1, -1):
      if sum(curAntSol[-1]) + weights[x] > binCapacity:
         break

      curWeight = weights[x]
      pheromonePart = 0
      if not curAntSol[-1]:
         pheromonePart = 1
      for j in curAntSol[-1]:
         pheromonePart += pheromoneScores[(curWeight, j)]

      pheromonePart = math.pow(pheromonePart, pheromoneImportance)
      heuristicScore = math.pow(curWeight, heuristicImportance)

      # These need to be scaled using the total later
      pheromoneParts.append(pheromonePart)
      probability = pheromonePart * heuristicScore
      probabilities.append(probability)
      total +=  probability
   
   thisProbability = random.random()
   for x in range(0, len(probabilities)):
      thisProbability -= probabilities[x]/total
      if thisProbability <= 0:
         weightIndex = len(weights) - 1 - x
         curAntSol[-1].append(weights[weightIndex])
         weights.pop(weightIndex)
         return

def updatePheromones(solution, fitness, pheromoneScores):
   # Note we already carried out evaporation
   # Here we are adding to the pheromones based on the solution and its fitness
   for bin in solution:
      for x in range(0, len(bin)):
         for y in range(0, len(bin)):
            if x != y:
               pheromoneScores[(bin[x], bin[y])] += fitness

def antColonyOptimisation(binCapacity, weights):
   populationSize = math.ceil(len(weights) * 0.8) # Number of "ant trails"
   evaporationParameter = 0.95
   heuristicImportance = 2
   pheromoneImportance = 5
   numItems = len(weights)
   
   iterationsBetweenGlobalReinforcement = math.ceil(500/numItems)
   # Approximate τmax based on work by Stutzle and Hoos
   tmax = 1/(1- evaporationParameter)
   
   # Define a lower limit τmax also based on work by Stutzle and Hoos
   avg = numItems/2
   pbest = 0.05
   tmin = (tmax * (1 - math.pow(pbest, 1/numItems))) / ((avg - 1) * math.pow(pbest, 1/numItems))

   weights.sort(reverse=True)

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
   
   # Have each ant build a solution
   for y in range (0, 15):
      bestSolutionThisIterationFitness = 0
      bestSolutionThisIteration = []
      for x in range(0, populationSize):
         antWeightSet = weights.copy()
         curAntSol = []
         while antWeightSet:
            pickWeightIndex(curAntSol, antWeightSet, pheromoneScores, pheromoneImportance, heuristicImportance, binCapacity)
         
   
         fitness = 0
         for group in curAntSol:
            groupWeight = sum(group)
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

      if y % iterationsBetweenGlobalReinforcement == 0:
         # update pheromone using best
         updatePheromones(bestSolutionSoFar, bestSolutionSoFarFitness, pheromoneScores)
      else:
         updatePheromones(bestSolutionThisIteration, bestSolutionThisIterationFitness, pheromoneScores)

      for pair in pheromoneScores:
         pheromoneScores[pair] = max(tmin, pheromoneScores[pair])


   bins = {}
   bins["packing"], bins["bin_weights"] = [], []
   for group in bestSolutionSoFar:
      bins["packing"].append(group)
      bins["bin_weights"].append(sum(group))

   return bins
   




