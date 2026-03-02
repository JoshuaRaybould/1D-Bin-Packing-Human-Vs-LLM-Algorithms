from . import algorithms_tests
from . import providers
import bin_packing.main
import bin_packing.utilities.load_data
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

numCandidates = 3

#We have 1 intial prompt to obtain an algorithm, 3 for fixing and improving and 2 for making and applying a plan after the first correct algorithm
availablePrompts = 5

algoNamesForFiles = ["ant_colony_optimisation", "fitness_dependent_optimiser", "greedy_random_adaptive_search_procedure", "genetic_algorithm", "simulated_annealing", "tabu_search", "variable_neighbourhood_search"]
selected = 3
algoName = " ".join(algoNamesForFiles[selected].split("_"))
algoNameForFiles = algoNamesForFiles[selected]

llmProviders = ["openai", "anthropic", "google"]
chosenProvider = llmProviders[0]

# File name, average ratio of alg bins to opt bins, results, code
bestAlg = {"filename":"", "avg_ratio":float("infinity"), "results":None, "code":""}

minCorrectnessTime = 20
performanceRuns = 2
# We now need initial performance metrics
# We desire 20s per run on the instances, and have 10 instances of 400 items, and 10 instances of 800 items
# Average instance size = 600 items, and we desire a time of 1s on average giving 1/600
timePerItem = 1/600 
timePerRun = 20 # due to the above (this is purely for informing the LLM of time usage, it does not effect anything, if timePerItem is changed this should be too)


def append_correctness_log(filePath, name, correct, error):
    with filePath.open("a", encoding="utf-8") as f:
        if correct:
            f.write(f"{name} : CORRECT\n")
        else:
            f.write(f"{name} : {error}\n")

def performanceImprovement(llmProviders, chosenProvider, name, filePath, correctnessPath, resFileName, resultsDir,
                           algoName, code, results, instances, timePerItem, minCorrectnessTime, performanceRuns, timePerRun):

    if chosenProvider == llmProviders[0]: # openai
        code = providers.performanceOpenAIPrompt(algoName, code, results, performanceRuns * timePerRun)
    elif chosenProvider == llmProviders[1]: # anthropic
        code = providers.performanceAnthropicPrompt(algoName, code, results, performanceRuns * timePerRun)
    elif chosenProvider == llmProviders[2]: # google
        code = providers.performanceGooglePrompt(algoName, code, results, performanceRuns * timePerRun)

    with filePath.open("w", encoding="utf-8", newline="\n") as f:
        f.write(code)

    # Again we still check we have correctness
    correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
    append_correctness_log(correctnessPath, name, correct, error)

    if correct:                
        results = bin_packing.main.applyAlgorithm(instances, algorithm, performanceRuns, timePerItem)
        results["algorithm"] =  name
        print(results)
        bin_packing.main.append_summary_to_csv(resFileName, results, resultsDir)
        return correct, results, code
    print(error)
    return correct, None, code
    


ARTIFACTS_DIR = PROJECT_ROOT / "llm_artifacts" / chosenProvider / algoNameForFiles
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
responsesDir = ARTIFACTS_DIR / "responses"
responsesDir.mkdir(parents=True, exist_ok=True)
resultsDir = ARTIFACTS_DIR
correctnessPath = ARTIFACTS_DIR / "correctness.txt"

for candidateNum in range(numCandidates):
    promptsUsed = 0
    bestCurAlgo = {"filename":"", "avg_ratio":float("infinity"), "results":None, "code":""}

    name = algoNameForFiles + "_" + str(candidateNum) + "_initial"
    filePath = responsesDir / f"{name}.py"
    code = ""

    if chosenProvider == llmProviders[0]: # openai
        code = providers.initialOpenAIPrompt(algoName)
    elif chosenProvider == llmProviders[1]: # anthropic
        code = providers.initialAnthropicPrompt(algoName)
    elif chosenProvider == llmProviders[2]: # google
        code = providers.initialGooglePrompt(algoName)
    promptsUsed += 1

    with filePath.open("w", encoding="utf-8", newline="\n") as f:
        f.write(code)

    correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
    append_correctness_log(correctnessPath, name, correct, error)
    if not correct:
        print(error)

    # If the algorithm was not correct we reprompt, providing the code and error
    while not correct and promptsUsed < availablePrompts:
        print("Not correct")
        name = algoNameForFiles + "_" + str(candidateNum) + "_correctness_" + str(promptsUsed)
        filePath = responsesDir / f"{name}.py"

        # Attempt to fix algorithm
        if chosenProvider == llmProviders[0]: # openai
            code = providers.correctnessOpenAIPrompt(algoName, code, error)
        elif chosenProvider == llmProviders[1]: # anthropic
            code = providers.correctnessAnthropicPrompt(algoName, code, error)
        elif chosenProvider == llmProviders[2]: # google
            code = providers.correctnessGooglePrompt(algoName, code, error)
        promptsUsed += 1

        with filePath.open("w", encoding="utf-8", newline="\n") as f:
            f.write(code)
            
        correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
        append_correctness_log(correctnessPath, name, correct, error)
        if not correct:
            print(error)
  


    if not correct:
        continue
    
    instances = bin_packing.utilities.load_data.getTestInstances()

    resFileName = algoNameForFiles + "_results.csv"


    print("Correct")
    # We now need initial performance metrics

    results = bin_packing.main.applyAlgorithm(instances, algorithm, performanceRuns, timePerItem)
    results["algorithm"] =  name

    bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}

    bin_packing.main.append_summary_to_csv(resFileName, results, resultsDir)

    # We do not have enough prompts left to plan and also apply the plan, so attempt to improve in a single prompt
    if availablePrompts - promptsUsed == 1:
        name = algoNameForFiles + "_" + str(candidateNum) + "_performance_" + str(promptsUsed)
        filePath = responsesDir / f"{name}.py"

        correct, results, code = performanceImprovement(llmProviders, chosenProvider, name, filePath, correctnessPath, resFileName, resultsDir,
                           algoName, code, results, instances, timePerItem, minCorrectnessTime, performanceRuns, timePerRun)

        if not correct:
            code = bestCurAlgo["code"]
            results = bestCurAlgo["results"]
            print("Not correct")
        elif results is not None and results["avg_ratio"] < bestCurAlgo["avg_ratio"]:
            bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}
            print("Improved")
        else:
            print("Correct")
        
        promptsUsed += 1
    
        if bestCurAlgo["avg_ratio"] < bestAlg["avg_ratio"]:
            bestAlg = bestCurAlgo
        continue


        
    planFileName = algoNameForFiles + "_" + str(candidateNum) + "_plan.txt"
    planPath = responsesDir / planFileName
    name =  algoNameForFiles + "_" + str(candidateNum) + "_post_plan"
    filePath = responsesDir / f"{name}.py"

    if chosenProvider == llmProviders[0]: # openai
        performancePlan = providers.performancePlanOpenAIPrompt(algoName, code, results, performanceRuns * timePerRun)
        code = providers.applyPerformancePlanOpenAIPrompt(algoName, code, performancePlan)
    elif chosenProvider == llmProviders[1]: # anthropic
        performancePlan = providers.performancePlanAnthropicPrompt(algoName, code, results, performanceRuns * timePerRun)
        code = providers.applyPerformancePlanAnthropicPrompt(algoName, code, performancePlan)
    elif chosenProvider == llmProviders[2]: # google
        performancePlan = providers.performancePlanGooglePrompt(algoName, code, results, performanceRuns * timePerRun)
        code = providers.applyPerformancePlanGooglePrompt(algoName, code, performancePlan)
    promptsUsed += 2

    with planPath.open("w", encoding="utf-8", newline="\n") as f:
        f.write(performancePlan)

    with filePath.open("w", encoding="utf-8", newline="\n") as f:
        f.write(code)

    # We must now test we still have correctness, if this fails we revert back to the correct one (pre-plan)
    correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
    append_correctness_log(correctnessPath, name, correct, error)
    if correct:
        print("Correct")
        results = bin_packing.main.applyAlgorithm(instances, algorithm, performanceRuns, timePerItem)
        results["algorithm"] =  name
        if results["avg_ratio"] < bestCurAlgo["avg_ratio"]:
            bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}
            print("Improved")

        bin_packing.main.append_summary_to_csv(resFileName, results, resultsDir)


    while promptsUsed < availablePrompts:
        name = algoNameForFiles + "_" + str(candidateNum) + "_performance_" + str(promptsUsed)
        filePath = responsesDir / f"{name}.py"

        correct, results, code = performanceImprovement(llmProviders, chosenProvider, name, filePath, correctnessPath, resFileName, resultsDir,
                           algoName, code, results, instances, timePerItem, minCorrectnessTime, performanceRuns, timePerRun)

        if not correct:
            print("Not correct")
            code = bestCurAlgo["code"]
            results = bestCurAlgo["results"]
        elif results is not None and results["avg_ratio"] < bestCurAlgo["avg_ratio"]:
            print("Improved")
            bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}
        else:
            print("Correct")
            
        promptsUsed += 1
    
    if bestCurAlgo["avg_ratio"] < bestAlg["avg_ratio"]:
        bestAlg = bestCurAlgo


print(bestAlg["filename"])
print(bestAlg["avg_ratio"])