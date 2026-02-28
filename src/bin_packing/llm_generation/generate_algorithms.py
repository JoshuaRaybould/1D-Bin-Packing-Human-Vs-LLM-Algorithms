from . import algorithms_tests
from . import providers
import bin_packing.main
import bin_packing.utilities.load_data
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

numCandidates = 2

#We have 1 intial prompt to obtain an algorithm, 6 for fixing and improving and 2 for making and applying a plan after the first correct algorithm
availablePrompts = 3

algoName = "simulated annealing"
algoNameForFiles = "simulated_annealing"

llmProviders = ["openai", "anthropic", "google"]
chosenProvider = llmProviders[0]

# File name, average bins used, results, code
bestAlg = {"filename":"", "avg_ratio":float("infinity"), "results":None, "code":""}

minCorrectnessTime = 20
performanceRuns = 2
# We now need initial performance metrics
# We desire 20s total, and have 10 instances of 400 items, and 10 instances of 800 items
# Average instance size = 600 items, and we desire a time of 1s on average giving 1/600
timePerItem = 1/600 
timePerRun = 20 # due to the above (this is purely for informing the LLM of time usage, it does not effect anything)

def performanceImprovement(llmProviders, chosenProvider, name, filePath, resFileName, resultsDir,
                           algoName, code, results, instances, timePerItem, minCorrectnessTime, performanceRuns, timePerRun):

    if chosenProvider == llmProviders[0]: # openai
        code = providers.performanceOpenAIPrompt(algoName, code, results, performanceRuns * timePerRun)

        with filePath.open("w", encoding="utf-8", newline="\n") as f:
            f.write(code)

        # Again we still check we have correctness
        correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
        
        if correct:                
            results = bin_packing.main.applyAlgorithm(instances, algorithm, performanceRuns, timePerItem)
            results["algorithm"] =  name
            print(results)
            bin_packing.main.append_summary_to_csv(resFileName, results, resultsDir)
            return correct, results, code
        return correct, None, code
    


ARTIFACTS_DIR = PROJECT_ROOT / "llm_artifacts" / chosenProvider / algoNameForFiles
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
responsesDir = ARTIFACTS_DIR / "responses"
responsesDir.mkdir(parents=True, exist_ok=True)
resultsDir = ARTIFACTS_DIR

for candidateNum in range(numCandidates):
    promptsUsed = 0
    bestCurAlgo = {"filename":"", "avg_ratio":float("infinity"), "results":None, "code":""}

    name = algoNameForFiles + "_" + str(candidateNum) + "_initial"
    filePath = responsesDir / f"{name}.py"
    code = ""

    if chosenProvider == llmProviders[0]: # openai
        print("Do intiial")
        code = providers.initialOpenAIPrompt(algoName)
        promptsUsed += 1

        with filePath.open("w", encoding="utf-8", newline="\n") as f:
            f.write(code)

    correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
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

            with filePath.open("w", encoding="utf-8", newline="\n") as f:
                f.write(code)
            
        correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
        if not correct:
            print(error)
        promptsUsed += 1


    if not correct:
        continue
    
    instances = bin_packing.utilities.load_data.getTestInstances()

    resFileName = algoNameForFiles + "_results.csv"


    print("Correct")
    # We now need initial performance metrics
    # We desire 20s total, and have 10 instances of 400 items, and 10 instances of 800 items
    # Average instance size = 600 items, and we desire a time of 1s on average giving 1/600

    timePerItem = 1/600 
    results = bin_packing.main.applyAlgorithm(instances, algorithm, performanceRuns, timePerItem)
    results["algorithm"] =  name

    bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}

    bin_packing.main.append_summary_to_csv(resFileName, results, resultsDir)

    # We do not have enough prompts left to plan and also apply the plan, so attempt to improve in a single prompt
    if availablePrompts - promptsUsed == 1:
        name = algoNameForFiles + "_" + str(candidateNum) + "_performance_" + str(promptsUsed)
        filePath = responsesDir / f"{name}.py"

        correct, results, code = performanceImprovement(llmProviders, chosenProvider, name, filePath, resFileName, resultsDir,
                           algoName, code, results, instances, timePerItem, minCorrectnessTime, performanceRuns, timePerRun)

        if not correct:
            code = bestCurAlgo["code"]
            results = bestCurAlgo["results"]
        elif results is not None and results["avg_ratio"] < bestCurAlgo["avg_ratio"]:
            bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}
        
        promptsUsed += 1
    
        if bestCurAlgo["avg_ratio"] < bestAlg["avg_ratio"]:
            bestAlg = bestCurAlgo
        continue


        
    planFileName = algoNameForFiles + "_" + str(candidateNum) + "_plan.txt"
    planPath = responsesDir / planFileName
    name =  algoNameForFiles + "_" + str(candidateNum) + "_post_plan"
    filePath = responsesDir / f"{name}.py"

    performancePlan = providers.performancePlanOpenAIPrompt(algoName, code, results, performanceRuns * timePerRun)
    promptsUsed += 1

    with planPath.open("w", encoding="utf-8", newline="\n") as f:
        f.write(performancePlan)

    code = providers.applyPerformancePlanOpenAIPrompt(algoName, code, performancePlan)
    promptsUsed += 1

    with filePath.open("w", encoding="utf-8", newline="\n") as f:
        f.write(code)

    # We must now test we still have correctness, if this fails we revert back to the correct one (pre-plan)
    correct, error, algorithm = algorithms_tests.testCorrectness(name, filePath, minCorrectnessTime)
    if correct:

        results = bin_packing.main.applyAlgorithm(instances, algorithm, performanceRuns, timePerItem)
        results["algorithm"] =  name
        if results["avg_ratio"] < bestCurAlgo["avg_ratio"]:
            bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}

        print(results)
        print("SURESURE")

        bin_packing.main.append_summary_to_csv(resFileName, results, resultsDir)
        print("RESfile")
        print(resFileName)
        print("Resfile end")

    while promptsUsed < availablePrompts:
        name = algoNameForFiles + "_" + str(candidateNum) + "_performance_" + str(promptsUsed)
        filePath = responsesDir / f"{name}.py"

        correct, results, code = performanceImprovement(llmProviders, chosenProvider, name, filePath, resFileName, resultsDir,
                           algoName, code, results, instances, timePerItem, minCorrectnessTime, performanceRuns, timePerRun)

        if not correct:
            code = bestCurAlgo["code"]
            results = bestCurAlgo["results"]
        elif results is not None and results["avg_ratio"] < bestCurAlgo["avg_ratio"]:
            bestCurAlgo = {"filename":name, "avg_ratio":results["avg_ratio"], "results":results, "code":code}
            
        promptsUsed += 1
    
    if bestCurAlgo["avg_ratio"] < bestAlg["avg_ratio"]:
        bestAlg = bestCurAlgo


print(bestAlg)