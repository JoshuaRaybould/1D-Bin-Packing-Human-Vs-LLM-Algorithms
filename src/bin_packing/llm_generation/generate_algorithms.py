from . import algorithms_tests
from . import providers
import bin_packing.main
import bin_packing.utilities.load_data

# If the algorithm takes longer than this on a single run we say it is not correct
timeLimit = 40 

numCandidates = 1

#We have 1 intial propmpt to obtain an algorithm
availablePrompts = 6 # We then have 5 available prompts to fix and improve the algorithm
# After obtaining a working algorithm we apply a prompt to plan how to improve our algorithm then a prompt to apply this plan
# (this part is not effected by the availablePrompt value above)

# Total prompts is initial + correctness + plan + apply plan + performance improvement
# correctness + performance improvement = availablePrompts. So total is initial + plan + apply plan + availablePrompts = 3 + availabePrompts

promptsUsed = 0

algoName = "simulated annealing"
algoNameForFile = "simulated_annealing"

llmProviders = ["openai", "anthropic", "google"]
chosenProvider = llmProviders[0]

minCorrectnessTime = 20

dir_name = "../llm_artifacts/generation/" + chosenProvider + "/"

for candidateNum in range(numCandidates):

    name = algoNameForFile + "_" + str(candidateNum) + "_initial"
    fileName = dir_name + name + ".py"
    code = ""

    if chosenProvider == llmProviders[0]: # openai
        print("Do intiial")
        code = providers.initialOpenAIPrompt(algoName)

        with open(fileName, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)

    correct, error, algorithm = algorithms_tests.testCorrectness(name, fileName, minCorrectnessTime)
    if not correct:
        print(error)

    promptsUsed = 0
    # If the algorithm was not correct we reprompt, providing the code and error
    while not correct and promptsUsed < availablePrompts:
        print("Not correct")
        name = algoNameForFile + "_" + str(candidateNum) + "_correctness_" + str(promptsUsed)
        fileName = dir_name + name + ".py"

        # Attempt to fix algorithm
        if chosenProvider == llmProviders[0]: # openai
            code = providers.correctnessOpenAIPrompt(algoName, code, error)

            with open(fileName, "w", encoding="utf-8", newline="\n") as f:
                f.write(code)
            
        correct, error, algorithm = algorithms_tests.testCorrectness(name, fileName, minCorrectnessTime)
        if not correct:
            print(error)

        promptsUsed += 1

    
    instances = bin_packing.utilities.load_data.getTestInstances()

    curName = name
    curCode = code
    curAlgorithm = algorithm

    resFile = dir_name + algoNameForFile + "_results.csv"
    performanceRuns = 2

    if correct:
        print("Correct")
        # We now need initial performance metrics
        # We desire 20s total, and have 10 instances of 400 items, and 10 instances of 800 items
        # Average instance size = 600 items, and we desire a time of 1s on average giving 1/600

        timePerItem = 1/600 
        results = bin_packing.main.applyAlgorithm(instances, curAlgorithm, performanceRuns, timePerItem)
        results["algorithm"] =  curName

        bin_packing.main.append_summary_to_csv(resFile, results)
    
        planFileName =  dir_name + algoNameForFile + "_" + str(candidateNum) + "_plan.txt"
        name =  algoNameForFile + "_" + str(candidateNum) + "_post_plan"
        fileName = dir_name + name + ".py"

        performancePlan = providers.performancePlanOpenAIPrompt(algoName, curCode)

        with open(planFileName, "w", encoding="utf-8", newline="\n") as f:
            f.write(planFileName)

        code = providers.applyPerformancePlanOpenAIPrompt(algoName, curCode, performancePlan)

        with open(fileName, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)

        # We must now test we still have correctness, if this fails we revert back to the correct one (pre-plan)
        # If it is correct we will assume it has at least not gotten worse than the previous one
        correct, error, algorithm = algorithms_tests.testCorrectness(name, fileName, minCorrectnessTime)
        if correct:
            curName = name
            curCode = code
            curAlgorithm = algorithm

            # We now need initial performance metrics
            # We desire 20s total, and have 10 instances of 400 items, and 10 instances of 800 items
            # Average instance size = 600 items, and we desire a time of 1s on average giving 1/600
            timePerItem = 1/600 
            results = bin_packing.main.applyAlgorithm(instances, curAlgorithm, performanceRuns, timePerItem)
            results["algorithm"] =  curName
            print(results)
            print("SURESURE")

            bin_packing.main.append_summary_to_csv(resFile, results)
            print("RESfile")
            print(resFile)
            print("Resfile end")

        while promptsUsed < availablePrompts:
            name = algoNameForFile + "_" + str(candidateNum) + "_performance_" + str(promptsUsed)
            fileName = dir_name + name + ".py"

            if chosenProvider == llmProviders[0]: # openai
                code = providers.performanceOpenAIPrompt(algoName, curCode, results["time_sec"], results["avg_ratio"], results["extra_bins"])

                with open(fileName, "w", encoding="utf-8", newline="\n") as f:
                    f.write(code)

                # Again we still check we have correctness
                correct, error, algorithm = algorithms_tests.testCorrectness(name, fileName, minCorrectnessTime)
                if not correct:
                    print(error)
                
                if correct:
                    curName = name
                    curCode = code
                    curAlgorithm = algorithm
                    
                    results = bin_packing.main.applyAlgorithm(instances, curAlgorithm, performanceRuns, timePerItem)
                    results["algorithm"] =  curName
                    print(results)
                    bin_packing.main.append_summary_to_csv(resFile, results)
            
            promptsUsed += 1
