import importlib.util
import time
import traceback
from bin_packing.utilities import test_correctness
from bin_packing.utilities import load_data
# from ..main import applyAlgorithm

def loadModuleFromPath(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for path={path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # executes the file
    return module

def testCorrectness(algoName, algoPath, minTime):

    try:
        module = loadModuleFromPath(algoPath, algoName)
    except Exception as e:
        tb = "\n".join(traceback.format_exc().splitlines()[-5:])
        return False, f"IMPORT_ERROR: {type(e).__name__}: {e}\n{tb}", None

        
    solve = getattr(module, "solve", None)
    if solve is None or not callable(solve):
        return False, "ENTRYPOINT_ERROR: missing callable solve(bin_capacity, weights, time_limit)", None
   
    timeLimit = 2 # This is for a single instance
    instances = load_data.getTestInstances()
    timeBudget = 2 * len(instances)
    totalTime = 0

    for instance in instances:

        try:
            startTime = time.perf_counter()
            res = solve(instance["bin_capacity"], instance["weights"], timeLimit) 
            if not (isinstance(res, dict) and "packing" in res and "bin_weights" in res):
                return False, "INVALID_OUTPUT: solve must return (binPacking, binWeights)", None
            
            endTime = time.perf_counter()
            totalTime += (endTime - startTime)
            print(totalTime)
        except Exception as e:
            tb = "\n".join(traceback.format_exc().splitlines()[-5:])
            return False, f"RUNTIME_ERROR: {type(e).__name__}: {e}\n{tb}", None
   
        algBins = len(res["packing"])
        optBins = instance["optimal_solution"]
        try:
            test_correctness.validatePacking(instance, res, algBins, optBins)
        except Exception as e:
            return False, f"INVALID_ANSWER: {e}", None
    
    print(totalTime)
    if totalTime < minTime:
        return False, f"TIME: algorithm used too little time of the time budget. Took: " + str(totalTime) + ", Time budget: " + str(timeBudget) + " (for 20 instances). This means when given a time limit, the algorithm terminates much before the time is reached.", None
    
        
    # Check that it is actually staying true to the time limit
    chosenInstance = instances[-1] # This will be a bigger instance
    shortTimeLimit = 0.01

    startTime = time.perf_counter()
    res = solve(chosenInstance["bin_capacity"], chosenInstance["weights"], shortTimeLimit) 
    timeUsed = startTime - time.perf_counter()
    if timeUsed > 5 * shortTimeLimit:
        # Time limit has not been respected
        return False, f"Time limit given to the algorithm was not respected: {e}", None
    
    longerTimeLimit = 0.1
    
    startTime = time.perf_counter()
    res = solve(chosenInstance["bin_capacity"], chosenInstance["weights"], longerTimeLimit) 
    timeUsed = startTime - time.perf_counter()
    if timeUsed > 2 * longerTimeLimit:
        # Time limit has not been respected
        return False, f"Time limit given to the algorithm was not respected: {e}", None

        
    return True, "Success", solve
