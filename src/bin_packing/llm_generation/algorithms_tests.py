import importlib.util
import time
import traceback
from bin_packing.utilities import test_correctness
from bin_packing.utilities import load_data
# from ..main import applyAlgorithm

def loadModuleFromPath(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
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
            endTime = time.perf_counter()
            if not isinstance(res, dict):
                return False, "INVALID_OUTPUT: solve must return a dict with keys 'packing' and 'bin_weights'.", None

            if "packing" not in res or "bin_weights" not in res:
                return False, (
                    "INVALID_OUTPUT: solve must return a dict with:\n"
                    "- 'packing': list[list[int]] (item indices per bin)\n"
                    "- 'bin_weights': list[float|int] (total weight per bin)\n"
                    "Both lists must be aligned (len(bin_weights) == len(packing))."
                ), None
            
            packing = res["packing"]
            bin_weights = res["bin_weights"]

            if not isinstance(packing, list) or not all(isinstance(b, list) for b in packing):
                return False, "INVALID_OUTPUT: 'packing' must be a list of lists of item indices.", None

            if not isinstance(bin_weights, list) or not all(isinstance(w, (int, float)) for w in bin_weights):
                return False, "INVALID_OUTPUT: 'bin_weights' must be a list of numbers.", None

            if len(packing) != len(bin_weights):
                return False, "INVALID_OUTPUT: 'packing' and 'bin_weights' must be aligned (same length).", None
                
            totalTime += (endTime - startTime)
            if endTime - startTime > 2*timeLimit:
                return False, "Time limit given to the algorithm was not respected.", None
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
    chosenInstance = instances[-1]

    shortTimeLimit = 0.1
    try:
        startTime = time.perf_counter()
        _ = solve(chosenInstance["bin_capacity"], chosenInstance["weights"], shortTimeLimit)
        timeUsed = time.perf_counter() - startTime
    except Exception as e:
        return False, f"RUNTIME_ERROR_UNDER_TIME_LIMIT({shortTimeLimit}): {type(e).__name__}: {e}", None

    if timeUsed > 0.2 + shortTimeLimit:
        return False, "Time limit given to the algorithm was not respected.", None

    longerTimeLimit = 0.5
    try:
        startTime = time.perf_counter()
        _ = solve(chosenInstance["bin_capacity"], chosenInstance["weights"], longerTimeLimit)
        timeUsed = time.perf_counter() - startTime
    except Exception as e:
        return False, f"RUNTIME_ERROR_UNDER_TIME_LIMIT({longerTimeLimit}): {type(e).__name__}: {e}", None

    if timeUsed > 0.2 + longerTimeLimit:
        return False, "Time limit given to the algorithm was not respected.", None
    
    # Check we still get valid solutions under small time budget
    shortTimeBudget = 0.2
    for instance in instances:
        try:
            res = solve(instance["bin_capacity"], instance["weights"], shortTimeBudget) 
            packing = res["packing"]
            bin_weights = res["bin_weights"]
            algBins = len(packing)
            optBins = instance["optimal_solution"]
            test_correctness.validatePacking(instance, res, algBins, optBins)
        except Exception as e:
            return False, f"INVALID_ANSWER: {e}", None
        
    return True, "Success", solve
