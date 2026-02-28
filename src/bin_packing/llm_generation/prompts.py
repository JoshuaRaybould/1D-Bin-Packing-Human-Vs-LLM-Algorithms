CODING_ROLE = "You are a senior algorithm designer and Python engineer specializing in combinatorial optimization and metaheuristics."
CODE_SYSTEM_PROMPT = CODING_ROLE + " Reason about and write the required Python code."
CORRECTNESS_SYSTEM_PROMPT = CODING_ROLE + " Reason about and fix the Python code."
PLAN_SYSTEM_PROMPT = "You are a senior algorithm architect and code‑review strategist. You analyse designs, identify weaknesses, propose improvements, and produce detailed implementation plans."
APPLY_PLAN_SYSTEM_PROMPT = CODING_ROLE + " Follow the provided plan exactly to improve the given Python code."
IMPROVEMENT_SYSTEM_PROMPT = CODING_ROLE + " Use the provided performance results to improve the existing code, focusing on solution quality."
COT = "Let's think step by step."

PROBLEM_STATEMENT = """Problem
You are given:
- an integer bin capacity,
- a list of item weights (all integers),
- a time limit in seconds.

This is the integer variant of the offline bin packing problem: bins have an integer capacity C, and each item has an integer weight. All items are known in advance. Your goal is to produce a high-quality packing using the specified algorithmic approach. You may design any internal heuristics, metaheuristics, or hybrid strategies you find appropriate, as long as they produce strong solutions.

Required function signature:

    def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:"""

HARD_REQUIREMENTS = """Hard requirements:
- The function must be named `solve` and accept exactly the three arguments above, in that order, with no additional parameters.
- The return value must be a dictionary with:
  - "packing": a list of lists of item indices (bin 0 contains the indices in packing[0], etc.)
  - "bin_weights": a list of total weights for each bin, aligned with the packing.
- The algorithm must run for a fixed number of iterations of your choosing, and must also check the time limit periodically. If the time limit is exceeded, return the best solution found so far.
- Randomness must use only Python’s global `random` module. Do not call random.seed() inside solve.
- The code must not execute anything at import time.
- Do not use external libraries beyond the Python standard library.
- The implementation should aim for strong, competitive solutions rather than trivial heuristics."""

def getInitialPrompt(algoName):
    userPrompt = f"""
Your task is to implement {algoName} for the 1D offline integer bin packing problem in Python.

{PROBLEM_STATEMENT}

{HARD_REQUIREMENTS}

{COT}
"""
    print(userPrompt)
    return userPrompt

def getCorrectnessPrompt(algoName, code, error):
    userPrompt = f"""
Your task is to fix the given implementation of {algoName} for the 1D offline integer bin packing problem in Python.

{PROBLEM_STATEMENT}

{HARD_REQUIREMENTS}

You must:
- Determine the cause of the error.
- Correct the implementation.
- Preserve or improve the algorithmic quality of the approach.

Code:
{code}

Error:
{error}

{COT}
"""
    print(userPrompt)
    return userPrompt

def getPlanPrompt(algoName, prevCode, performanceMetrics, maxTime):
    avgRatio = performanceMetrics["avg_ratio"] 
    extraBins = performanceMetrics["extra_bins"]
    timeTaken = min(performanceMetrics["time_sec"], maxTime)

    userPrompt = f"""
Your task is to plan how to improve this implementation of {algoName} for the 1D offline integer bin packing problem in Python.

Current implementation:
{prevCode}

Your goal is to produce a concrete, implementation‑ready plan that will improve the algorithm so it reaches stronger solutions within the given time limit.

Performance feedback:
- Average ratio (bins used by algorithm divided by best known lower bound): {avgRatio}
- Extra bins used over the lower bound: {extraBins}
- Time used: {timeTaken} seconds out of the {maxTime}‑second budget.

The algorithm is expected to use the entire time budget to search for better solutions. There is no penalty for longer runtimes; execution will simply stop at 100 seconds.

Use this feedback to guide your improvements, focus on reducing the average ratio (the average number of bins used by the algorithm divided by the best available lower bound). Identify weaknesses and propose improvements.

The following hard requirements will be enforced when implementing your plan, so your plan must not conflict with any of them:
{HARD_REQUIREMENTS}

Do not write any code. Produce only a detailed plan.
{COT}
"""
    print(userPrompt)
    return userPrompt

def applyPlanPrompt(algoName, code, plan):
    userPrompt = f"""Your task is to apply the provided plan to improve the implementation of {algoName} for the 1D offline integer bin packing problem in Python.

{PROBLEM_STATEMENT}

{HARD_REQUIREMENTS}

Follow the plan exactly. Do not omit steps or introduce ideas that are not present in the plan.

Code:
{code}

Plan:
{plan}

{COT}
"""
    print(userPrompt)
    return userPrompt

def improvePerformancePrompt(algoName, code, performanceMetrics, maxTime):
    avgRatio = performanceMetrics["avg_ratio"] 
    extraBins = performanceMetrics["extra_bins"]
    timeTaken = min(performanceMetrics["time_sec"], maxTime)

    userPrompt = f"""The following code implements the algorithm {algoName} for the 1D offline integer bin packing problem in Python.
Code:
{code}

Performance metrics:
- Average ratio (bins used by algorithm divided by best known lower bound): {avgRatio}
- Extra bins used over the lower bound: {extraBins}
- Time used: {timeTaken} seconds out of the {maxTime}‑second budget.

The algorithm is expected to use the entire time budget to search for better solutions. There is no penalty for longer runtimes; execution will simply stop at 100 seconds.

Please improve the code and adjust parameters where appropriate. The specifics of the implementation are entirely up to you, but it should arrive at very strong solutions.

{HARD_REQUIREMENTS}

{COT}
"""
    print(userPrompt)
    return userPrompt
