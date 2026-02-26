from openai import OpenAI
from pydantic import BaseModel

import bin_packing.main
import bin_packing.utilities.load_data

class PythonCode(BaseModel):
    reasoning: str
    code: str

class PerformancePlan(BaseModel):
    reasoning: str
    plan: str

openAIModel = "gpt-5-nano"

HARD_REQUIREMENTS = "Requirements:\n " \
    "- The main method must be called solve and take as input the bin capacity as an integer, the weights of the items in an array and a time limit in seconds in that order only. No other arguments are allowed.\n" \
    "- It must return the packing you arrive at as a dictionary: where the key 'packing' corresponds to a list of lists, where each list contains the indexes of the items contained in it." \
    "For example index 0 will contain all indexes of the items packed in the 0th bin." \
    "Secondly the key 'bin_weights'  must correspond to a list of the total weight of each bin, such that index 0 corresponds to the weight of the 0th bin.\n" \
    "- The algorithm should terminate after a set number of iterations you decide. If a time limit is provided then it must be checked at regular intervals, and if it is exceeded then the best solution so far should be returned." \
    "- The evaluation harness seeds Python’s global RNG with random.seed(...). Therefore your code must use only the global random module for randomness. Do not call random.seed() inside solve.\n" \
    "- Your code must not have anything in it that will run when imported.\n" \

def initialOpenAIPrompt(algoName):
    client = OpenAI()

    systemPrompt = "You are an experienced python developer. Reason about and write the required python code"

    userPrompt = " Your task is: implement " + algoName + " for the 1D offline bin packing problem in python. " \
        "The specifics are entirely up to you but it should arrive at very strong solutions.\n" \
        "" + HARD_REQUIREMENTS +"\n" 

    # Initial attempt to get a correct algorithm
    response = client.responses.parse(
        model=openAIModel,
        input=[
            {"role": "system", "content": systemPrompt},
            {
                "role": "user",
                "content": userPrompt
                ,
            },
        ],
        text_format=PythonCode,
    )

    event = response.output_parsed

    return event.code


def correctnessOpenAIPrompt(algoName, prevCode, error):
    client = OpenAI()

    systemPrompt = "You are an experienced python developer. Reason about and fix the python code"
    
    userPrompt =  "The following code is for the algorithm: " + algoName + " for the 1D offline bin packing problem in python.\n" \
        "Code: \n" \
        "" + prevCode + "\n" \
        "Running this gave the error: " + error + "\n" \
        "Please fix the code. The specifics of the implementation are entirely up to you but it should arrive at very strong solutions.\n" \
        "" + HARD_REQUIREMENTS

    # Prompt with the code and error in attempt to fix it
    response = client.responses.parse(
        model=openAIModel,
        input=[
            {"role": "system", "content": systemPrompt},
            {
                "role": "user",
                "content": userPrompt,
            },
        ],
        text_format=PythonCode,
    )

    event = response.output_parsed

    return event.code

def performancePlanOpenAIPrompt(algoName, prevCode):
    client = OpenAI()

    systemPrompt = "You are an experienced python developer, with deep understanding of algorithms. Write a detailed plan of how to improve the performance of the given code."

    userPrompt = " Your task is: plan how to improve this implementation of " + algoName + " for the 1D offline bin packing problem in python. " \
        "implementation:\n" \
        "" + prevCode +"\n" \
        "The specifics are entirely up to you but when followed your plan should improve the code allowing it to reach stronger solutions in a given time.\n" \
        "The following requirements will be given along with your plan, so your plan must not conflict with any requirement.\n" \
        "" + HARD_REQUIREMENTS

    # Initial attempt to get a correct algorithm
    response = client.responses.parse(
        model=openAIModel,
        input=[
            {"role": "system", "content": systemPrompt},
            {
                "role": "user",
                "content": userPrompt
                ,
            },
        ],
        text_format=PythonCode,
    )

    event = response.output_parsed

    return event.code

def applyPerformancePlanOpenAIPrompt(algoName, prevCode, performancePlan):
    client = OpenAI()

    systemPrompt = "You are an experienced python developer. Apply the plan to improve the given code."

    userPrompt = " Your task is: plan how to improve this implementation of " + algoName + " for the 1D offline bin packing problem in python. " \
        "implementation:\n" \
        "" + prevCode +"\n" \
        "" + performancePlan + "\n" \
        "The specifics are entirely up to you but when followed your plan should improve the code allowing it to reach stronger solutions in a given time.\n" \
        "" + HARD_REQUIREMENTS

    # Initial attempt to get a correct algorithm
    response = client.responses.parse(
        model=openAIModel,
        input=[
            {"role": "system", "content": systemPrompt},
            {
                "role": "user",
                "content": userPrompt
                ,
            },
        ],
        text_format=PythonCode,
    )

    event = response.output_parsed

    return event.code

def performanceOpenAIPrompt(algoName, prevCode, timeTaken, avgRatio, extraBins):
    client = OpenAI()

    systemPrompt = "You are an experienced python developer. Reason about and improve the python code"
    
    userPrompt =  "The following code is for the algorithm: " + algoName + " for the 1D offline bin packing problem in python.\n" \
        "Code: \n" \
        "" + prevCode + "\n" \
        "Here are some peroformance details: \n" \
        "Average ratio of algorithm solution to that of the optimal (higher is worse): " + str(avgRatio) + "\n" \
        "Extra bins used compared to optimal (or lower bound): " + str(extraBins) + "\n" \
        "Time taken: " + str(timeTaken) + "out of the provided 100s (note: it is preferable to use all of the time to get better solutions, using extra time is not punished the algorithm will simply be stopped at 100s)" \
        "Please improve the code and change parameters where you see fit. The specifics of the implementation are entirely up to you but it should arrive at very strong solutions.\n" \
        "" + HARD_REQUIREMENTS


    # Prompt with the code and error in attempt to fix it
    response = client.responses.parse(
        model=openAIModel,
        input=[
            {"role": "system", "content": systemPrompt},
            {
                "role": "user",
                "content": userPrompt,
            },
        ],
        text_format=PythonCode,
    )

    event = response.output_parsed

    return event.code