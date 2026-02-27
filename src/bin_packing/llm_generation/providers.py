from openai import OpenAI
from pydantic import BaseModel
import prompts

class PythonCode(BaseModel):
    reasoning: str
    code: str

class PerformancePlan(BaseModel):
    reasoning: str
    plan: str

openAIModel = "gpt-5-nano"

def initialOpenAIPrompt(algoName):
    client = OpenAI()

    systemPrompt = prompts.CODE_SYSTEM_PROMPT
    userPrompt = prompts.getInitialPrompt(algoName)

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

    systemPrompt = prompts.CORRECTNESS_SYSTEM_PROMPT
    userPrompt =  prompts.getCorrectnessPrompt(algoName, prevCode, error)

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

def performancePlanOpenAIPrompt(algoName, prevCode, performanceMetrics):
    client = OpenAI()

    systemPrompt = prompts.PLAN_SYSTEM_PROMPT
    userPrompt = prompts.getPlanPrompt(algoName, prevCode, performanceMetrics)

    # Prompt to get plan for improving code performance
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
        text_format=PerformancePlan,
    )

    event = response.output_parsed

    return event.code

def applyPerformancePlanOpenAIPrompt(algoName, prevCode, performancePlan):
    client = OpenAI()

    systemPrompt = prompts.APPLY_PLAN_SYSTEM_PROMPT
    userPrompt = prompts.applyPlanPrompt(algoName, prevCode, performancePlan)

    # Prompt to apply performance plan
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

def performanceOpenAIPrompt(algoName, prevCode, performanceMetrics):
    client = OpenAI()

    systemPrompt = prompts.IMPROVEMENT_SYSTEM_PROMPT
    userPrompt =  prompts.improvePerformancePrompt(algoName, prevCode, performanceMetrics)


    # Prompt with the code and performance results to improve performance
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