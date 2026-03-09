from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from . import prompts
import json

class PythonCode(BaseModel):
    reasoning: str = Field(description="The reasoning behind your code.")
    code: str = Field(description="The Python code.")

class PerformancePlan(BaseModel):
    reasoning: str = Field(description="The reasoning behind your plan.")
    plan: str = Field(description="The plan.")

MAX_TOKENS = 20000

openAIModel = "gpt-5.2"
openAIClient = OpenAI()
anthropicModel = "claude-opus-4-6"
anthropicClient = Anthropic()
googleModel = "gemini-3-flash-preview"
googleClient = genai.Client()

def generalOpenAIPrompt(systemPrompt, userPrompt):

    response = openAIClient.responses.parse(
        model=openAIModel,
        max_output_tokens=MAX_TOKENS,
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
    print("OpenAI")
    return event.code

def initialOpenAIPrompt(algoName):

    systemPrompt = prompts.CODE_SYSTEM_PROMPT
    userPrompt = prompts.getInitialPrompt(algoName)

    # Initial attempt to get a correct algorithm
    code = generalOpenAIPrompt(systemPrompt, userPrompt)

    return code


def correctnessOpenAIPrompt(algoName, prevCode, error):

    systemPrompt = prompts.CORRECTNESS_SYSTEM_PROMPT
    userPrompt =  prompts.getCorrectnessPrompt(algoName, prevCode, error)

    # Prompt with the code and error in attempt to fix it
    code = generalOpenAIPrompt(systemPrompt, userPrompt)

    return code


def performancePlanOpenAIPrompt(algoName, prevCode, performanceMetrics, maxTime):

    systemPrompt = prompts.PLAN_SYSTEM_PROMPT
    userPrompt = prompts.getPlanPrompt(algoName, prevCode, performanceMetrics, maxTime)

    # Prompt to get plan for improving code performance
    response = openAIClient.responses.parse(
        model=openAIModel,
        max_output_tokens=MAX_TOKENS,
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
    print("OpenAI")
    return event.plan

def applyPerformancePlanOpenAIPrompt(algoName, prevCode, performancePlan):

    systemPrompt = prompts.APPLY_PLAN_SYSTEM_PROMPT
    userPrompt = prompts.applyPlanPrompt(algoName, prevCode, performancePlan)

    # Prompt to apply performance plan
    code = generalOpenAIPrompt(systemPrompt, userPrompt)

    return code


def performanceOpenAIPrompt(algoName, prevCode, performanceMetrics, maxTime):

    systemPrompt = prompts.IMPROVEMENT_SYSTEM_PROMPT
    userPrompt =  prompts.improvePerformancePrompt(algoName, prevCode, performanceMetrics, maxTime)

    # Prompt with the code and performance results to improve performance
    code = generalOpenAIPrompt(systemPrompt, userPrompt)

    return code


# Anthropic

def generalAnthropicPrompt(systemPrompt, userPrompt):

    response = anthropicClient.messages.create(
        model=anthropicModel,
        max_tokens=MAX_TOKENS,
        system=systemPrompt,
        messages=[
            {
                "role": "user",
                "content": userPrompt
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "code": {"type": "string"},
                    },
                    "required": ["reasoning", "code"],
                    "additionalProperties": False,
                },
            }
        },
    )

    parsed = json.loads(response.content[0].text)
    print("Anthropic")
    return parsed["code"]


def initialAnthropicPrompt(algoName):

    systemPrompt = prompts.CODE_SYSTEM_PROMPT
    userPrompt = prompts.getInitialPrompt(algoName)

    code = generalAnthropicPrompt(systemPrompt, userPrompt)

    return code


def correctnessAnthropicPrompt(algoName, prevCode, error):

    systemPrompt = prompts.CORRECTNESS_SYSTEM_PROMPT
    userPrompt =  prompts.getCorrectnessPrompt(algoName, prevCode, error)

    # Prompt with the code and error in attempt to fix it
    code = generalAnthropicPrompt(systemPrompt, userPrompt)

    return code


def performancePlanAnthropicPrompt(algoName, prevCode, performanceMetrics, maxTime):

    systemPrompt = prompts.PLAN_SYSTEM_PROMPT
    userPrompt = prompts.getPlanPrompt(algoName, prevCode, performanceMetrics, maxTime)

    response = anthropicClient.messages.create(
        model=anthropicModel,
        max_tokens=MAX_TOKENS,
        system=systemPrompt,
        messages=[
            {
                "role": "user",
                "content": userPrompt
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "plan": {"type": "string"},
                    },
                    "required": ["reasoning", "plan"],
                    "additionalProperties": False,
                },
            }
        },
    )

    parsed = json.loads(response.content[0].text)
    print("Anthropic")
    return parsed["plan"]


def applyPerformancePlanAnthropicPrompt(algoName, prevCode, performancePlan):

    systemPrompt = prompts.APPLY_PLAN_SYSTEM_PROMPT
    userPrompt = prompts.applyPlanPrompt(algoName, prevCode, performancePlan)

    # Prompt to apply performance plan
    code = generalAnthropicPrompt(systemPrompt, userPrompt)

    return code


def performanceAnthropicPrompt(algoName, prevCode, performanceMetrics, maxTime):

    systemPrompt = prompts.IMPROVEMENT_SYSTEM_PROMPT
    userPrompt =  prompts.improvePerformancePrompt(algoName, prevCode, performanceMetrics, maxTime)

    # Prompt with the code and performance results to improve performance
    code = generalAnthropicPrompt(systemPrompt, userPrompt)

    return code

# Google
"""

def generalGooglePrompt(systemPrompt, userPrompt):

    response = googleClient.models.generate_content(
        model=googleModel,
        contents=userPrompt,
        config={
            "system_instruction": systemPrompt,
            "response_mime_type": "application/json",
            "response_json_schema": PythonCode.model_json_schema(),
            "max_output_tokens": MAX_TOKENS,
            "thinking_config": types.ThinkingConfig(thinking_level="minimal")
        },
    )

    print(response.text)
    fullRes = PythonCode.model_validate_json(response.text)
    code = fullRes.code

    print("Google")

    return code

def initialGooglePrompt(algoName):

    systemPrompt = prompts.CODE_SYSTEM_PROMPT
    userPrompt = prompts.getInitialPrompt(algoName)

    # Initial attempt to get a correct algorithm
    code = generalGooglePrompt(systemPrompt, userPrompt)

    return code


def correctnessGooglePrompt(algoName, prevCode, error):

    systemPrompt = prompts.CORRECTNESS_SYSTEM_PROMPT
    userPrompt =  prompts.getCorrectnessPrompt(algoName, prevCode, error)

    # Prompt with the code and error in attempt to fix it
    code = generalGooglePrompt(systemPrompt, userPrompt)

    return code

def performancePlanGooglePrompt(algoName, prevCode, performanceMetrics, maxTime):

    systemPrompt = prompts.PLAN_SYSTEM_PROMPT
    userPrompt = prompts.getPlanPrompt(algoName, prevCode, performanceMetrics, maxTime)

    response = googleClient.models.generate_content(
        model=googleModel,
        contents=userPrompt,
        config={
            "system_instruction": systemPrompt,
            "response_mime_type": "application/json",
            "response_json_schema": PerformancePlan.model_json_schema(),
            "max_output_tokens": MAX_TOKENS,
        },
    )

    fullRes = PerformancePlan.model_validate_json(response.text)
    plan = fullRes.plan

    print("Google")

    return plan


def applyPerformancePlanGooglePrompt(algoName, prevCode, performancePlan):

    systemPrompt = prompts.APPLY_PLAN_SYSTEM_PROMPT
    userPrompt = prompts.applyPlanPrompt(algoName, prevCode, performancePlan)

    # Prompt to apply performance plan
    code = generalGooglePrompt(systemPrompt, userPrompt)

    return code


def performanceGooglePrompt(algoName, prevCode, performanceMetrics, maxTime):

    systemPrompt = prompts.IMPROVEMENT_SYSTEM_PROMPT
    userPrompt =  prompts.improvePerformancePrompt(algoName, prevCode, performanceMetrics, maxTime)

    # Prompt with the code and performance results to improve performance
    code = generalGooglePrompt(systemPrompt, userPrompt)

    return code
    """