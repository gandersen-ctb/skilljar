from anthropic import Anthropic
from dotenv import load_dotenv
from statistics import mean

load_dotenv()
client = Anthropic()

import json

max_tokens = 1000
model = "claude-sonnet-4-6"

def add_role_message(messages, role, text):
    message = {"role": role, "content": text}
    messages.append(message)

def chat(messages, model="claude-sonnet-4-6",
         system=None, temperature=1.0, stop_sequences=None):

    if stop_sequences is None:
        stop_sequences = []

    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)

    return message.content[0].text

def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
    Please solve the following task:

    {test_case["task"]}
    """
    messages = []
    add_role_message(messages, 'user', prompt)
    output = chat(messages)
    return output

def grade_by_model(test_case, output):
    eval_prompt = f"""
    You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

    Original Task:
    <task>
    {test_case["task"]}
    </task>

    Solution to Evaluate:
    <solution>
    {output}
    </solution>

    Output Format
    Provide your evaluation as a structured JSON object with the following fields, in this specific order:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement
    - "reasoning": A concise explanation of your overall assessment
    - "score": A number between 1-10. To chose a score try justifying internally what the other 9 options should look like (better and worst than) and adjust so the score is fair.

    Respond with JSON. Keep your response concise and direct.
    Example response shape:
    {{
        "strengths": string[],
        "weaknesses": string[],
        "reasoning": string,
        "score": number
    }}
        """
    messages = []
    add_role_message(messages, 'user', eval_prompt)
    add_role_message(messages, 'assistant', "```json")
    model = "claude-haiku-4-5"

    eval_text = chat(messages, model, stop_sequences=["```"])
    return json.loads(eval_text)

def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)
    
    # TODO - Grading
    # score = 10
    grade = grade_by_model(test_case, output)
    score = grade['score']
    reasoning = grade['reasoning']

    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning
    }

def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case"""
    results = []
    
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    
    avg_score = mean([result["score"] for result in results])
    print(f"Average score: {avg_score}")
    
    return results
