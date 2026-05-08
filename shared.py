# FIRST OFF LOAD ENV AND GENERATE CLIENT.
from anthropic import Anthropic
from dotenv import load_dotenv
from statistics import mean

load_dotenv()
client = Anthropic()

# NOW IMPORTS
import json # JSON Parser.
import ast # For Python compile assert.
import re # Handles REG-EX

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

    * Respond only with Python, JSON or a plain Regex
    * Do not add comments, comentary or explanation
    """
    messages = []
    add_role_message(messages, 'user', prompt)
    add_role_message(messages, 'assistant', "```code")
    model = "claude-haiku-4-5"

    output = chat(messages, model, stop_sequences=["```"])
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

def validate_json(data):
    try:
        json.loads(data.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(data):
    try:
        ast.parse(data.strip())
        return 10
    except SyntaxError:
        return 0

def validate_regex(data):
    try:
        re.compile(data.strip())
        return 10
    except re.Error:
        return 0

def grade_by_code(data, test_case):
    format = test_case["format"]
    if format == "json":
        return validate_json(data)
    elif format == "python":
        return validate_python(data)
    else:
        return validate_regex(data)

def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)
    
    # Grading the output by model.
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade['score']
    reasoning = model_grade['reasoning']

    # Grading the output format and syntax by code.
    # This DOES NOT RETURN a struct, only score.
    code_score = grade_by_code(output, test_case)

    print(f"Model eval score: {model_score}")
    print(f"Code eval score: {code_score}")

    score = mean([model_score, code_score])

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
