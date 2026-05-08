from anthropic import Anthropic
import json

max_tokens = 1000
model = "claude-sonnet-4-6"
client = Anthropic()

def add_role_message(messages, role, text):
    message = {"role": role, "content": text}
    messages.append(message)

def chat(client, messages, model="claude-sonnet-4-6",
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


def generate_dataset():
    prompt = """
    Generate a evaluation dataset for a prompt evaluation.
    Please generate 3 objects.
    """

    messages = []

    add_role_message(messages, 'user', prompt)
    add_role_message(messages, 'assistant', "```json")

    text = chat(
        client,
        messages,
        model,
        stop_sequences=["```"]
    )

    return json.loads(text)
