from anthropic import Anthropic

model = "claude-sonnet-4-6";
max_tokens = 1000;
client = Anthropic(
    # This is the default and can be omitted
    # api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

def add_role_message(messages, role, text):
  message = {"role": role, "content": text}
  messages.append(message)

def chat(client, messages, system=None, temperature=1.0, stop_sequences=[]):
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
  return messages.content[0].text
