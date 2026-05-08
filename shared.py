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

def chat(client, messages):
  message = client.messages.create(
    model = model,
    max_tokens = max_tokens,
    messages = messages
  )
  return messages.content[0].text
