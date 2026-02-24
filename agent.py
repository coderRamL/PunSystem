import os
from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import null_callback_handler
from strands.models.ollama import OllamaModel
from tools.ambiguous_words import identify_ambiguous_words

load_dotenv()

model = OllamaModel(
    model_id="gpt-oss:120b-cloud",
    host="https://ollama.com"
)

agent = Agent(
    model=model,
    tools=[identify_ambiguous_words],
    callback_handler=null_callback_handler,
    system_prompt="You are a pun assistant. Use the tool to find ambiguous words and explain the joke. If there is no joke, explain the failure in humor."
)

if __name__ == "__main__":
    pun = input("Enter a pun: ")
    # The agent calls your function, gets the dictionary, and explains it to you
    print(agent(pun))