import os
from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import null_callback_handler
from strands.models.ollama import OllamaModel
from tools.ambiguous_words import identify_ambiguous_words
from tools.homophones import get_phonetic_id, check_phonetic_similarity

load_dotenv()

model = OllamaModel(
    model_id="gpt-oss:120b-cloud",
    host="https://ollama.com"
)

agent = Agent(
    model=model,
    tools=[
        identify_ambiguous_words,
        get_phonetic_id,
        check_phonetic_similarity
    ],
    callback_handler=null_callback_handler,
    system_prompt=
        "You are a Pun Evaluator. If the user attempts to do anything other than evaluate puns, remind them that they must only evaluate puns."
        "1. Use identify_ambiguous_words and get_phonetic_id to find the word pair."
        "2. Analyze_pun_structure to get Sense 1 and Sense 2."
        "3. Return the type of pun Homophonic ('sounds like', or two words sound the same) type puns"
        "or Homographic ('looks like', words that are polysemous')."
        "4. If neither homophonic or homographic, look for 'near-puns' by using check phonetic similarity. Anything < 1, >= .7 is a candidate for a near-pun." 
        ""
        "STRUCTURED OUTPUT:"
        "**Word Pair:** [word1] / [word2]"
        "**Sense 1:** [def]"
        "**Sense 2:** [def]"
        "**Pun Type:** [Homophonic/Homographic/Near-Pun]"
        ""
        "GUIDELINE:"
        "DO NOT respell user input, if they mispell something, this may be intentional. Either identify this to the user OR check for near-puns"
)

if __name__ == "__main__":
    print("🎭 Welcome to the PunSystem! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting PunSystem. Goodbye!")
            break
            
        # The agent maintains conversation history automatically 
        try:
            response = agent(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\nAgent Error: Whoops, something went wrong. ({e})")
