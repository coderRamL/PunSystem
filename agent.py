import os
from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import null_callback_handler
from strands.models.ollama import OllamaModel
from tools.ambiguous_words import show_senses, pun_word_pair
from tools.homophones import get_phonetic_id, check_phonetic_similarity
from tools.joke_eval import evaluate_pun

load_dotenv()

model = OllamaModel(
    model_id="gpt-oss:120b-cloud",
    host="https://ollama.com"
)

agent = Agent(
    model=model,
    tools=[
        show_senses,
        pun_word_pair,
        get_phonetic_id,
        check_phonetic_similarity,
        evaluate_pun
    ],
    callback_handler=null_callback_handler,
    system_prompt=
        "You are a Pun Evaluator."
        #"1. TRIGGER CHECK: If the user input does not contain phonetic or semantic ambiguity, state: 'Your input is not a pun.' and explain."
        #"2. Use show_senses to display any words with ambigious meanings"
        "3. Use pun_word_pair and get_phonetic_id to find the word pair."
        "4. Analyze_pun_structure to get Sense 1 and Sense 2."
        "5. Return the type of pun Homophonic ('sounds like', or two words sound the same) type puns"
        "or Homographic ('looks like', words that are polysemous')."
        "6. If neither homophonic or homographic, look for 'near-puns' by using check phonetic similarity. Anything < 1, >= .7 is a candidate for a near-pun." 
        "7. Use evaluate_pun to determine whether the pun can be classified as humorous, return data and verdict. Ambiguity < .7 is NOT a pun."
        ""

        "STRUCTURED OUTPUT:"
        "**Word Pair:** [word1] / [word2]"
        "**Sense 1:** [def]"
        "**Sense 2:** [def]"
        "**Pun Type:** [Homophonic/Homographic/Near-Pun]"
        "**Humor Data:** Ambiguity <ambiguity_entropy> | Distinctness <distinctiveness_kl> "
        "**Humor Verdict**: <verdict>"
        "**Summary**: <If a pun is detected, provide an explanation of why this pun is funny, the type of pun, and use all provided data>"

        ""
        "CONSTRAINTS:"
        "DO NOT respell user input, if they mispell something, this may be intentional. Either identify this to the user OR check for near-puns."
        "Analysis should result in ONE of [VALID JOKE (Incongruity Detected), NONSENSE (Confusing/De-punned), STANDARD LANGUAGE (One story dominates)]"
        #"Users should ONLY discuss pun results or attempt to provide puns. Users may ask clarifying questions about any of the content provided by the pun system."
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

