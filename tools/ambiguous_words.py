import spacy
from typing import Annotated
from nltk.corpus import wordnet as wn
from strands import tool

# Keeping your original POS dictionary
POS_DICT = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV
}

language = spacy.load("en_core_web_sm")

# Your original helper function
def get_senses(word, pos):
    if pos in POS_DICT:
        return wn.synsets(word, pos=POS_DICT[pos])
    else: 
        return []

# Your main logic, now marked as a tool
@tool
def identify_ambiguous_words(phrase: Annotated[str, "The pun or phrase to analyze"]):
    """
    Parses senses from the text to identify ambiguous words.
    """
    amb_words = {}
    doc = language(phrase)
    for i in doc:
        if i.pos_ == "NOUN":
            senses = get_senses(i.lemma_, "NOUN")
            if len(senses) >= 2:
                # We convert synsets to definitions so the LLM can read them
                amb_words[i.text] = [s.definition() for s in senses]
    return amb_words