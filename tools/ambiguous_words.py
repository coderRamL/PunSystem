import spacy
from typing import Annotated
from nltk.corpus import wordnet as wn
from strands import tool

POS_DICT = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV
}

language = spacy.load("en_core_web_md")

def get_senses(word, pos):
    if pos in POS_DICT:
        return wn.synsets(word, pos=POS_DICT[pos])
    else: 
        return []

def identify_ambiguous_words(phrase):
    amb_words = {}
    doc = language(phrase)
    for i in doc:
        if i.pos_ == "NOUN":
            senses = get_senses(i.lemma_, "NOUN")
            if len(senses) >= 2:
                amb_words[i.text] = senses
    return amb_words

@tool
def show_senses(amb_words): #original show senses function for user clarity
    senses=[]
    for i, j in amb_words.items():
        print(f"\nAmbiguous word: {i}")
        for k, l in enumerate(j[:5], 1):
            print(f"Sense {k}: {l.definition()}")

@tool
def pun_word_pair(phrase: Annotated[str, "The full pun or sentence to analyze"]): #final pun word pair
    """
    Analyzes a phrase to return word pairs, their distinct senses, and the pun type.
    Detects Homographic (look alike) puns.
    """
    doc = language(phrase)
    results = []

    for token in doc:
        if token.pos_ not in ["NOUN", "VERB", "ADJ"]:
            continue

        word = token.text.lower()
        lemma = token.lemma_.lower()
        
        #Check for Homographic Puns (Multiple senses for the same spelling)
        senses = wn.synsets(lemma)
        if len(senses) >= 2:
            results.append({
                "word_pair": (word, word),
                "wordsense1": senses[0].definition(),
                "wordsense2": senses[1].definition(),
                "pun_type": "Homographic (Polysemy)",
                "target": word
            })
    
    return results

