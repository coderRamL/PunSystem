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

language = spacy.load("en_core_web_sm")

@tool
def identify_ambiguous_words(phrase: Annotated[str, "The full pun or sentence to analyze"]):
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