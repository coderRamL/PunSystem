import jellyfish
from typing import Annotated
from strands import tool

@tool
def get_phonetic_id(word: Annotated[str, "The word to analyze"]):
    """
    Generates a phonetic representation of a word using the Metaphone algorithm.
    Words that sound the same (like 'hair' and 'hare') will return the same ID.
    """
    phonetic_code = jellyfish.metaphone(word)
    
    return {
        "word": word,
        "phonetic_code": phonetic_code
    }

@tool
def check_phonetic_similarity(word1: str, word2: str):
    """
    Compares two words and returns a similarity score (0 to 1).
    1.0 is a perfect match (homophone), lower is a 'near-pun'.
    """
    # Jaro-Winkler compares word similarity
    score = jellyfish.jaro_winkler_similarity(word1, word2)
    return f"Similarity Score: {round(score, 2)}"