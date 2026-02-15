import spacy
from nltk.corpus import wordnet as wn

POS_DICT = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV
}

language = spacy.load("en_core_web_sm")

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

def show_senses(amb_words):
    for i, j in amb_words.items():
        print(f"\nAmbiguous word: {i}")
        for k, l in enumerate(j[:5], 1):
            print(f"Sense {k}: {l.definition()}")

if __name__ == "__main__":
    pun = input("Enter a pun: ")
    amb_words = identify_ambiguous_words(pun)
    if amb_words:
        show_senses(amb_words)
    else:
       print("No ambiguous words found.") 
       

