
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import os

# Set NLTK data path
local_nltk_path = r'C:\Users\KRISH\Desktop\code agent\prompt_optimizer\nltk_data'
if local_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk_path)

def get_synonyms(word, pos):
    lemmatizer = WordNetLemmatizer()
    wn_pos = None
    if pos.startswith('J'):
        wn_pos = wordnet.ADJ
    elif pos.startswith('V'):
        wn_pos = wordnet.VERB
    elif pos.startswith('N'):
        wn_pos = wordnet.NOUN

    if not wn_pos:
        return []

    lemma_word = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
    synsets = wordnet.synsets(lemma_word, pos=wn_pos)
    if not synsets:
        print(f"No synsets for {lemma_word}")
        return []

    original_synset = synsets[0]
    candidates = set()
    for syn in synsets:
        for lm in syn.lemmas():
            name = lm.name().replace('_', ' ')
            if (name.lower() != lemma_word and
                    name.lower() != word.lower() and
                    len(name) > 1 and
                    '_' not in name):
                candidates.add(name)
    
    print(f"Candidates for {word}: {candidates}")
    return list(candidates)

text = "a boy is eating banana"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

for word, pos in tagged:
    print(f"Word: {word}, POS: {pos}")
    if pos.startswith(('V', 'J')):
        syns = get_synonyms(word, pos)
        print(f"Synonyms for {word}: {syns}")
