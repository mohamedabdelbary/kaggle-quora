import spacy
import re
from bs4 import UnicodeDammit

nlp = spacy.load("en")


def clean_statement(s):
    """
    Remove punctuation, stop words and standardise casing
    words
    """

    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', UnicodeDammit(str(s)).markup)
    sentence = nlp(s)
    sentence_with_stop_checks = [(sentence[i], sentence[i].is_stop) for i in range(len(sentence))]

    return sorted([str(w.lower_) for (w, stop_bool) in sentence_with_stop_checks if not stop_bool])
