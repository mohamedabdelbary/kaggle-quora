import numpy as np
import spacy
import re
from bs4 import UnicodeDammit

from sklearn.metrics.pairwise import cosine_similarity

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

    return sorted([w for (w, stop_bool) in sentence_with_stop_checks if not stop_bool])


def features(row):
    q1, q2 = row["question1"], row["question2"]
    q1_tokens, q2_tokens = clean_statement(q1), clean_statement(q2)

    q1_doc = nlp(UnicodeDammit(' '.join([w.lemma_.lower() for w in q1_tokens])).markup) if q1_tokens else None
    q2_doc = nlp(UnicodeDammit(' '.join([w.lemma_.lower() for w in q2_tokens])).markup) if q2_tokens else None

    q1_vector, q2_vector = (
        q1_doc.vector if q1_doc and q1_doc.has_vector else None,
        q2_doc.vector if q2_doc and q2_doc.has_vector else None
    )

    q1_tokens_set = set(q1_tokens)
    q2_tokens_set = set(q2_tokens)

    token_overlap_ratio = (
        0.0 if not len(q1_tokens_set.union(q2_tokens_set))
        else 1.0 * len(q1_tokens_set.intersection(q2_tokens_set)) / len(q1_tokens_set.union(q2_tokens_set))
    )

    if q1_vector is not None and q2_vector is not None:
        dot_product = q1_vector.dot(q2_vector) 
        cosine_sim = cosine_similarity(q1_vector, q2_vector)[0][0]
        euclidean_dist = np.linalg.norm(q1_vector - q2_vector)
    else:
        dot_product = cosine_sim = euclidean_dist = 0.0

    return [token_overlap_ratio, dot_product, cosine_sim, euclidean_dist]
