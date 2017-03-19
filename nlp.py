import numpy as np
import spacy
import re
from collections import Counter
from bs4 import UnicodeDammit

from gensim import corpora
from gensim.models.ldamodel import LdaModel

from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en")


def clean_statement(s):
    """
    Remove punctuation, stop words and standardise casing
    words, and return remaining tokens
    """

    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', UnicodeDammit(str(s)).markup)
    sentence = nlp(s)
    sentence_with_stop_checks = [(sentence[i], sentence[i].is_stop) for i in range(len(sentence))]

    return sorted([w for (w, stop_bool) in sentence_with_stop_checks if not stop_bool])


def construct_doc_list(df):
    """
    Take the question pair df's and return a list of 2 docs per
    row with the cleaned up sentence
    """
    for index, row in df.iterrows():
        q1, q2 = row["question1"], row["question2"]
        q1_tokens, q2_tokens = clean_statement(q1), clean_statement(q2)

        q1_doc = [UnicodeDammit(w.lemma_.lower()).markup for w in q1_tokens]
        q2_doc = [UnicodeDammit(w.lemma_.lower()).markup for w in q2_tokens]

        yield q1_doc
        yield q2_doc


def train_lda(n_topics, dictionary=None, documents=None, corpus=None):
    """
    Training method for LDA. documents is a list of lists of words/tokens
    documents is used to construct a dictionary and a corpus from which the
    topics for LDA are inferred
    """
    # Construct dictionary of words if it's not passed
    if not dictionary:
        dictionary = corpora.Dictionary(documents)

    # Construct corpus for model
    if documents and not corpus:
        corpus = [dictionary.doc2bow(document) for document in documents]

    # Cluster the documents into topics using LDA. number of topics is given
    # by n_topics
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=n_topics,
                         update_every=1,
                         chunksize=10000,
                         passes=1)

    """
    Default value for topn (number of top words to show by probability) is 10.
    A high enough value should return the words covering most or all of the
    probability mass
    """
    topics = [lda_model.show_topic(idx, topn=50000)
              for idx in range(0, n_topics)]

    return lda_model, dictionary, topics


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
        else 1.0 * float(len(q1_tokens_set.intersection(q2_tokens_set))) / len(q1_tokens_set.union(q2_tokens_set))
    )

    if q1_vector is not None and q2_vector is not None:
        dot_product = q1_vector.dot(q2_vector) 
        cosine_sim = cosine_similarity(q1_vector, q2_vector)[0][0]
        euclidean_dist = np.linalg.norm(q1_vector - q2_vector)
    else:
        dot_product = cosine_sim = euclidean_dist = 0.0

    return [token_overlap_ratio, dot_product, cosine_sim, euclidean_dist]
