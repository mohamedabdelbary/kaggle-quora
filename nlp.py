from __future__ import division
import numpy as np
import pandas
import spacy
import re
import math
import jellyfish
from functools import partial
from collections import Counter
from bs4 import UnicodeDammit
from itertools import permutations

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en")
stops = set(stopwords.words("english"))

question_tokens = set(["why", "how", "what", "when", "which", "who", "whose", "whom"])
common_question_tokens = set(["why", "how", "what", "when", "which", "who"])

common_q_token_pairs = [("why", "why"), ("how", "how"), ("what", "what"), ("when", "when"), ("which", "which"), ("who", "who")]
common_q_token_pairs.extend(
    list(permutations(list(common_question_tokens), 2))
)


def remove_punc(s):
    return re.sub(r'[^\w\s]', '', UnicodeDammit(str(s)).markup)


def clean_statement(s):
    """
    Remove punctuation, stop words and standardise casing
    words, and return remaining tokens
    """

    # Remove punctuation
    s = remove_punc(s).lower()
    sentence = nlp(s)
    sentence_with_stop_checks = [(sentence[i], sentence[i].is_stop) for i in range(len(sentence))]

    return sorted([w for (w, stop_bool) in sentence_with_stop_checks if not stop_bool])


def construct_doc_list(df):
    """
    Take the question pairs DF and return a list of 2 docs per
    row with the cleaned up sentence
    """
    for index, row in df.iterrows():
        q1, q2 = row["question1"], row["question2"]
        q1_tokens, q2_tokens = clean_statement(q1), clean_statement(q2)

        q1_doc = [w.lemma_.lower() for w in q1_tokens]
        q2_doc = [w.lemma_.lower() for w in q2_tokens]

        yield q1_doc
        yield q2_doc


def train_naive_bayes(documents):
    X = pandas.DataFrame(documents.text.apply(lambda x: unicode(x, errors='replace')))
    y = documents.target

    model = Pipeline([
        ('count', CountVectorizer(ngram_range=(1, 3), min_df=1)),
        ('tfidf', TfidfTransformer()),
        ('clf',   MultinomialNB(alpha=0.1)),
    ])

    model.fit(X.text, y)

    return model


def train_lda(n_topics, id2word_dictionary=None, documents=None, corpus=None):
    """
    Training method for LDA. documents is a list of lists of words/tokens
    documents is used to construct a dictionary and a corpus from which the
    topics for LDA are inferred
    """
    # Construct dictionary of words if it's not passed
    if not id2word_dictionary:
        id2word_dictionary = corpora.Dictionary(documents)

    word2idx_dictionary = dict([(w, idx) for (idx, w) in id2word_dictionary.items()])

    # Construct corpus for model
    if documents and not corpus:
        corpus = [id2word_dictionary.doc2bow(document) for document in documents]

    # Cluster the documents into topics using LDA. number of topics is given
    # by n_topics
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word_dictionary,
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

    return lda_model, id2word_dictionary, word2idx_dictionary, topics


# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1.0 / (count + eps)


def get_word_weights(df):
    questions = pandas.Series(df['question1'].tolist() + df['question2'].tolist()).astype(str)
    questions = [remove_punc(q).lower() for q in questions]
    eps = 500
    words = (" ".join(questions)).lower().split()
    counts = Counter(words)
    return {word: get_weight(count, eps=eps) for word, count in counts.items()}


def tfidf_word_match_share(row, weights):
    q1words = {}
    q2words = {}
    for word in remove_punc(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in remove_punc(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def weighted_token_overlap_score(row):
    cleaned_question1_words = clean_statement(row["question1"])
    cleaned_question2_words = clean_statement(row["question2"])
    
    set1, set2 = \
        (set([w.lemma_.lower() for w in cleaned_question1_words]),
         set([w.lemma_.lower() for w in cleaned_question2_words]))
        
    return \
    (1.0 * len(set1.intersection(set2)) / (len(set1.union(set2)) or 1)) * \
    (
        min(len(str(row["question1"])), len(str(row["question2"]))) / 
        (1.0 * max(len(str(row["question1"])), len(str(row["question2"]))))
    )
    

def stops_ratios(row):
    q1_tokens = [t.lower() for t in remove_punc(row["question1"]).split()]
    q2_tokens = [t.lower() for t in remove_punc(row["question2"]).split()]
    q1_stops = set([t for t in q1_tokens if t in stops])
    q2_stops = set([t for t in q2_tokens if t in stops])
    return (
        float(len(q1_stops.intersection(q2_stops))) / (len(q1_stops.union(q2_stops)) or 1.0),
        float(len(q1_stops)) / (len(q1_tokens) or 1.0),
        float(len(q2_stops)) / (len(q2_tokens) or 1.0),
        math.fabs(float(len(q1_stops)) / (len(q1_tokens) or 1.0) - float(len(q2_stops)) / (len(q2_tokens) or 1.0))
    )


def count_grams_full(df, n):
    c = Counter()
    for (idx, row) in list(df.iterrows()):
        q_tokens = remove_punc(str(row["question1"])).lower().split() + remove_punc(str(row["question2"])).lower().split()
        c.update(map(lambda x: '_'.join(x), zip(*[q_tokens[i:] for i in range(n)])))
        
    return c


def count_grams(input_list, n):
    """ Returns a count of n-grams """
    return Counter(map(lambda x: '_'.join(x), zip(*[input_list[i:] for i in range(n)])))


def shared_ngrams(row, n):
    """Ratio of shared n-grams to total n-grams across both questions"""
    
    q1_words, q2_words = count_grams(str(row["question1"]), n), count_grams(str(row["question2"]), n)
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0.0
        
    shared_words = [w for w in q1_words if w in q2_words]
    R = (2 * len(shared_words)) / float(len(q1_words) + len(q2_words))
    return R


def tf_idf_ngrams_match(row, weights, n=2):
    q1_words, q2_words = (
        count_grams(remove_punc(str(row["question1"])).lower().split(), n),
        count_grams(remove_punc(str(row["question2"])).lower().split(), n)
    )
    
    shared_weights = [weights.get(w, 0) for w in q1_words.keys() if w in q2_words] +\
                     [weights.get(w, 0) for w in q2_words.keys() if w in q1_words]
    total_weights = [weights.get(w, 0) for w in q1_words] + [weights.get(w, 0) for w in q2_words]
    
    R = np.sum(shared_weights) / float(np.sum(total_weights))
    return R


def noun_phrase_overlap(row):
    q1_doc = nlp(UnicodeDammit(str(row["question1"])).markup)
    q2_doc = nlp(UnicodeDammit(str(row["question2"])).markup)
    q1_np = set([noun_p.text for noun_p in q1_doc.noun_chunks])
    q2_np = set([noun_p.text for noun_p in q2_doc.noun_chunks])
    return len(q1_np.intersection(q2_np)) / (float(len(q1_np.union(q2_np))) or 1.0)


def num_noun_phrases(s):
    doc = nlp(UnicodeDammit(str(s)).markup)
    return len(set([noun_p.text for noun_p in doc.noun_chunks]))


def stops_ratio_q1_q2(row):
    q1_tokens = [t.lower() for t in remove_punc(row["question1"]).split()]
    q2_tokens = [t.lower() for t in remove_punc(row["question2"]).split()]
    q1_stops = set([t for t in q1_tokens if t in stops])
    q2_stops = set([t for t in q2_tokens if t in stops])
    return float(len(q1_stops.intersection(q2_stops))) / (len(q1_stops.union(q2_stops)) or 1.0)


def stops_ratio_q1(row):
    q1_tokens = [t.lower() for t in remove_punc(row["question1"]).split()]
    q1_stops = set([t for t in q1_tokens if t in stops])
    return float(len(q1_stops)) / (len(q1_tokens) or 1.0)


def stops_ratio_q2(row):
    q2_tokens = [t.lower() for t in remove_punc(row["question2"]).split()]
    q2_stops = set([t for t in q2_tokens if t in stops])
    return float(len(q2_stops)) / (len(q2_tokens) or 1.0)


def stops_diff(row):
    q1_tokens = [t.lower() for t in remove_punc(row["question1"]).split()]
    q2_tokens = [t.lower() for t in remove_punc(row["question2"]).split()]
    q1_stops = set([t for t in q1_tokens if t in stops])
    q2_stops = set([t for t in q2_tokens if t in stops])
    return math.fabs(float(len(q1_stops)) / (len(q1_tokens) or 1.0) - float(len(q2_stops)) / (len(q2_tokens) or 1.0))


def question_tokens_ratio(row):
    q1_quest_tokens = set([t.lower() for t in remove_punc(row["question1"]) if t.lower() in question_tokens])
    q2_quest_tokens = set([t.lower() for t in remove_punc(row["question2"]) if t.lower() in question_tokens])
    return (
        float(len(q1_quest_tokens.intersection(q2_quest_tokens))) / (len(q1_quest_tokens.union(q2_quest_tokens)) or 1.0)
    )


def lda_topic_probs_diff(row, lda_model=None, word2idx_dict={}, n_lda_topics=None):
    # LDA related features
    q1_tokens, q2_tokens = row["q1_tokens"], row["q2_tokens"]
    q1_lda_doc = [w.lemma_.lower() for w in q1_tokens]
    q2_lda_doc = [w.lemma_.lower() for w in q2_tokens]
    q1_topic_probs = dict(
        lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q1_lda_doc if w in word2idx_dict]).items())
    )
    q2_topic_probs = dict(
        lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q2_lda_doc if w in word2idx_dict]).items())
    )

    q1_topic_probs = [(t, q1_topic_probs[t]) if t in q1_topic_probs else (t, 0.0) for t in range(n_lda_topics)]
    q2_topic_probs = [(t, q2_topic_probs[t]) if t in q2_topic_probs else (t, 0.0) for t in range(n_lda_topics)]

    q1_topic_vector = np.array([prob for (topic, prob) in q1_topic_probs])
    q2_topic_vector = np.array([prob for (topic, prob) in q2_topic_probs])
    return list(q1_topic_vector - q2_topic_vector)


def token_overlap_ratio(row):
    q1_tokens, q2_tokens = row["q1_tokens"], row["q2_tokens"]
    q1_tokens_set = set([w.lemma_.lower() for w in q1_tokens])
    q2_tokens_set = set([w.lemma_.lower() for w in q2_tokens])

    return float(len(q1_tokens_set.intersection(q2_tokens_set))) / (len(q1_tokens_set.union(q2_tokens_set)) or 1.0)


def no_token_overlap(row):
    return float(row["token_overlap_ratio"] == 0.0)


def full_token_overlap(row):
    return float(row["token_overlap_ratio"] == 1.0)


def token_vector_dot_prod(row):
    q1_tokens, q2_tokens = row["q1_tokens"], row["q2_tokens"]
    q1_doc = nlp(UnicodeDammit(' '.join([w.lemma_.lower() for w in q1_tokens])).markup) if q1_tokens else None
    q2_doc = nlp(UnicodeDammit(' '.join([w.lemma_.lower() for w in q2_tokens])).markup) if q2_tokens else None

    q1_vector, q2_vector = (
        q1_doc.vector if q1_doc and q1_doc.has_vector else None,
        q2_doc.vector if q2_doc and q2_doc.has_vector else None
    )

    return q1_vector.dot(q2_vector) if q1_vector is not None and q2_vector is not None else 0.0


def token_vector_cosine_sim(row):
    q1_tokens, q2_tokens = row["q1_tokens"], row["q2_tokens"]
    q1_doc = nlp(UnicodeDammit(' '.join([w.lemma_.lower() for w in q1_tokens])).markup) if q1_tokens else None
    q2_doc = nlp(UnicodeDammit(' '.join([w.lemma_.lower() for w in q2_tokens])).markup) if q2_tokens else None

    q1_vector, q2_vector = (
        q1_doc.vector if q1_doc and q1_doc.has_vector else None,
        q2_doc.vector if q2_doc and q2_doc.has_vector else None
    )

    return cosine_similarity(q1_vector, q2_vector)[0][0] if q1_vector is not None and q2_vector is not None else 0.0


def lda_vector_dot_product(row, lda_model=None, word2idx_dict={}, n_lda_topics=None):
    q1_tokens, q2_tokens = row["q1_tokens"], row["q2_tokens"]
    q1_lda_doc = [w.lemma_.lower() for w in q1_tokens]
    q2_lda_doc = [w.lemma_.lower() for w in q2_tokens]
    q1_topic_probs = dict(
        lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q1_lda_doc if w in word2idx_dict]).items())
    )
    q2_topic_probs = dict(
        lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q2_lda_doc if w in word2idx_dict]).items())
    )

    q1_topic_probs = [(t, q1_topic_probs[t]) if t in q1_topic_probs else (t, 0.0) for t in range(n_lda_topics)]
    q2_topic_probs = [(t, q2_topic_probs[t]) if t in q2_topic_probs else (t, 0.0) for t in range(n_lda_topics)]

    q1_topic_vector = np.array([prob for (topic, prob) in q1_topic_probs])
    q2_topic_vector = np.array([prob for (topic, prob) in q2_topic_probs])

    return q1_topic_vector.dot(q2_topic_vector) if q1_topic_vector is not None and q2_topic_vector is not None else 0.0


def lda_vector_cosine_sim(row, lda_model=None, word2idx_dict={}, n_lda_topics=None):
    q1_tokens, q2_tokens = row["q1_tokens"], row["q2_tokens"]
    q1_lda_doc = [w.lemma_.lower() for w in q1_tokens]
    q2_lda_doc = [w.lemma_.lower() for w in q2_tokens]
    q1_topic_probs = dict(
        lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q1_lda_doc if w in word2idx_dict]).items())
    )
    q2_topic_probs = dict(
        lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q2_lda_doc if w in word2idx_dict]).items())
    )

    q1_topic_probs = [(t, q1_topic_probs[t]) if t in q1_topic_probs else (t, 0.0) for t in range(n_lda_topics)]
    q2_topic_probs = [(t, q2_topic_probs[t]) if t in q2_topic_probs else (t, 0.0) for t in range(n_lda_topics)]

    q1_topic_vector = np.array([prob for (topic, prob) in q1_topic_probs])
    q2_topic_vector = np.array([prob for (topic, prob) in q2_topic_probs])

    return cosine_similarity(q1_topic_vector, q2_topic_vector)[0][0] if q1_topic_vector is not None and q2_topic_vector is not None else 0.0


def q_token_pair_vars(row):
    q_token_vars = []
    for (q_token_q1, q_token_q2) in sorted(common_q_token_pairs):
        q_token_vars.append(
            float(q_token_q1 in (str(row["question1"]).lower() or '') and q_token_q2 in (str(row["question2"]).lower() or ''))
        )
    return q_token_vars


def string_subjectivity(s):
    s_text_blob = TextBlob(s.lower()) 
    return s_text_blob.subjectivity


def string_polarity(s):
    s_text_blob = TextBlob(s.lower())
    return s_text_blob.polarity


def polarity_abs_diff(row):
    q1_text_blob = TextBlob(row["q1_no_punc"].lower()) 
    q2_text_blob = TextBlob(row["q2_no_punc"].lower())
    q1_polarity = q1_text_blob.polarity
    q2_polarity = q2_text_blob.polarity
    return math.fabs(q1_polarity - q2_polarity)


def subjectivity_abs_diff(row):
    q1_text_blob = TextBlob(row["q1_no_punc"].lower()) 
    q2_text_blob = TextBlob(row["q2_no_punc"].lower())
    q1_subjectivity = q1_text_blob.subjectivity
    q2_subjectivity = q2_text_blob.subjectivity
    return math.fabs(q1_subjectivity - q2_subjectivity)


def q1_bigger_subjectivity(row):
    q1_text_blob = TextBlob(row["q1_no_punc"].lower()) 
    q2_text_blob = TextBlob(row["q2_no_punc"].lower())
    q1_subjectivity = q1_text_blob.subjectivity
    q2_subjectivity = q2_text_blob.subjectivity
    return float(q1_subjectivity > q2_subjectivity)


def equal_subjectivity(row):
    q1_text_blob = TextBlob(row["q1_no_punc"].lower()) 
    q2_text_blob = TextBlob(row["q2_no_punc"].lower())
    q1_subjectivity = q1_text_blob.subjectivity
    q2_subjectivity = q2_text_blob.subjectivity
    return float(q1_subjectivity == q2_subjectivity)


def q1_bigger_polarity(row):
    q1_text_blob = TextBlob(row["q1_no_punc"].lower()) 
    q2_text_blob = TextBlob(row["q2_no_punc"].lower())
    q1_polarity = q1_text_blob.polarity
    q2_polarity = q2_text_blob.polarity
    return float(q1_polarity > q2_polarity)


def equal_polarity(row):
    q1_text_blob = TextBlob(row["q1_no_punc"].lower()) 
    q2_text_blob = TextBlob(row["q2_no_punc"].lower())
    q1_polarity = q1_text_blob.polarity
    q2_polarity = q2_text_blob.polarity
    return float(q1_polarity == q2_polarity)


def opposite_polarity(row):
    q1_text_blob = TextBlob(row["q1_no_punc"].lower()) 
    q2_text_blob = TextBlob(row["q2_no_punc"].lower())
    q1_polarity = q1_text_blob.polarity
    q2_polarity = q2_text_blob.polarity
    return float(np.sign(q1_polarity) != np.sign(q2_polarity))


def jaro_winkler_sim(row):
    jw = jellyfish.jaro_winkler(
        UnicodeDammit(str(row["question1"])).markup.lower(),
        UnicodeDammit(str(row["question2"])).markup.lower()
    )

    if type(jw) == tuple:
        jw = jw[0]

    return jw


def levenshtein_dist(row):
    return jellyfish.levenshtein_distance(
        UnicodeDammit(str(row["question1"])).markup.lower(),
        UnicodeDammit(str(row["question2"])).markup.lower()
    )


def hamming_dist(row):
    return jellyfish.hamming_distance(
        UnicodeDammit(str(row["question1"])).markup.lower(),
        UnicodeDammit(str(row["question2"])).markup.lower()
    )


def inverse_hamming_dist(row):
    return 1.0 / jellyfish.hamming_distance(
        UnicodeDammit(str(row["question1"])).markup.lower(),
        UnicodeDammit(str(row["question2"])).markup.lower()
    )


def compute_features(df, lda_model, word2idx_dict, n_lda_topics=10, word_weights={}):
    df["q1_tokens"] = df["question1"].apply(clean_statement)
    df["q2_tokens"] = df["question2"].apply(clean_statement)

    df["q1_no_punc"] = df["question1"].apply(remove_punc)
    df["q2_no_punc"] = df["question2"].apply(remove_punc)

    df["len_q1_no_punc"] = df["q1_no_punc"].apply(lambda s: len(s))
    df["len_q2_no_punc"] = df["q2_no_punc"].apply(lambda s: len(s))

    lda_meth = partial(lda_topic_probs_diff, lda_model=lda_model, word2idx_dict=word2idx_dict, n_lda_topics=n_lda_topics)
    
    # Vector feature
    df["diff_lda_topic_vector"] = df.apply(lda_meth, axis=1)

    df["token_overlap_ratio"] = df.apply(token_overlap_ratio, axis=1)
    df["no_token_overlap"] = df.apply(no_token_overlap, axis=1)
    df["full_token_overlap"] = df.apply(full_token_overlap, axis=1)

    df["token_vector_dot_product"] = df.apply(token_vector_dot_prod, axis=1)
    df["token_vector_cosine_sim"] = df.apply(token_vector_cosine_sim, axis=1)

    lda_vector_dot_product_meth = partial(lda_vector_dot_product, lda_model=lda_model, word2idx_dict=word2idx_dict, n_lda_topics=n_lda_topics)
    lda_vector_dot_cosine_sim_meth = partial(lda_vector_cosine_sim, lda_model=lda_model, word2idx_dict=word2idx_dict, n_lda_topics=n_lda_topics)
    df["lda_vector_dot_product"] = df.apply(lda_vector_dot_product_meth, axis=1)
    df["lda_vector_cosine_sim"] = df.apply(lda_vector_dot_cosine_sim_meth, axis=1)

    # TF-IDF sim and n-gram analysis
    tfidf_word_match_share_meth = partial(tfidf_word_match_share, weights=word_weights[1])
    df["tf_idf_word_match_share"] = df.apply(tfidf_word_match_share_meth, axis=1)

    # n-gram analysis
    df["shared_2_gram_ratio"] = df.apply(partial(shared_ngrams, n=2), axis=1)
    df["shared_3_gram_ratio"] = df.apply(partial(shared_ngrams, n=3), axis=1)
    df["shared_4_gram_ratio"] = df.apply(partial(shared_ngrams, n=4), axis=1)
    df["shared_5_gram_ratio"] = df.apply(partial(shared_ngrams, n=5), axis=1)
    df["shared_6_gram_ratio"] = df.apply(partial(shared_ngrams, n=6), axis=1)
        
    # n-gram TF-IDF sim
    df["two_gram_tfidf_sim"] = df.apply(partial(tf_idf_ngrams_match, weights=word_weights[2], n=2), axis=1)
    df["three_gram_tfidf_sim"] = df.apply(partial(tf_idf_ngrams_match, weights=word_weights[3], n=3), axis=1)

    # token overlap weighted by min_q_length / max_q_length
    df["weighted_token_overlap_score"] = df.apply(weighted_token_overlap_score, axis=1)
    
    # Stop word occurrence
    df["stops_ratio_q1_q2"] = df.apply(stops_ratio_q1_q2, axis=1)
    df["stops_ratio_q1"] = df.apply(stops_ratio_q1, axis=1)
    df["stops_ratio_q2"] = df.apply(stops_ratio_q2, axis=1)
    df["stops_diff"] = df.apply(stops_diff, axis=1)

    # Question token pair vars
    # Vector feature
    df["q_token_pair_vars"] = df.apply(q_token_pair_vars, axis=1)
    
    # Basic Sentiment analysis
    df["q1_subjectivity"] = df["q1_no_punc"].apply(string_subjectivity)
    df["q2_subjectivity"] = df["q2_no_punc"].apply(string_subjectivity)

    df["q1_polarity"] = df["q1_no_punc"].apply(string_polarity)
    df["q2_polarity"] = df["q2_no_punc"].apply(string_polarity)

    df["polarity_abs_diff"] = df.apply(polarity_abs_diff, axis=1)
    df["subjectivity_abs_diff"] = df.apply(subjectivity_abs_diff, axis=1)

    df["q1_bigger_subjectivity"] = df.apply(q1_bigger_subjectivity, axis=1)
    df["equal_subjectivity"] = df.apply(equal_subjectivity, axis=1)
    df["q1_bigger_polarity"] = df.apply(q1_bigger_polarity, axis=1)
    df["equal_polarity"] = df.apply(equal_polarity, axis=1)
    df["opposite_polarity"] = df.apply(opposite_polarity, axis=1)
    
    # Noun phrases
    df["noun_phrase_overlap"] = df.apply(noun_phrase_overlap, axis=1)
    df["q1_num_noun_phrases"] = df["question1"].apply(num_noun_phrases)
    df["q2_num_noun_phrases"] = df["question2"].apply(num_noun_phrases)
    
    # name similarity metrics
    df["jaro_winkler_sim"] = df.apply(jaro_winkler_sim, axis=1)
    df["levenshtein_dist"] = df.apply(levenshtein_dist, axis=1)
    df["hamming_dist"] = df.apply(hamming_dist, axis=1)
    df["inverse_hamming_dist"] = df.apply(inverse_hamming_dist, axis=1)

    return df


def features(df, lda_model, word2idx_dict, n_lda_topics=10, word_weights={}, naive_bayes_models={}):
    """
    More features to implement:
    - TF-IDF or similar scheme string similarity (with and without stopwords)
    - Better LDA model by incorporating children, synonyms, related concepts, subtrees
    - Difference in lengths between both questions, ratio of lengths
        - for full original questions
        - noun phrases
        - after filtering stopwords
    - Number of sentences in both questions, ratios, difference in number
    - Stop words in both questions, stopq1/len(q1), stopq2/len(q2), stopq1.intersect(stopq2),...

    - Common Bigrams/Trigrams
    - Country specific features: countries or locations mentioned in both questions
    - More features from LDA model topic probability vectors
        - Appending both vectors
        - [p1/p2 for (p1,p2) in zip(vector_1, vector2)]
        - cosine sim
    - Features specific to each question separately
        - Length of q1
        - Length of q2
        - # sentences in q1
        - # sentences in q2
        - # words in q1
        - # words in q2
    - Question tokens in both questions (why, how, when, what, ..): count in each q, set intersection, difference, etc
    - Naive encoding of question word rules as boolean vars
        - Is "what" in q1 and in q2?
        - Is "what" in q1 and "how" in q2?
        ... repeat for the most common 6 tokens "why", "how", "what", "when", "which", "who"
    - Sentiment analysis (see https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis) -> **should provide BIG boost**
    """

    features_col = pandas.Series([[]], index=np.arange(df.shape[0]))

    for (idx, row) in list(df.iterrows()):
        q1, q2 = row["question1"], row["question2"]
        q1_tokens, q2_tokens = clean_statement(q1), clean_statement(q2)
        q1_no_punc, q2_no_punc = remove_punc(q1), remove_punc(q2)

        # LDA related features
        q1_lda_doc = [w.lemma_.lower() for w in q1_tokens]
        q2_lda_doc = [w.lemma_.lower() for w in q2_tokens]
        q1_topic_probs = dict(
            lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q1_lda_doc if w in word2idx_dict]).items())
        )
        q2_topic_probs = dict(
            lda_model.get_document_topics(Counter([word2idx_dict[w] for w in q2_lda_doc if w in word2idx_dict]).items())
        )

        q1_topic_probs = [(t, q1_topic_probs[t]) if t in q1_topic_probs else (t, 0.0) for t in range(n_lda_topics)]
        q2_topic_probs = [(t, q2_topic_probs[t]) if t in q2_topic_probs else (t, 0.0) for t in range(n_lda_topics)]

        q1_topic_vector = np.array([prob for (topic, prob) in q1_topic_probs])
        q2_topic_vector = np.array([prob for (topic, prob) in q2_topic_probs])
        diff_topic_vector = q1_topic_vector - q2_topic_vector

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
        
        # TF-IDF sim
        tf_idf_sim = tfidf_word_match_share(row, word_weights[1])

        # token overlap weighted by min_q_length / max_q_length
        wt_token_overlap_score = weighted_token_overlap_score(row)
        
        # Stop word occurrence
        (stops_ratio, stops_ratio_q1, stops_ratio_q2, stops_diff) = stops_ratios(row)

        # Question token pair vars
        q_token_vars = []
        for (q_token_q1, q_token_q2) in sorted(common_q_token_pairs):
            q_token_vars.append(
                float(q_token_q1 in (str(q1).lower() or '') and q_token_q2 in (str(q2).lower() or ''))
            )
            
        # Basic Sentiment analysis
        q1_text_blob = TextBlob(q1_no_punc.lower()) 
        q2_text_blob = TextBlob(q2_no_punc.lower())
        q1_polarity = q1_text_blob.polarity
        q2_polarity = q2_text_blob.polarity
        q1_subjectivity = q1_text_blob.subjectivity
        q2_subjectivity = q2_text_blob.subjectivity
        polarity_abs_diff = math.fabs(q1_polarity - q2_polarity)
        subjectivity_abs_diff = math.fabs(q1_subjectivity - q2_subjectivity)
        q1_bigger_subjectivity = float(q1_subjectivity > q2_subjectivity)
        equal_subjectivity = float(q1_subjectivity == q2_subjectivity)
        q1_bigger_polarity = float(q1_polarity > q2_polarity)
        equal_polarity = float(q1_polarity == q2_polarity)
        opposite_polarity = float(np.sign(q1_polarity) != np.sign(q2_polarity))
        
        # Noun phrases
        n_phrase_overlap = noun_phrase_overlap(row)
        q1_doc = nlp(UnicodeDammit(str(q1)).markup)
        q2_doc = nlp(UnicodeDammit(str(q2)).markup)
        q1_np = set([noun_p.text for noun_p in q1_doc.noun_chunks])
        q2_np = set([noun_p.text for noun_p in q2_doc.noun_chunks])
        
        # name similarity metrics
        jaro_winkler_sim = jellyfish.jaro_winkler(UnicodeDammit(str(q1)).markup.lower(), UnicodeDammit(str(q2)).markup.lower()),
        levenshtein_dist = jellyfish.levenshtein_distance(UnicodeDammit(str(q1)).markup.lower(), UnicodeDammit(str(q2)).markup.lower()),
        hamming_dist = jellyfish.hamming_distance(UnicodeDammit(str(q1)).markup.lower(), UnicodeDammit(str(q2)).markup.lower())
        
        # n-gram analysis
        shared_n_gram_vars = []
        for n in range(2, 8):
            shared_n_gram_vars.append(shared_ngrams(row, n))
            
        # n-gram TF-IDF sim
        two_gram_tfidf_sim = tf_idf_ngrams_match(row, word_weights[2], n=2)
        three_gram_tfidf_sim = tf_idf_ngrams_match(row, word_weights[3], n=3)
        
        # Prob vectors of question classification based on NB models
        # trained on the Univ Illinois dataset
        fine_grained_nb = naive_bayes_models["fine_grained"]
        coarse_grained_nb = naive_bayes_models["coarse_grained"]
        
        # Fine grained classification model
        try:
            p_q1_fine_grained_vec = fine_grained_nb.predict_proba([q1_no_punc.lower()])[0]
            p_q2_fine_grained_vec = fine_grained_nb.predict_proba([q2_no_punc.lower()])[0]

            diff_fine_grained_nb_vec = list(np.abs(p_q1_fine_grained_vec - p_q2_fine_grained_vec))
        except ValueError:
            diff_fine_grained_nb_vec = [1.0] * len(fine_grained_nb.classes_)
        
        # Coarse grained classification model
        try:
            p_q1_coarse_grained_vec = coarse_grained_nb.predict_proba([q1_no_punc.lower()])[0]
            p_q2_coarse_grained_vec = coarse_grained_nb.predict_proba([q2_no_punc.lower()])[0]

            diff_coarse_grained_nb_vec = list(np.abs(p_q1_coarse_grained_vec - p_q2_coarse_grained_vec))
        except ValueError:
            diff_coarse_grained_nb_vec = [1.0] * len(coarse_grained_nb.classes_)
        
        if q1_vector is not None and q2_vector is not None:
            dot_product = q1_vector.dot(q2_vector) 
            cosine_sim = cosine_similarity(q1_vector, q2_vector)[0][0]
            euclidean_dist = np.linalg.norm(q1_vector - q2_vector)
            euclidean_lda_probs_dist = np.linalg.norm(diff_topic_vector)
        else:
            dot_product = cosine_sim = 0.0
            euclidean_dist = euclidean_lda_probs_dist = 100.0 # Not a very good hack
            
        if type(jaro_winkler_sim) == tuple:
            jaro_winkler_sim = jaro_winkler_sim[0]
            
        feature_list = [
            token_overlap_ratio,
            float(token_overlap_ratio == 0),
            float(token_overlap_ratio == 1),
            dot_product,
            cosine_sim,
            # euclidean_dist,
            # euclidean_lda_probs_dist,
            tf_idf_sim,
            wt_token_overlap_score,
            stops_ratio,
            stops_ratio_q1,
            stops_ratio_q2,
            stops_diff,
            q1_polarity,
            q2_polarity,
            q1_subjectivity,
            q2_subjectivity,
            q1_bigger_subjectivity,
            equal_subjectivity,
            q1_bigger_polarity,
            equal_polarity,
            # polarity_abs_diff,
            subjectivity_abs_diff,
            opposite_polarity,
            len(q1_no_punc),
            len(q2_no_punc),
            len(q1_lda_doc),
            len(q2_lda_doc),
            n_phrase_overlap,
            len(q1_np),
            len(q2_np),
            two_gram_tfidf_sim,
            three_gram_tfidf_sim,
            jaro_winkler_sim,
            1 / float(hamming_dist or 1.0)
        ]
        feature_list.extend(q_token_vars)
        feature_list.extend(shared_n_gram_vars)
        feature_list.extend(diff_fine_grained_nb_vec)
        feature_list.extend(diff_coarse_grained_nb_vec)
        # feature_list.extend(list(diff_vector))  # Wasn't a good feature. Could be the way it was constructed
        feature_list.extend(list(diff_topic_vector))

        # return feature_list
        features_col[idx] = feature_list

    df["features"] = features_col
    return df
