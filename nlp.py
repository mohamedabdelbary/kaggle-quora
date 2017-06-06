from __future__ import division
import numpy as np
import pandas
import spacy
import re
import math
import jellyfish
from fuzzywuzzy import fuzz
from functools import partial
from collections import Counter, defaultdict
from bs4 import UnicodeDammit
from itertools import permutations

from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from textblob import TextBlob
from pattern.en import parse

from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

base_data_dir = "/Users/mohamedabdelbary/Documents/kaggle_quora/"
# google_news_vectors_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/GoogleNews-vectors-negative300.bin.gz"
# google_news_vec_model = KeyedVectors.load_word2vec_format(google_news_vectors_path, binary=True)


nlp = spacy.load("en")
stops = set(stopwords.words("english"))

question_tokens = set(["why", "how", "what", "when", "which", "who", "whose", "whom"])
common_question_tokens = set(["why", "how", "what", "when", "which", "who"])

common_q_token_pairs = [("why", "why"), ("how", "how"), ("what", "what"), ("when", "when"), ("which", "which"), ("who", "who")]
common_q_token_pairs.extend(
    list(permutations(list(common_question_tokens), 2))
)

noun_tags = set(['NN', 'NNS', 'NNP', 'NNPS'])
verb_tags = set(['VB', 'VBZ', 'VBP', 'VBN', 'VBG'])
adj_tags = set(['JJ', 'JJR', 'JJS'])
adv_tags = set(['RB', 'RBR', 'RBS'])
general_tags = set(['CC', 'DT', 'EX', 'IN'])
pronoun_tags = set(['PRP'])


# Leaky questions analysis (see https://www.kaggle.com/act444/lb-0-158-xgb-handcrafted-leaky)
# df_train = pandas.read_csv(base_data_dir + 'train.csv')
# df_train = df_train.fillna(' ')

# df_test = pandas.read_csv(base_data_dir + 'test.csv')
# ques = pandas.concat([df_train[['question1', 'question2']], \
#     df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
# q_dict = defaultdict(set)
# for i in range(ques.shape[0]):
#     q_dict[ques.question1[i]].add(ques.question2[i])
#     q_dict[ques.question2[i]].add(ques.question1[i])


# del df_train
# del df_test


def q1_freq(row):
    return(len(q_dict[row['question1']]))


def q2_freq(row):
    return(len(q_dict[row['question2']]))


def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


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
    return 1.0 / (
        jellyfish.hamming_distance(
        UnicodeDammit(str(row["question1"])).markup.lower(),
        UnicodeDammit(str(row["question2"])).markup.lower())
        or
        1.0
    )


def wmd(row, model):
    s1 = str(row["q1_no_punc"]).lower().split()
    s2 = str(row["q2_no_punc"]).lower().split()
    s1 = [w for w in s1 if w not in stops]
    s2 = [w for w in s2 if w not in stops]
    return model.wmdistance(s1, s2)


def punc_overlap(row):
    return (1.0 * len(row["q1_punc"].intersection(row["q2_punc"]))) / (len(row["q1_punc"].union(row["q2_punc"])) or 1.0)


def standard_token_overlap_ratio(row):
    q1_tokens, q2_tokens = row["q1_standard_tokens"], row["q2_standard_tokens"]

    return float(len(q1_tokens.intersection(q2_tokens))) / (len(q1_tokens.union(q2_tokens)) or 1.0)


def parse_structure(s):
    if not isinstance(s, str):
        return []
    parts = parse(s, relations=True).split()
    return parts[0] if parts else []


def gen_token_overlap(token_set_1, token_set_2):
    return float(len(token_set_1.intersection(token_set_2))) / (len(token_set_1.union(token_set_2)) or 1.0)


def weighted_gen_token_overlap_score(token_set_1, token_set_2, row):
    return float(len(token_set_1.intersection(token_set_2))) / (len(token_set_1.union(token_set_2)) or 1.0) * \
        (
            min(len(str(row["question1"])), len(str(row["question2"]))) / 
            (1.0 * max(len(str(row["question1"])), len(str(row["question2"]))))
        )


def compute_features(df, lda_model, word2idx_dict, n_lda_topics=10, word_weights={}, q_vectors={}):
    """
    TODO:
    - extract nouns, pronouns, verbs, propositions and compute
        - shared nouns (singular), verbs (standardise tense, say present), object, pronouns, etc
        - # of nouns q1
        - # of nouns q2
        - abs(# nouns q1 - # nouns q2)
        - same above 3 features for pronouns, verbs, etc
        - Are objects the same?
    - fuzzy and distance features from
    https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py
    - Vector features from ^
    - Geography specific features: countries/cities in both questions
    - Predictions from naive bayes questions classifier (for the U-Illinois labelled question set)
    - Variables from questions without cleaning
    - Punctuation set overlap

    - Num commas, periods, question marks, exclamation marks in each question
    - Noun/Verb/Object/Pronoun overlap
    """
    # df["q1_tokens"] = df["question1"].apply(clean_statement)
    # df["q2_tokens"] = df["question2"].apply(clean_statement)

    # df["q1_no_punc"] = df["question1"].apply(remove_punc)
    # df["q2_no_punc"] = df["question2"].apply(remove_punc)

    # df["len_q1_no_punc"] = df["q1_no_punc"].apply(lambda s: len(s))
    # df["len_q2_no_punc"] = df["q2_no_punc"].apply(lambda s: len(s))

    # lda_meth = partial(lda_topic_probs_diff, lda_model=lda_model, word2idx_dict=word2idx_dict, n_lda_topics=n_lda_topics)
    
    # # Vector feature
    # df["diff_lda_topic_vector"] = df.apply(lda_meth, axis=1)

    # # Expand vector feature into multiple columns
    # diff_lda_topic_vector = df['diff_lda_topic_vector'].apply(pandas.Series)

    # # rename each variable is tags
    # diff_lda_topic_vector = diff_lda_topic_vector.rename(columns=lambda x : 'diff_lda_topic_vector_' + str(x))

    # df = pandas.concat([df[:], diff_lda_topic_vector[:]], axis=1)

    # df["token_overlap_ratio"] = df.apply(token_overlap_ratio, axis=1)
    # df["no_token_overlap"] = df.apply(no_token_overlap, axis=1)
    # df["full_token_overlap"] = df.apply(full_token_overlap, axis=1)

    # df["token_vector_dot_product"] = df.apply(token_vector_dot_prod, axis=1)
    # df["token_vector_cosine_sim"] = df.apply(token_vector_cosine_sim, axis=1)

    # lda_vector_dot_product_meth = partial(lda_vector_dot_product, lda_model=lda_model, word2idx_dict=word2idx_dict, n_lda_topics=n_lda_topics)
    # lda_vector_dot_cosine_sim_meth = partial(lda_vector_cosine_sim, lda_model=lda_model, word2idx_dict=word2idx_dict, n_lda_topics=n_lda_topics)
    # df["lda_vector_dot_product"] = df.apply(lda_vector_dot_product_meth, axis=1)
    # df["lda_vector_cosine_sim"] = df.apply(lda_vector_dot_cosine_sim_meth, axis=1)

    # # TF-IDF sim and n-gram analysis
    # tfidf_word_match_share_meth = partial(tfidf_word_match_share, weights=word_weights[1])
    # df["tf_idf_word_match_share"] = df.apply(tfidf_word_match_share_meth, axis=1)

    # # n-gram analysis
    # df["shared_2_gram_ratio"] = df.apply(partial(shared_ngrams, n=2), axis=1)
    # df["shared_3_gram_ratio"] = df.apply(partial(shared_ngrams, n=3), axis=1)
    # df["shared_4_gram_ratio"] = df.apply(partial(shared_ngrams, n=4), axis=1)
    # df["shared_5_gram_ratio"] = df.apply(partial(shared_ngrams, n=5), axis=1)
    # df["shared_6_gram_ratio"] = df.apply(partial(shared_ngrams, n=6), axis=1)
        
    # # n-gram TF-IDF sim
    # df["two_gram_tfidf_sim"] = df.apply(partial(tf_idf_ngrams_match, weights=word_weights[2], n=2), axis=1)
    # df["three_gram_tfidf_sim"] = df.apply(partial(tf_idf_ngrams_match, weights=word_weights[3], n=3), axis=1)

    # # token overlap weighted by min_q_length / max_q_length
    # df["weighted_token_overlap_score"] = df.apply(weighted_token_overlap_score, axis=1)
    
    # # Stop word occurrence
    # df["stops_ratio_q1_q2"] = df.apply(stops_ratio_q1_q2, axis=1)
    # df["stops_ratio_q1"] = df.apply(stops_ratio_q1, axis=1)
    # df["stops_ratio_q2"] = df.apply(stops_ratio_q2, axis=1)
    # df["stops_diff"] = df.apply(stops_diff, axis=1)

    # # Question token pair vars
    # # Vector feature
    # df["q_token_pair_vars"] = df.apply(q_token_pair_vars, axis=1)

    # # Expand vector feature into multiple columns
    # q_token_pair_vars_series = df['q_token_pair_vars'].apply(pandas.Series)

    # # rename each variable is tags
    # q_token_pair_vars_series = q_token_pair_vars_series.rename(columns=lambda x : 'q_token_pair_vars_' + str(x))

    # df = pandas.concat([df[:], q_token_pair_vars_series[:]], axis=1)
    
    # # Basic Sentiment analysis
    # df["q1_subjectivity"] = df["q1_no_punc"].apply(string_subjectivity)
    # df["q2_subjectivity"] = df["q2_no_punc"].apply(string_subjectivity)

    # df["q1_polarity"] = df["q1_no_punc"].apply(string_polarity)
    # df["q2_polarity"] = df["q2_no_punc"].apply(string_polarity)

    # df["polarity_abs_diff"] = df.apply(polarity_abs_diff, axis=1)
    # df["subjectivity_abs_diff"] = df.apply(subjectivity_abs_diff, axis=1)

    # df["q1_bigger_subjectivity"] = df.apply(q1_bigger_subjectivity, axis=1)
    # df["equal_subjectivity"] = df.apply(equal_subjectivity, axis=1)
    # df["q1_bigger_polarity"] = df.apply(q1_bigger_polarity, axis=1)
    # df["equal_polarity"] = df.apply(equal_polarity, axis=1)
    # df["opposite_polarity"] = df.apply(opposite_polarity, axis=1)
    
    # # Noun phrases
    # df["noun_phrase_overlap"] = df.apply(noun_phrase_overlap, axis=1)
    # df["q1_num_noun_phrases"] = df["question1"].apply(num_noun_phrases)
    # df["q2_num_noun_phrases"] = df["question2"].apply(num_noun_phrases)
    
    # # name similarity metrics
    # df["jaro_winkler_sim"] = df.apply(jaro_winkler_sim, axis=1)
    # df["levenshtein_dist"] = df.apply(levenshtein_dist, axis=1)
    # df["hamming_dist"] = df.apply(hamming_dist, axis=1)
    # df["inverse_hamming_dist"] = df.apply(inverse_hamming_dist, axis=1)

    # # fuzzy features (from https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py)
    # df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    # df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    # wmd_meth = partial(wmd, model=google_news_vec_model)
    # df['wmd'] = df.apply(wmd_meth, axis=1)

    # google_news_vec_model.init_sims(replace=True)
    # wmd_meth = partial(wmd, model=google_news_vec_model)
    # df['norm_wmd'] = df.apply(wmd_meth, axis=1)

    # question1_vectors = q_vectors["q1"]
    # question2_vectors = q_vectors["q2"]

    # df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                         np.nan_to_num(question2_vectors))]

    # df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                               np.nan_to_num(question2_vectors))]

    # df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                           np.nan_to_num(question2_vectors))]

    # df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                             np.nan_to_num(question2_vectors))]

    # df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                               np.nan_to_num(question2_vectors))]

    # df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                  np.nan_to_num(question2_vectors))]

    # df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
    #                                                                 np.nan_to_num(question2_vectors))]

    # df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    # df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    # df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    # df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

    # # Previously computed features without cleaning up strings
    # # Just lower casing
    # df["q1_standard"] = df["question1"].apply(lambda x: str(x).lower())
    # df["q2_standard"] = df["question2"].apply(lambda x: str(x).lower())

    # df["q1_standard_tokens"] = df["q1_standard"].apply(lambda x: set(x.split(" ")))
    # df["q2_standard_tokens"] = df["q2_standard"].apply(lambda x: set(x.split(" ")))

    # df["standard_token_overlap_ratio"] = df.apply(standard_token_overlap_ratio, axis=1)

    # df["len_q1_standard"] = df["q1_standard"].apply(lambda x: len(x))
    # df["len_q2_standard"] = df["q2_standard"].apply(lambda x: len(x))

    # # Get punctuation marks in questions
    # df["q1_punc"] = df["q1_standard"].apply(lambda x: set(re.findall(r'[^\w\s]', x)))
    # df["q2_punc"] = df["q2_standard"].apply(lambda x: set(re.findall(r'[^\w\s]', x)))

    # # Punctuation features
    # df["len_q1_punc"] = df["q1_punc"].apply(lambda x: len(x))
    # df["len_q2_punc"] = df["q2_punc"].apply(lambda x: len(x))
    # df["punc_overlap"] = df.apply(punc_overlap, axis=1)

    # df["q1_structure_parts"] = df["q1_standard"].apply(lambda x: parse_structure(x))
    # df["q2_structure_parts"] = df["q2_standard"].apply(lambda x: parse_structure(x))

    # df["q1_nouns"] = df["q1_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in noun_tags]))
    # df["q2_nouns"] = df["q2_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in noun_tags]))

    # df["q1_verbs"] = df["q1_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in verb_tags]))
    # df["q2_verbs"] = df["q2_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in verb_tags]))

    # df["q1_adjs"] = df["q1_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in adj_tags]))
    # df["q2_adjs"] = df["q2_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in adj_tags]))

    # df["q1_advs"] = df["q1_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in adv_tags]))
    # df["q2_advs"] = df["q2_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in adv_tags]))

    # df["q1_gen_tags"] = df["q1_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in general_tags]))
    # df["q2_gen_tags"] = df["q2_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in general_tags]))

    # df["q1_pronouns"] = df["q1_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in pronoun_tags]))
    # df["q2_pronouns"] = df["q2_structure_parts"].apply(lambda x: set([n[0] for n in x if n[1] in pronoun_tags]))

    # df["q1_num_nouns"] = df["q1_nouns"].apply(lambda x: len(x))
    # df["q2_num_nouns"] = df["q2_nouns"].apply(lambda x: len(x))
    # df["noun_overlap"] = df.apply(lambda r: gen_token_overlap(r["q1_nouns"], r["q2_nouns"]), axis=1)
    # df["weighted_noun_overlap"] = df.apply(lambda r: weighted_gen_token_overlap_score(r["q1_nouns"], r["q2_nouns"], r), axis=1)
    # df["noun_sets_equal"] = df.apply(lambda r: float(r["noun_overlap"] == 1.0), axis=1)

    # df["q1_num_verbs"] = df["q1_verbs"].apply(lambda x: len(x))
    # df["q2_num_verbs"] = df["q2_verbs"].apply(lambda x: len(x))
    # df["verb_overlap"] = df.apply(lambda r: gen_token_overlap(r["q1_verbs"], r["q2_verbs"]), axis=1)
    # df["weighted_verb_overlap"] = df.apply(lambda r: weighted_gen_token_overlap_score(r["q1_verbs"], r["q2_verbs"], r), axis=1)
    # df["verb_sets_equal"] = df.apply(lambda r: float(r["verb_overlap"] == 1.0), axis=1)

    # df["q1_num_adjs"] = df["q1_adjs"].apply(lambda x: len(x))
    # df["q2_num_adjs"] = df["q2_adjs"].apply(lambda x: len(x))
    # df["adj_overlap"] = df.apply(lambda r: gen_token_overlap(r["q1_adjs"], r["q2_adjs"]), axis=1)
    # df["weighted_adj_overlap"] = df.apply(lambda r: weighted_gen_token_overlap_score(r["q1_adjs"], r["q2_adjs"], r), axis=1)
    # df["adj_set_equal"] = df.apply(lambda r: float(r["adj_overlap"] == 1.0), axis=1)

    # df["q1_num_advs"] = df["q1_advs"].apply(lambda x: len(x))
    # df["q2_num_advs"] = df["q2_advs"].apply(lambda x: len(x))
    # df["adv_overlap"] = df.apply(lambda r: gen_token_overlap(r["q1_advs"], r["q2_advs"]), axis=1)
    # df["weighted_adv_overlap"] = df.apply(lambda r: weighted_gen_token_overlap_score(r["q1_advs"], r["q2_advs"], r), axis=1)
    # df["adv_set_equal"] = df.apply(lambda r: float(r["adv_overlap"] == 1.0), axis=1)

    # df["q1_num_gen_tags"] = df["q1_gen_tags"].apply(lambda x: len(x))
    # df["q2_num_gen_tags"] = df["q2_gen_tags"].apply(lambda x: len(x))
    # df["gen_tag_overlap"] = df.apply(lambda r: gen_token_overlap(r["q1_gen_tags"], r["q2_gen_tags"]), axis=1)
    # df["weighted_gen_tag_overlap"] = df.apply(lambda r: weighted_gen_token_overlap_score(r["q1_gen_tags"], r["q2_gen_tags"], r), axis=1)
    # df["gen_tag_set_equal"] = df.apply(lambda r: float(r["gen_tag_overlap"] == 1.0), axis=1)

    # df["q1_num_pronouns"] = df["q1_pronouns"].apply(lambda x: len(x))
    # df["q2_num_pronouns"] = df["q2_pronouns"].apply(lambda x: len(x))
    # df["pronoun_overlap"] = df.apply(lambda r: gen_token_overlap(r["q1_pronouns"], r["q2_pronouns"]), axis=1)
    # df["weighted_pronoun_overlap"] = df.apply(lambda r: weighted_gen_token_overlap_score(r["q1_pronouns"], r["q2_pronouns"], r), axis=1)
    # df["pronoun_set_equal"] = df.apply(lambda r: float(r["pronoun_overlap"] == 1.0), axis=1)

    # Leaky features analysis
    df['q1_q2_intersect'] = df.apply(q1_q2_intersect, axis=1, raw=True)
    df['q1_freq'] = df.apply(q1_freq, axis=1, raw=True)
    df['q2_freq'] = df.apply(q2_freq, axis=1, raw=True)

    return df
