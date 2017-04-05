import numpy as np
import pandas
import spacy
import re
import math
from collections import Counter
from bs4 import UnicodeDammit

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en")
stops = set(stopwords.words("english"))

question_tokens = set(["why", "how", "what", "when", "which", "who", "whose", "whom"])


def remove_punc(s):
    return re.sub(r'[^\w\s]', '', UnicodeDammit(str(s)).markup)


def clean_statement(s):
    """
    Remove punctuation, stop words and standardise casing
    words, and return remaining tokens
    """

    # Remove punctuation
    s = remove_punc(s)
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
    # count_vect = CountVectorizer(ngram_range=(1, 5), min_df=1)
    # counts = count_vect.fit_transform(documents.text.values)
    # model = MultinomialNB()
    # model.fit(counts, documents.target.ravel())

    # return count_vect, model

    X = pandas.DataFrame(documents.text)
    y = pandas.DataFrame(documents.target)

    model = Pipeline([
        ('count', CountVectorizer(ngram_range=(1, 3), min_df=1)),
        ('tfidf', TfidfTransformer()),
        ('clf',   MultinomialNB(alpha=0.1)),
    ])

    model.fit(X.text, y.values)

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


def question_tokens_ratio(row):
    q1_quest_tokens = set([t.lower() for t in remove_punc(row["question1"]) if t.lower() in question_tokens])
    q2_quest_tokens = set([t.lower() for t in remove_punc(row["question2"]) if t.lower() in question_tokens])
    return (
        float(len(q1_quest_tokens.intersection(q2_quest_tokens))) / (len(q1_quest_tokens.union(q2_quest_tokens)) or 1.0)
    )


def features(df, lda_model, word2idx_dict, n_lda_topics=10, word_weights={}):
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
        - # sentences in q2
        - # words in q1
        - # words in q2
    - Question tokens in both questions (why, how, when, what, ..): count in each q, set intersection, difference, etc
    - Naive encoding of question word rules as boolean vars
        - Is "what" in q1 and in q2?
        - Is "what" in q1 and "how" in q2?
        ... repeat for the most common 6 tokens "why", "how", "what", "when", "which", "who"
    """

    features_col = pandas.Series([[]], index=np.arange(df.shape[0]))

    for (idx, row) in list(df.iterrows()):
        q1, q2 = row["question1"], row["question2"]
        q1_tokens, q2_tokens = clean_statement(q1), clean_statement(q2)

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
        tf_idf_sim = tfidf_word_match_share(row, word_weights)

        # token overlap weighted by min_q_length / max_q_length
        wt_token_overlap_score = weighted_token_overlap_score(row)
        
        # Stop word occurrence
        (stops_ratio, stops_ratio_q1, stops_ratio_q2, stops_diff) = stops_ratios(row)

        if q1_vector is not None and q2_vector is not None:
            dot_product = q1_vector.dot(q2_vector) 
            cosine_sim = cosine_similarity(q1_vector, q2_vector)[0][0]
            euclidean_dist = np.linalg.norm(q1_vector - q2_vector)
            euclidean_lda_probs_dist = np.linalg.norm(diff_topic_vector)
        else:
            dot_product = cosine_sim = 0.0
            euclidean_dist = euclidean_lda_probs_dist = 100.0 # Not a very good hack

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
            stops_diff
        ]
        feature_list.extend(list(diff_topic_vector))

        # return feature_list
        features_col[idx] = feature_list

    df["features"] = features_col
    return df
