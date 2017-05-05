"""
Module for computing the question w2v vectors
taken from https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py
"""
import os
import sys
import pickle
import pandas
import numpy as np
import gensim
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk import word_tokenize
stop_words = stopwords.words('english')


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

data_path = sys.argv[1]
google_news_vectors_path = sys.argv[2]
output_path = sys.argv[3]

df = pandas.read_csv(data_path)
model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(google_news_vectors_path, binary=True)

question1_vectors = np.zeros((df.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((df.shape[0], 300))

for i, q in tqdm(enumerate(df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

import pudb
pudb.set_trace()

f_q1 = open(os.path.join(output_path, 'q1_w2v.pkl'), 'wb')
f_q2 = open(os.path.join(output_path, 'q2_w2v.pkl'), 'wb')

for q in question1_vectors:
    pickle.dump(q, f_q1, protocol=pickle.HIGHEST_PROTOCOL)

f_q1.close()

for q in question2_vectors:
    pickle.dump(q, f_q2, protocol=pickle.HIGHEST_PROTOCOL)

f_q2.close()
