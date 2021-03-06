lda_model_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/lda_model.pkl"
word_weights_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/word_weights.pkl"


import os
import sys
import pickle
import cPickle
import numpy as np
import pandas
from functools import partial
from collections import Counter
from nlp import compute_features, get_word_weights, remove_punc, get_weight, count_grams_full


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    num_split = int(sys.argv[3])

    full_df = pandas.read_csv(input_path)

    with open(lda_model_path, 'rb') as lda_model_file:
        lda_model = pickle.load(lda_model_file)

    lda_model, word2idx_dictionary, n_lda_topics = lda_model["lda_model"], lda_model["word2idx_dict"], len(lda_model["topics"])

    with open(word_weights_path, 'rb') as word_weights_file:
        ngram_weights = pickle.load(word_weights_file)

    try:
        google_vectors_path = sys.argv[4]
        google_q_vectors = {"q1": [], "q2": []}
        q1 = []
        with open(os.path.join(google_vectors_path, 'q1_w2v.pkl'), 'rb') as f:
            try:
                while True:
                    google_q_vectors["q1"].append(cPickle.load(f))
            except EOFError:
                pass

        with open(os.path.join(google_vectors_path, 'q2_w2v.pkl'), 'rb') as f:
            try:
                while True:
                    google_q_vectors["q2"].append(cPickle.load(f))
            except EOFError:
                pass

        # google_q_vectors = {
        #     "q1": pickle.load(open(os.path.join(google_vectors_path, 'q1_w2v.pkl'), 'rb')),
        #     "q2": pickle.load(open(os.path.join(google_vectors_path, 'q2_w2v.pkl'), 'rb'))
        # }
    except Exception:
        google_q_vectors = {}

    print "Splitting dataset"
    idx = 0
    for df in np.array_split(full_df, num_split):
        chunk_path = input_path.split(".")[0] + str(idx) + "." + input_path.split(".")[-1]
        df.to_csv(chunk_path, index=False)
        idx += 1

    del df
    del full_df

    feature_method = partial(
        compute_features,
        lda_model=lda_model,
        word2idx_dict=word2idx_dictionary,
        n_lda_topics=n_lda_topics,
        word_weights=ngram_weights)

    print "Starting feature construction!"
    df_idx = 0
    for idx in range(num_split):
        print "DF chunk %s" % idx
        chunk_input_path = input_path.split(".")[0] + str(idx) + "." + input_path.split(".")[-1]
        df = pandas.read_csv(chunk_input_path)
        df["label"] = df["is_duplicate"].map(int)
        google_q_vectors_subset = {
            "q1": google_q_vectors["q1"][df_idx: df_idx + df.shape[0]],
            "q2": google_q_vectors["q2"][df_idx: df_idx + df.shape[0]]
        }
        feature_method = partial(feature_method, q_vectors=google_q_vectors_subset)
        df = feature_method(df)

        chunk_output_path = output_path.split(".")[0] + str(idx) + "." + output_path.split(".")[-1]
        df.to_csv(chunk_output_path, index=False)

        df_idx = df_idx + df.shape[0]

        # Removing temp input file
        os.remove(chunk_input_path)

    print "Finished feature construction!"
    print "Saving output!"

    print "<===============================>"

    print "Combining feature DF's!"
    outpath_list = [output_path.split(".")[0] + str(idx) + "." + output_path.split(".")[-1] for idx in range(num_split)]

    full_df_features = pandas.concat([pandas.read_csv(df_path) for df_path in outpath_list])

    print "Removing remaining temp files!"
    for p in outpath_list:
        os.remove(p)

    print "Saving full output!"
    full_df_features.to_csv(output_path, index=False)
