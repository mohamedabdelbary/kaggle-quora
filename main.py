train_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train.csv"
test_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/test.csv"
output_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/test_predictions.csv"

import numpy as np
import pandas
from functools import partial
from nlp import features, construct_doc_list, train_lda
from model import set_overlap_score_model, binary_logloss, RandomForestModel, predict


def read_data(path):
    return pandas.read_csv(path)


if __name__ == "__main__":

    # n_sample = 100000
    # full_df = read_data(train_path)
    # rows = np.random.choice(full_df.index.values, n_sample)
    # df = full_df.ix[rows]

    df = read_data(train_path)
    test_df = read_data(test_path)

    n_lda_topics = 10
    print "Starting LDA modelling!"

    doc_list_lda_train = list(construct_doc_list(df))
    doc_list_lda_test = list(construct_doc_list(test_df))
    lda_model, id2word_dictionary, word2idx_dictionary, topics = \
        train_lda(n_lda_topics,
                  documents=doc_list_lda_train + doc_list_lda_test)

    print "<================================>"

    print "Starting feature construction!"
    feature_method = partial(
        features,
        lda_model=lda_model,
        word2idx_dict=word2idx_dictionary,
        n_lda_topics=n_lda_topics)
    df["features"] = df.apply(feature_method, axis=1)
    df["label"] = df["is_duplicate"].map(int)

    print "<=================================>"
    print "Starting model training!"
    model = RandomForestModel()
    model_obj = model.train(df, cv=False)

    print "<==================================>"
    print "Finished model training!"

    import pudb
    pudb.set_trace()

    print "Running Predictions!"
    test_df["features"] = test_df.apply(feature_method, axis=1)
    test_df["is_duplicate"] = test_df.apply(predict, axis=1)

    print "<===================================>"
    print "Finished predictions. Saving output!"

    test_df[["id", "is_duplicate"]].to_csv(output_path, index=False)

    # print "Log Loss is: {}".format(model_obj["logloss"])
