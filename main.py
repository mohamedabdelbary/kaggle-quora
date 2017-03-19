train_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train.csv"

import numpy as np
import pandas
from nlp import features, construct_doc_list, train_lda
from model import set_overlap_score_model, binary_logloss, RandomForestModel


def read_data():
    return pandas.read_csv(train_path)


if __name__ == "__main__":

    n_sample = 100000
    full_df = read_data()
    rows = np.random.choice(full_df.index.values, n_sample)
    df = full_df.ix[rows]
    # df_scored = set_overlap_score_model(df)

    print "Starting LDA modelling!"

    doc_list_lda = list(construct_doc_list(df))
    lda_model, dictionary, topics = train_lda(10, documents=doc_list_lda)

    import pudb
    pudb.set_trace()

    print "Starting feature construction!"
    df["features"] = df.apply(features, axis=1)
    df["label"] = df["is_duplicate"].map(int)

    print "<=================>"
    print "Starting model training!"
    model = RandomForestModel()
    model_obj = model.train(df)

    # act, pred = list(df["label"]), list(df["score"])
    # logloss = binary_logloss(act, pred)

    print "<================>"
    print "Finished model training!"

    import pudb
    pudb.set_trace()

    print "Log Loss is: {}".format(model_obj["logloss"])
