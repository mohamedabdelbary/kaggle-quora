train_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train.csv"
models_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/models_v1_with_oversampling.pkl"
train_pred_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train_preds.csv"


import pickle
import numpy as np
import pandas
from functools import partial
from collections import Counter
from nlp import features, construct_doc_list, train_lda, get_word_weights
from model import set_overlap_score_model, binary_logloss, RandomForestModel, XgBoostModel, predict_rf


def read_data(path):
    return pandas.read_csv(path)


def oversample_non_duplicates(df):
    # Purely for experimenting!! This oversampling process can lead to overfitting
    # and is generally not very good ML practise
    pos_train = df[df["is_duplicate"] == 1]
    neg_train = df[df["is_duplicate"] == 0]

    # Now we oversample the negative class
    # There is likely a much more elegant way to do this...
    p = 0.165
    scale = ((float(len(pos_train)) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pandas.concat([neg_train, neg_train])
        scale -=1
    neg_train = pandas.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print len(pos_train) / float(len(pos_train) + len(neg_train))

    df_resampled = pandas.concat([pos_train, neg_train])
    del pos_train, neg_train

    return df_resampled


if __name__ == "__main__":

    n_sample = 200000
    full_df = read_data(train_path)
    rows = np.random.choice(full_df.index.values, n_sample)
    df = full_df.ix[rows]

    # df = read_data(train_path)

    # Resampling process to oversample negative cases
    print "Oversampling the majority class to match test set!"
    df = oversample_non_duplicates(df)

    # Get word weights based on counts
    weights = get_word_weights(df)

    n_lda_topics = 20
    print "Starting LDA modelling!"

    doc_list_lda_train = list(construct_doc_list(df))
    lda_model, id2word_dictionary, word2idx_dictionary, topics = \
        train_lda(n_lda_topics,
                  documents=doc_list_lda_train)

    print "<================================>"

    print "Starting feature construction!"
    feature_method = partial(
        features,
        lda_model=lda_model,
        word2idx_dict=word2idx_dictionary,
        n_lda_topics=n_lda_topics,
        word_weights=weights)
    df = feature_method(df)
    df["label"] = df["is_duplicate"].map(int)

    print "<=================================>"
    print "Starting model training!"
    model = RandomForestModel()
    model_obj = model.train(df, cv=False)

    # model = XgBoostModel()
    # model_obj = model.train(df)

    print "<==================================>"
    print "Finished model training!"

    print "Running predictions on training set!"

    # predict_method = partial(predict, model=model_obj["model"])
    # df["pred"] = df.apply(predict_method, axis=1)

    print "Saving training results!"
    models = {
        "rf": model_obj["model"],
        "lda": lda_model,
        "id2word_dict": id2word_dictionary,
        "word2idx_dict": word2idx_dictionary,
        "topics": topics,
        "word_weights": weights
    }

    # df.to_csv(train_pred_path, index=False)
    with open(models_path, 'wb') as models_file:
        pickle.dump(models, models_file, protocol=pickle.HIGHEST_PROTOCOL)

    print "Done with Training!!"
