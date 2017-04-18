lda_model_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/lda_model.pkl"

import sys
import pickle
import numpy as np
import pandas
from functools import partial
from collections import Counter
from nlp import construct_doc_list, train_lda, get_word_weights
from model import set_overlap_score_model, binary_logloss, RandomForestModel, XgBoostModel, NeuralNetModel

from train_config import train_features


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
        scale -= 1
    neg_train = pandas.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print len(pos_train) / float(len(pos_train) + len(neg_train))

    df_resampled = pandas.concat([pos_train, neg_train])
    del pos_train, neg_train

    return df_resampled


if __name__ == "__main__":

    train_set_features_path = sys.argv[1]
    models_path = sys.argv[2]

    # n_sample = 200000
    # full_df = read_data(train_set_features_path)
    # rows = np.random.choice(full_df.index.values, n_sample)
    # df = full_df.ix[rows]

    df = read_data(train_set_features_path)
    df = oversample_non_duplicates(df)

    print "<=================================>"
    print "Starting model training!"

    model_type = sys.argv[3]
    if model_type == 'rf':
        model = RandomForestModel()
        # model_obj = model.train(df, cv=False)
        model_obj = model.train(df, feature_cols=train_features)

    elif model_type == 'xgb':
        model = XgBoostModel()
        model_obj = model.train(df, feature_cols=train_features)

    elif model_type == 'nnet':
        model = NeuralNetModel()
        model_obj = model.train(df, feature_cols=train_features)

    print "<==================================>"
    print "Finished model training!"

    print "Saving training results!"
    models = {
        model_type: model_obj["model"],
    }

    # df.to_csv(train_pred_path, index=False)
    with open(models_path, 'wb') as models_file:
        pickle.dump(models, models_file, protocol=pickle.HIGHEST_PROTOCOL)

    print "Done with Training!!"
