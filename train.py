train_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train.csv"
models_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/models_v1_with_oversampling.pkl"
lda_model_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/lda_model.pkl"
train_pred_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train_preds.csv"


import sys
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

    # n_sample = 200000
    # full_df = read_data(train_path)
    # rows = np.random.choice(full_df.index.values, n_sample)
    # df = full_df.ix[rows]

    train_set_features_path = sys.argv[1]
    df = read_data(train_set_features_path)

    print "<=================================>"
    print "Starting model training!"

    model_type = sys.argv[2]
    if model_type == 'rf':
        model = RandomForestModel()
        model_obj = model.train(df, cv=False)

    elif model_type == 'xgb':
        model = XgBoostModel()
        model_obj = model.train(df)

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
