train_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train.csv"
models_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/models.pkl"
train_pred_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train_preds.csv"


import pickle
import numpy as np
import pandas
from functools import partial
from nlp import features, construct_doc_list, train_lda
from model import set_overlap_score_model, binary_logloss, RandomForestModel, XgBoostModel, predict_rf


def read_data(path):
    return pandas.read_csv(path)


if __name__ == "__main__":

    # n_sample = 100000
    # full_df = read_data(train_path)
    # rows = np.random.choice(full_df.index.values, n_sample)
    # df = full_df.ix[rows]

    df = read_data(train_path)

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
        n_lda_topics=n_lda_topics)
    df = feature_method(df)
    df["label"] = df["is_duplicate"].map(int)

    print "<=================================>"
    print "Starting model training!"
    # model = RandomForestModel()
    # model_obj = model.train(df, cv=False)

    model = XgBoostModel()
    model_obj = model.train(df)

    import pudb
    pudb.set_trace()

    print "<==================================>"
    print "Finished model training!"

    print "Running predictions on training set!"

    predict_method = partial(predict, model=model_obj["model"])
    df["pred"] = df.apply(predict_method, axis=1)

    print "Saving training results!"
    models = {
        "rf": model_obj["model"],
        "lda": lda_model,
        "id2word_dict": id2word_dictionary,
        "word2idx_dict": word2idx_dictionary,
        "topics": topics
    }

    df.to_csv(train_pred_path, index=False)
    with open(models_path, 'wb') as models_file:
        pickle.dump(models, models_file, protocol=pickle.HIGHEST_PROTOCOL)

    print "Done with Training!!"
