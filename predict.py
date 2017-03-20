test_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/test.csv"
models_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/models.pkl"
output_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/test_predictions.csv"


import pickle
import numpy as np
import pandas
from functools import partial
from nlp import features, construct_doc_list, train_lda
from model import set_overlap_score_model, binary_logloss, RandomForestModel, predict


if __name__ == "__main__":
    test_df = pandas.read_csv(test_path)
    print "Loading models!"
    with open(models_path, 'rb') as models_file:
        models = pickle.load(models_file)

    print "Starting feature construction!"
    feature_method = partial(
        features,
        lda_model=models["lda"],
        word2idx_dict=models["word2idx_dict"],
        n_lda_topics=len(models["topics"]))
    test_df["features"] = test_df.apply(feature_method, axis=1)

    print  "<==================================>"
    print "Starting predicitons!"

    predict_method = partial(predict, model=model_obj["model"])
    test_df["is_duplicate"] = test_df.apply(predict_method, axis=1)

    print "<===================================>"
    print "Finished predictions. Saving output!"

    test_df[["id", "is_duplicate"]].to_csv(output_path, index=False)

    print "Done!"
