import sys
import pickle
import numpy as np
import pandas
from functools import partial
from train_config import train_features
from model import binary_logloss, RandomForestModel, predict_rf, predict_xgboost


if __name__ == "__main__":
    test_set_features_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    relevant_cols = ["test_id"] + train_features
    
    print "Loading test dataset with features!"
    test_df = pandas.read_csv(test_set_features_path)[relevant_cols]
    print "<===================================>"

    print "Loading models!"
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    print  "<==================================>"
    print "Starting predictions!"

    if model['type'] == 'rf':
        predict_method = partial(predict_rf, model=model["model"], feature_cols=train_features)
        test_df["is_duplicate"] = test_df.apply(predict_method, axis=1)
    elif model['type'] == 'xgb':
        predict_method = partial(predict_xgboost, model=model["model"], feature_cols=train_features)
        test_df["is_duplicate"] = test_df.apply(predict_method, axis=1)

    print "<===================================>"
    print "Finished predictions. Saving output!"

    test_df[["test_id", "is_duplicate"]].to_csv(output_path, index=False)

    print "Done!"
