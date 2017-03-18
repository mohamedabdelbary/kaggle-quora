train_path = "/Users/mohamedabdelbary/Documents/kaggle_quora/train.csv"

import pandas
from nlp import features
from model import set_overlap_score_model, binary_logloss, RandomForestModel


def read_data():
    return pandas.read_csv(train_path)


if __name__ == "__main__":
    df = read_data()
    # df_scored = set_overlap_score_model(df)

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
