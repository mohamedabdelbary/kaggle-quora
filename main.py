train_path = "/Users/mohamedabdelbary/Documents/train_kaggle_quora/train.csv"

import pandas
from model import set_overlap_score_model, binary_logloss


def read_data():
    return pandas.read_csv(train_path)


if __name__ == "__main__":
    df = read_data()
    df_scored = set_overlap_score_model(df)
    df["label"] = df["label"].map(int)

    import pudb
    pudb.set_trace()

    act, pred = list(df["label"]), list(df["set_overlap_score"])
    logloss = binary_logloss(act, pred)

    pudb.set_trace()
