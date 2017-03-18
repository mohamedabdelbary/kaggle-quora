import scipy as sp
from nlp import clean_statement


def set_overlap_score_model(questions_df):
    def set_overlap_score(row_1, row_2):
        set1, set2 = set(row_1["cleaned_question1_words"]), set(row_2["cleaned_question2_words"])
        return 1.0 * len(set1.intersection(set2)) / len(set1.union(set2))
    questions_df["cleaned_question1_words"] = questions_df["question1"].map(clean_statement)
    questions_df["cleaned_question2_words"] = questions_df["question2"].map(clean_statement)

    questions_df["score"] = questions_df.apply(set_overlap_score, axis=1)

    return questions_df


def binary_logloss(act, pred):
    """
    act and pred are vectors of actual class
    and prediction probability of class 1,
    respectively
    """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll
