import numpy as np
import scipy as sp
from nlp import clean_statement

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix


def set_overlap_score_model(questions_df):
    """
    appending column of set overlap percentage, where each set
    is a set of tokens excluding stop-words and punctuation,
    and in lemmatised form
    """
    def set_overlap_score(row):
        set1, set2 = \
            (set([w.lemma_.lower() for w in row["cleaned_question1_words"]]),
             set([w.lemma_.lower() for w in row["cleaned_question2_words"]]))
        return 0.0 if not len(set1.union(set2)) else 1.0 * len(set1.intersection(set2)) / len(set1.union(set2))
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


def predict(row, model):
    """
    Assumes row object has a `features` column
    with the same features as those on which
    `model` was trained
    """
    return float(model.predict_proba(np.array(row["features"]))[0][1])


class RandomForestModel():
    n_trees = 400
    # test_size = 0.3
    rf_max_features = None
    folds = 10

    def train(self, training_df, cv=True):
        """
        Expects a `features` column which holds a
        list of floats to be used as features for
        the classifier and an integer `label` column
        encoding the output to be predicted
        """
        featureMatrix, labelVector = training_df["features"], training_df["label"]
        featureMatrix = np.array([list(f) for f in featureMatrix])
        labelVector = np.array(list(labelVector))

        auc_list = []
        logloss_list = []

        if cv:
            idx = 1
            for train, test in StratifiedKFold(labelVector, self.folds):
                print "Starting Cross Validation Fold {}".format(idx)

                x_train, y_train = featureMatrix[train], labelVector[train]
                x_test, y_test = featureMatrix[test], labelVector[test]
                x_train = np.asarray(x_train)
                y_train = np.asarray(y_train)
                x_test = np.asarray(x_test)
                y_test = np.asarray(y_test)

                model = RandomForestClassifier(n_estimators=self.n_trees, max_features=self.rf_max_features, class_weight="auto")\
                    if self.rf_max_features else RandomForestClassifier(n_estimators=self.n_trees, class_weight="auto")

                model.fit(x_train, y_train)

                predictions = model.predict_proba(x_test)[:, 1]
                fprArray, tprArray, thres = roc_curve(y_test, predictions)
                roc_auc = auc(fprArray, tprArray)
                logloss = binary_logloss(y_test, predictions)
                auc_list.append(roc_auc)
                logloss_list.append(logloss_list)

                print "CV Fold result: AUC is {auc} and Log Loss is {loss}".format(auc=roc_auc, loss=logloss)
                print "#########"

                idx += 1

            model = RandomForestClassifier(n_estimators=self.n_trees, max_features=self.rf_max_features, class_weight="auto")\
                if self.rf_max_features else RandomForestClassifier(n_estimators=self.n_trees, class_weight="auto")

            roc_auc = np.mean(auc_list)
            logloss = np.mean(logloss_list)
            print "<======================================>"
            print "Finished cross validation experiments!"
            print "Average AUC is {auc} and average Log Loss is {loss}".format(auc=roc_auc, loss=logloss)
            print "Starting full model training!"

            model.fit(featureMatrix, labelVector)

            return {'model': model, 'roc_auc': roc_auc, 'logloss': logloss}
        else:
            model = RandomForestClassifier(n_estimators=self.n_trees, max_features=self.rf_max_features, class_weight="auto")\
                if self.rf_max_features else RandomForestClassifier(n_estimators=self.n_trees, class_weight="auto")

            model.fit(featureMatrix, labelVector)

            return {'model': model}

    def compute_precision_scores(self, y_pred, y_true, prob_thresholds):
        """
        Compute precision scores at different probability thresholds
        This allows us to pick a probability threshold for the classifier
        given a desired precision score
            pr = tpr  /  (tpr + fpr)
        returns: list((precision_score, prob_thres))
        """
        precisions = []
        for prob_thres in prob_thresholds:
            flagged_idxes = filter(lambda idx: y_pred[idx] >= prob_thres, range(len(y_pred)))
            true_flagged_idxes = filter(lambda idx: y_pred[idx] >= prob_thres and y_true[idx] == 1, range(len(y_pred)))
            precision = (len(true_flagged_idxes) / float(len(flagged_idxes))) if len(flagged_idxes) else 0.0
            precisions.append((precision, prob_thres))

        return sorted(precisions, key=lambda (prec, prob): prec)

    def compute_accuracy_scores(self, y_pred, y_true, prob_thresholds):
        """
        Compute accuracy scores at different probability thresholds
        This allows us to pick a probability threshold for the classifier
        given a desired precision score
        returns: list((accuracy_score, prob_thres))
        """
        accuracy_scores = []
        for prob_thres in prob_thresholds:
            correct_predicted_data_points = filter(lambda prob_idx:
                                                   (y_pred[prob_idx] >= prob_thres and y_true[prob_idx] == 1) or
                                                   (y_pred[prob_idx] < prob_thres and y_true[prob_idx] == 0),
                                                   range(len(y_pred)))
            accuracy = len(correct_predicted_data_points) / float(len(y_true)) if len(y_true) else 0.0
            accuracy_scores.append((accuracy, prob_thres))

        return sorted(accuracy_scores, key=lambda (acc, prob): acc)
