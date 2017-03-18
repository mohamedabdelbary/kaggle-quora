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


class RandomForestModel():
    n_trees = 200
    # test_size = 0.3
    rf_max_features = None
    folds = 10

    def train(self, training_df):
        """
        Expects a `features` column which holds a
        list of floats to be used as features for
        the classifier
        """
        featureMatrix, labelVector = np.array(training_df["features"]), np.array(training_df["label"])

        fpr_arrays = []
        tpr_arrays = []
        auc_list = []
        logloss_list = []

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
            fpr_arrays.append(fprArray)
            tpr_arrays.append(tprArray)
            logloss_list.append(logloss_list)

            idx += 1

        final_model = RandomForestClassifier(n_estimators=self.n_trees, max_features=self.rf_max_features, class_weight="auto")\
            if self.rf_max_features else RandomForestClassifier(n_estimators=self.n_trees, class_weight="auto")

        final_model.fit(featureMatrix, labelVector)

        roc_auc = np.mean(auc_list)
        logloss = np.mean(logloss_list)

        return {'model': final_model, 'fpr_arrays': fpr_arrays, 'tpr_arrays': tpr_arrays, 'roc_auc': roc_auc}

    def predict(self, model, prediction_records):
        """
        Adds predictions as label onto the features dataframe by applying model
            :type model
            :param model: a model to be applied on the features_df
            :type prediction_records
            :param prediction_records: a list of dicts for which we want to produce predictions (no labels exist)
            :rtype: list
            :return: a list of dicts updated with `score` and `label` fields coming back from
            model predictions
        This is a default implementation that should work fine with a wide range of models (mainly sklearn)
        It can also be overriden in implementing sub-classes
        """

        for record in records:
            prediction = model.predict_proba(record.features)
            yield {
                "features": record["features"],
                "features_dict": record["features_dict"],
                "key": record["key"],
                # "label": int(float(prediction[0][1]) > threshold),
                "score": float(prediction[0][1])
            }

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
