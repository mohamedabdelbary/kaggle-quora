import numpy as np
import scipy as sp
import pandas
from nlp import clean_statement

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix

import xgboost as xgb

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


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


def xgboost_eval(act, pred):
    return 'error', binary_logloss(act, pred)


def predict_rf(row, model, feature_cols=["features"]):
    """
    Assumes row object has a `features` column
    with the same features as those on which
    `model` was trained
    """
    f = np.array(row["features"])
    f = np.nan_to_num(f)
    return float(model.predict_proba(f)[0][1])


def predict_xgboost(row, model, feature_cols=["features"]):
    """
    Assumes row object has a `features` column
    with the same features as those on which
    `model` was trained
    """
    row_df = pandas.DataFrame([row[feature_cols]])
    row_df.fillna(value=0.0, inplace=True)

    # row_array = np.array(row_df)
    # row_array = np.nan_to_num(row_array)

    return float(model.predict(xgb.DMatrix(row_df))[0])


def updownsampling(X_train, y_train):
    if type(X_train) != pandas.DataFrame:
        pos_train = pandas.DataFrame(X_train[y_train == 1])
        neg_train = pandas.DataFrame(X_train[y_train == 0])
    else:
        pos_train = X_train[y_train == 1]
        neg_train = X_train[y_train == 0]
    X_train = pandas.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
    y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    print(np.mean(y_train))
    del pos_train, neg_train

    # pos_valid = pandas.DataFrame(X_valid[y_valid == 1])
    # neg_valid = pandas.DataFrame(X_valid[y_valid == 0])
    # X_valid = pandas.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    # y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    # print(np.mean(y_valid))
    # del pos_valid, neg_valid

    return X_train, y_train


class XgBoostModel():
    n_boost_rounds = 2000
    max_depth = 6
    objective = 'binary:logistic'
    eval_metric = "logloss"
    early_stopping_rounds = 70
    folds = 10
    learning_rate = 0.05
    scale_pos_weight = 1
    gamma = 0.3
    subsample = 0.6

    def train(self, training_df, feature_cols=["features"], cv=True):
        featureMatrix, labelVector = training_df[feature_cols], training_df["label"]
        labelVector = np.array(list(labelVector))
        labelVector = np.nan_to_num(labelVector)

        auc_list = []
        logloss_list = []

        if cv:
            idx = 1
            featureMatrix = np.array([list(f) for f in featureMatrix.values])
            featureMatrix = np.nan_to_num(featureMatrix)
            for train, test in StratifiedKFold(labelVector, self.folds):
                print "Starting Cross Validation Fold {}".format(idx)

                x_train, y_train = featureMatrix[train], labelVector[train]
                x_test, y_test = featureMatrix[test], labelVector[test]

                x_train, y_train = updownsampling(x_train, y_train)
                x_test, y_test = updownsampling(x_test, y_test)

                x_train = np.asarray(x_train)
                y_train = np.asarray(y_train)
                x_test = np.asarray(x_test)
                y_test = np.asarray(y_test)

                params = {}
                params['objective'] = self.objective
                params['eval_metric'] = self.eval_metric
                params['eta'] = self.learning_rate
                params['max_depth'] = self.max_depth
                params['scale_pos_weight'] = self.scale_pos_weight
                params['gamma'] = self.gamma
                params['silent'] = 1
                # params['seed'] = 40 + idx
                params['subsample'] = self.subsample

                d_train = xgb.DMatrix(x_train, label=y_train)
                d_valid = xgb.DMatrix(x_test, label=y_test)
                
                watchlist = [(d_train, 'train'), (d_valid, 'valid')]
                
                model = xgb.train(
                    params,
                    d_train,
                    self.n_boost_rounds,
                    watchlist,
                    early_stopping_rounds=self.early_stopping_rounds
                )

                x_test_df = pandas.DataFrame(x_test, columns=["feature_%s" % str(i) for i in range(x_test.shape[1])])
                predictions = model.predict(xgb.DMatrix(x_test_df))

                fprArray, tprArray, thres = roc_curve(y_test, predictions)
                roc_auc = auc(fprArray, tprArray)
                logloss = binary_logloss(y_test, predictions)
                auc_list.append(roc_auc)
                logloss_list.append(logloss_list)

                print "CV Fold result: AUC is {auc} and Log Loss is {loss}".format(auc=roc_auc, loss=logloss)
                print "#########"

                idx += 1

            roc_auc = np.mean(auc_list)
            logloss = np.mean(logloss_list)
            print "<======================================>"
            print "Finished cross validation experiments!"
            print "Average AUC is {auc} and average Log Loss is {loss}".format(auc=roc_auc, loss=logloss)
            print "Starting full model training!"

            return {'model': model, 'roc_auc': roc_auc, 'logloss': logloss, 'type': 'xgb', 'features': feature_cols}
        else:
            params = {}
            params['objective'] = self.objective
            params['eval_metric'] = self.eval_metric
            params['eta'] = self.learning_rate
            params['max_depth'] = self.max_depth
            params['scale_pos_weight'] = self.scale_pos_weight
            params['gamma'] = self.gamma
            params['subsample'] = self.subsample
            params['silent'] = 1

            # featureMatrix.fillna(value=0.0, inplace=True)
            featureMatrix = np.array([list(f) for f in featureMatrix.values])
            featureMatrix = np.nan_to_num(featureMatrix)
            featureMatrix, labelVector = updownsampling(featureMatrix, labelVector)      
            d_train = xgb.DMatrix(featureMatrix, label=labelVector)
            d_valid = xgb.DMatrix(featureMatrix, label=labelVector)

            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            
            model = xgb.train(
                params,
                d_train,
                self.n_boost_rounds,
                watchlist,
                early_stopping_rounds=self.early_stopping_rounds
            )

            # xgb.plot_importance(model)

            return {'model': model, 'type': 'xgb', 'features': feature_cols}


class RandomForestModel():
    n_trees = 400
    # test_size = 0.3
    rf_max_features = None
    folds = 10

    def train(self, training_df, feature_cols=["features"], cv=True):
        """
        Expects a `features` column which holds a
        list of floats to be used as features for
        the classifier and an integer `label` column
        encoding the output to be predicted
        """
        featureMatrix, labelVector = training_df[feature_cols].values, training_df["label"]
        featureMatrix = np.array([list(f) for f in featureMatrix])
        featureMatrix = np.nan_to_num(featureMatrix)
        labelVector = np.array(list(labelVector))
        labelVector = np.nan_to_num(labelVector)

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

            return

            model = RandomForestClassifier(n_estimators=self.n_trees, max_features=self.rf_max_features, class_weight="auto")\
                if self.rf_max_features else RandomForestClassifier(n_estimators=self.n_trees, class_weight="auto")

            roc_auc = np.mean(auc_list)
            logloss = np.mean(logloss_list)
            print "<======================================>"
            print "Finished cross validation experiments!"
            print "Average AUC is {auc} and average Log Loss is {loss}".format(auc=roc_auc, loss=logloss)
            print "Starting full model training!"

            model.fit(featureMatrix, labelVector)

            return {'model': model, 'roc_auc': roc_auc, 'logloss': logloss, 'type': 'rf', 'features': feature_cols}
        else:
            model = RandomForestClassifier(n_estimators=self.n_trees, max_features=self.rf_max_features, class_weight="auto")\
                if self.rf_max_features else RandomForestClassifier(n_estimators=self.n_trees, class_weight="auto")

            model.fit(featureMatrix, labelVector)

            return {'model': model, 'type': 'rf', 'features': feature_cols}

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


class NeuralNetModel():
    folds = 10

    def train(self, training_df, feature_cols=["features"], cv=True):
        featureMatrix, labelVector = training_df[feature_cols].values, training_df["label"]
        featureMatrix = np.array([list(f) for f in featureMatrix])
        featureMatrix = np.nan_to_num(featureMatrix)
        labelVector = np.array(list(labelVector))
        labelVector = np.nan_to_num(labelVector)

        featureMatrix -= np.mean(featureMatrix, axis=0)
        featureMatrix /= np.std(featureMatrix, axis=0)

        featureMatrix = np.nan_to_num(featureMatrix)

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

                model = Sequential()
                model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer='normal', activation='sigmoid'))
                model.add(Dropout(0.5))
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.5))
                # model.add(Dense(8, activation='relu'))
                # model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))

                # early_stopping = keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=0, verbose=0, mode='auto')

                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(x_train, y_train, epochs=200, batch_size=10000) #, callbacks=[early_stopping])

                scores = model.evaluate(x_test, y_test)
                for m_idx in range(len(model.metrics_names)):
                    print("\n%s: %.2f" % (model.metrics_names[m_idx], scores[m_idx]))

                # predictions = model.predict_proba(x_test)[:, 1]
                # fprArray, tprArray, thres = roc_curve(y_test, predictions)
                # roc_auc = auc(fprArray, tprArray)
                # logloss = binary_logloss(y_test, predictions)
                # auc_list.append(roc_auc)
                # logloss_list.append(logloss_list)

                # print "CV Fold result: AUC is {auc} and Log Loss is {loss}".format(auc=roc_auc, loss=logloss)
                print "#########"

                idx += 1

            return

            model = Sequential()

            # TODO: Finish implementation

            # return {'model': model, 'roc_auc': roc_auc, 'logloss': logloss, 'type': 'nnet', 'features': feature_cols}
        else:
            model = Sequential()

            # TODO: Network architecture

            model.fit(featureMatrix, labelVector)

            return {'model': model, 'type': 'nnet', 'features': feature_cols}
