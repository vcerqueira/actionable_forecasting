from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score, f1_score, recall_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE

from wave_height.config import (DATA_DIR,
                                EMBED_DIM,
                                HORIZON,
                                TARGET,
                                THRESHOLD_PERCENTILE,
                                MC_N_TRIALS,
                                CV_N_FOLDS)
from wave_height.utils import remove_invalid_observations
from methods.embedding import MultivariateTDE
from methods.predict_proba_regression import mc_predict_proba
from methods.random_forest import predict_proba_from_trees

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv(DATA_DIR)
df.index = pd.to_datetime(df['time'])
df = df.drop('time', axis=1)
df = df.resample('H').mean()

#

cv = TimeSeriesSplit(n_splits=CV_N_FOLDS)

data_set = MultivariateTDE(data=df,
                           horizon=HORIZON,
                           k=EMBED_DIM,
                           target_col=TARGET)

LAG_COLUMNS = [f'{TARGET}-{i}' for i in range(1, EMBED_DIM + 1)]
TARGET_COLUMNS = [f'{TARGET}+{i}' for i in range(1, HORIZON + 1)]

data_set = data_set.dropna()

X = data_set.drop(TARGET_COLUMNS, axis=1)
Y = data_set[TARGET_COLUMNS]

y = Y.mean(axis=1)

METRICS = ['CLF_F1', 'CLF(SM)_F1', 'REG_F1',
           'CLF_REC', 'CLF(SM)_REC','REG_REC',
           'CLF_PREC', 'CLF(SM)_PREC', 'REG_PREC',
           'REG_R2',
           'CLF_AUC', 'CLF(SM)_AUC','REG_RF_AUC', 'REG_MC_AUC']

results = {m: [] for m in METRICS}
for train_index, test_index in cv.split(X):
    print('.')
    print('Subsetting iter data')
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]

    print('Getting pars')
    y_std = y_train.std()
    thr = np.quantile(y_train, THRESHOLD_PERCENTILE)

    print('Removing invalid data')
    X_train, y_train = remove_invalid_observations(X=X_train, y=y_train,
                                                   lag_columns=LAG_COLUMNS,
                                                   decision_thr=thr)
    X_test, y_test = remove_invalid_observations(X=X_test, y=y_test,
                                                 lag_columns=LAG_COLUMNS,
                                                 decision_thr=thr)

    print('Training')
    classifier = RandomForestClassifier()
    classifier_sm = RandomForestClassifier()
    regressor = RandomForestRegressor()

    y_train_clf = (y_train >= thr).astype(int)
    y_test_clf = (y_test >= thr).astype(int)

    print('..Classifier')
    classifier.fit(X_train, y_train_clf)
    print('..Smoting')
    X_train_sm, y_train_clf_sm = SMOTE().fit_resample(X_train, y_train_clf)

    print('..Classifier w smote')
    classifier_sm.fit(X_train_sm, y_train_clf_sm)
    # importances = dict(zip(X_train.columns, classifier.feature_importances_))
    # pprint(importances)
    print('..Regressor')
    regressor.fit(X_train, y_train)

    print('..Predicting')
    y_pred_clf = classifier.predict(X_test)
    y_pred_clf_sm = classifier_sm.predict(X_test)
    y_prob_clf = classifier.predict_proba(X_test)
    y_prob_clf = np.array([x[1] for x in y_prob_clf])
    y_prob_clf_sm = classifier_sm.predict_proba(X_test)
    y_prob_clf_sm = np.array([x[1] for x in y_prob_clf_sm])
    y_pred_reg = regressor.predict(X_test)
    y_pred_reg_thr = (regressor.predict(X_test) > thr).astype(int)
    y_prob_reg_rf = predict_proba_from_trees(model=regressor, X=X_test, thr=thr)
    y_prob_reg_mc = mc_predict_proba(y_hat=y_pred_reg,
                                     scale=y_std,
                                     decision_thr=thr,
                                     n_trials=MC_N_TRIALS)

    print('..Scoring')
    results['CLF_F1'].append(f1_score(y_test_clf, y_pred_clf))
    results['CLF(SM)_F1'].append(f1_score(y_test_clf, y_pred_clf_sm))
    results['REG_F1'].append(f1_score(y_test_clf, y_pred_reg_thr))
    results['CLF_REC'].append(recall_score(y_test_clf, y_pred_clf))
    results['CLF(SM)_REC'].append(recall_score(y_test_clf, y_pred_clf_sm))
    results['REG_REC'].append(recall_score(y_test_clf, y_pred_reg_thr))
    results['CLF_PREC'].append(precision_score(y_test_clf, y_pred_clf))
    results['REG_PREC'].append(precision_score(y_test_clf, y_pred_reg_thr))
    results['CLF(SM)_PREC'].append(precision_score(y_test_clf, y_pred_clf_sm))

    results['REG_R2'].append(r2_score(y_test, y_pred_reg))

    results['CLF_AUC'].append(roc_auc_score(y_test_clf, y_prob_clf))
    results['CLF(SM)_AUC'].append(roc_auc_score(y_test_clf, y_prob_clf_sm))
    results['REG_RF_AUC'].append(roc_auc_score(y_test_clf, y_prob_reg_rf))
    results['REG_MC_AUC'].append(roc_auc_score(y_test_clf, y_prob_reg_mc))

    pprint(results)

#
pd.DataFrame(results).mean()