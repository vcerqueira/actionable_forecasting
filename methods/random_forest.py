import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def rf_predict_all(model: RandomForestRegressor, X: pd.DataFrame):
    per_tree_pred = [tree.predict(X)
                     for tree in model.estimators_]

    preds = pd.DataFrame(per_tree_pred).T

    return preds


def predict_proba_from_trees(model: RandomForestRegressor,
                             X: pd.DataFrame,
                             thr: float):
    preds_all = rf_predict_all(model, X)

    rf_prob_ = preds_all.apply(lambda x: np.mean(x > thr), axis=1).values

    return rf_prob_
