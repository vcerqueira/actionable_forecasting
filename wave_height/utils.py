from typing import List

import pandas as pd


def remove_invalid_observations(X: pd.DataFrame,
                                y: pd.Series,
                                lag_columns: List[str],
                                decision_thr: float):
    """
    removing observations in which the phenomena (y>=thr) already occurs in the input
    :param X: predictors as pd.DF
    :param y: target variable
    :param lag_columns: predictors relative to the target variable (lags)
    :param decision_thr: decision thr
    :return:
    """

    if isinstance(y, pd.Series):
        y = y.values

    idx_to_kp = ~(X[lag_columns] >= decision_thr).any(axis=1)

    X_t = X.loc[idx_to_kp, :].reset_index(drop=True).copy()
    y_t = y[idx_to_kp]

    return X_t, y_t
