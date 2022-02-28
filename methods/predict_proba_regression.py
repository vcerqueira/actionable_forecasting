import numpy as np


def mc_predict_proba(y_hat: np.ndarray,
                     scale: float,
                     decision_thr: float,
                     n_trials: int):
    """
    retrieving probabilistic predictions (0--1)
    from numeric predictions where a decision threshold is available

    todo assumes normal dist, generalize

    :param y_hat: numeric predictions
    :param scale: standard deviation for the normal dist
    :param decision_thr: decision threshold as a float
    :param n_trials: number of monte carlo trials
    :return: probabilities for each instance
    """

    y_prob = []
    for pred in y_hat:
        mc_trials = np.random.normal(loc=pred, scale=scale, size=n_trials)
        prob = (mc_trials > decision_thr).mean()
        y_prob.append(prob)

    y_prob = np.asarray(y_prob)

    return y_prob
