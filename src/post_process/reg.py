import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

from post_process.outliers import find_outliers, find_outliers_order


def linear_regression(ticks, values, errors, points, mode="x", verbose=0):
    if len(np.unique(values)) == 1:
        return [values[0] for _ in range(len(points))]
    elif len(values) == 0:
        return [0 for _ in range(len(points))]

    ticks = np.array([t for i, t in enumerate(ticks) if i not in errors])

    if mode == "x":
        x_test = (points[:, 0] + points[:, 2]) / 2
        x_train = (ticks[:, 0] + ticks[:, 2]) / 2
    else:
        x_test = (points[:, 1] + points[:, 3]) / 2
        x_train = (ticks[:, 1] + ticks[:, 3]) / 2

    corr = np.abs(pearsonr(x_train, values).statistic)
    corr_rank = np.abs(spearmanr(x_train, values).statistic)

    if verbose:
        print("Correlations before pp", corr, corr_rank)

    outliers = find_outliers(x_train, values, verbose=verbose, corr="pearson")
    x_train = np.array([x for j, x in enumerate(x_train) if j not in outliers])
    values = np.array([v for j, v in enumerate(values) if j not in outliers])

    outliers = find_outliers_order(values, verbose=verbose)
    x_train = np.array([x for j, x in enumerate(x_train) if j not in outliers])
    values = np.array([v for j, v in enumerate(values) if j not in outliers])

    corr = np.abs(pearsonr(x_train, values).statistic)
    corr_rank = np.abs(spearmanr(x_train, values).statistic)

    if verbose:
        print("Correlations after pp", corr, corr_rank)

    #     log = False
    #     if corr > 0.99:
    #         pass
    #     else:
    #         if corr_rank > 0.99 and np.min(values) > 0:
    #             corr_log = np.abs(pearsonr(x_train, np.log(values)).statistic)

    # #             print("log", corr_log)
    #             if corr_log > 0.99:
    #                 log = True
    #                 values = np.log(values)

    model = LinearRegression()

    model.fit(x_train[:, None], values)

    pred = model.predict(x_test[:, None])

    #     if log:
    #         pred = np.exp(pred)

    #     print(x_test, pred)

    return pred


def rounding(x):
    thresholds = [40, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0001]
    for i, threshold in enumerate(thresholds):
        if x > threshold:
            return i
    return 100
