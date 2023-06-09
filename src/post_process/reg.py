import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

from post_process.outliers import find_outliers, find_outliers_order

def handle_sign(x_train, values, verbose=0, corr="spearman", th=0.99, mode="x"):
    corr_fct = spearmanr if corr == "spearman" else pearsonr
    
    if np.all(values >= 0):
        return values
    elif np.all(values <= 0):
        if mode == "x":
            if np.all(values[np.argsort(x_train)] == np.sort(values)):
                return values
        else:
            if np.all(values[np.argsort(x_train)] == np.sort(values)[::-1]):
                return values
    elif np.all(values[np.argsort(x_train)] == np.sort(values)):
        return values
    elif np.all(values[np.argsort(x_train)] == np.sort(values)[::-1]):
        return values
    
    corr_start = np.abs(corr_fct(x_train, values).statistic)
    
    values_abs = np.abs(values)
    corr_abs = np.abs(corr_fct(x_train, np.abs(values)).statistic)
    
#     print(mode)
#     print(corr_start, corr_abs)
#     print(values_abs)
#     print(values_abs[np.argsort(x_train)])
    
    if corr_abs >= corr_start:
        if mode == "x":
            if np.all(values_abs[np.argsort(x_train)] == np.sort(values_abs)):
                if verbose:
                    print('Increasing, using abs(x) !')
                return values_abs
            elif np.all(values_abs[np.argsort(x_train)] == np.sort(values_abs)[::-1]):
                if verbose:
                    print('Decreasing, using -abs(x) !')
                return - values_abs
        else:
            if np.all(values_abs[np.argsort(-x_train)] == np.sort(values_abs)):
                if verbose:
                    print('Increasing, using abs(y) !')
                return values_abs
            elif np.all(values_abs[np.argsort(-x_train)] == np.sort(values_abs)[::-1]):
                if verbose:
                    print('Decreasing, using -abs(y) !')
                return - values_abs
        
    return values


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
        print(f"Correlations {mode} before pp", corr, corr_rank)
        
    values = handle_sign(x_train, values, verbose=1, corr="pearson", mode=mode)

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
