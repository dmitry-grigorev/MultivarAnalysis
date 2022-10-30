import pandas as pd
import numpy as np
import scipy.stats as stats


def corr_test(data: pd.DataFrame, X: str, Y: str, printinfo=False):
    x = data[X].apply(lambda x: float(x))
    y = data[Y].apply(lambda x: float(x))
    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)  # matches Fisher transform
    # Corresponding standard deviation
    se = 1 / np.sqrt(x.size - 3)
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    if printinfo:
        print('Correlation coefficient = ', r)
        print('p-value =', p, ' given $\alpha = $', alpha)
        print('Confidence interval for the correlation coefficient ', lo_z, hi_z)
    return r, lo_z, hi_z, p


def multiple_corr_test(data, pivot, regressors):
    k = len(regressors)
    coefs = [0.] * k
    left, right = [0.] * k, [0.] * k
    pvals = [0.]*k
    for i in range(k):
        coefs[i], left[i], right[i], pvals[i] = corr_test(data, pivot, regressors[i])

    return pd.DataFrame([coefs, left, right, pvals], columns=regressors, index=['$\rho$', 'left', 'right', 'p-value']).transpose()
