import numpy as np
import pandas as pd
import seaborn as sns


def mahalanobis_dist(data: pd.DataFrame):
    covariance_matrix = data.cov()
    inv_covariance_matrix = np.linalg.inv(covariance_matrix.values)
    vars_mean = []
    for i in range(data.shape[0]):
        vars_mean.append(list(data.mean(axis=0)))
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return np.array(md)


#def plot_mahalanobis_dist():