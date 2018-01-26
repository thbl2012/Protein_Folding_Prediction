import matplotlib as mpl
import numpy as np
import os
mpl.use('TkAgg')
from matplotlib import pyplot as plt


runs = None
corr = None


def correlation_matrix_debug(feature_arr):
    feature_mean = np.mean(feature_arr, axis=0)
    num_runs, num_features = feature_arr.shape
    pairwise_prod_matrices = np.empty((num_runs, num_features, num_features))
    for i in range(feature_arr.shape[0]):
        pairwise_prod_matrices[i] = feature_arr[i].reshape(-1, 1) * feature_arr[i]
    pairwise_prod_mean = np.mean(pairwise_prod_matrices, axis=0)
    pairwise_mean_prod = feature_mean.reshape(-1, 1) * feature_mean

    self_corr = np.sqrt(pairwise_prod_mean.diagonal(0) - np.square(feature_mean))
    self_corr_matrix = self_corr.reshape(-1, 1) * self_corr
    results = (pairwise_prod_mean - pairwise_mean_prod) / self_corr_matrix
    results[~np.isfinite(results)] = 0
    return results, pairwise_prod_matrices, pairwise_prod_mean, pairwise_mean_prod, self_corr,


def get_fa(idx):
    global runs
    if runs is None:
        runs = np.load('run/data/runs.npy', fix_imports=False, allow_pickle=False)
    return runs[idx]


def get_co(idx):
    global corr
    if corr is None:
        corr = np.load('run/data/corr_diag_0.npy', fix_imports=False, allow_pickle=False)
    return corr[idx]

