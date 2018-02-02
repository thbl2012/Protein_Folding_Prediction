import os

import numpy as np
from matplotlib import colors, pyplot as plt


def run_to_corr_matrices(run_matrix):
    # print('runs shape: {}'.format(str(run_matrix.shape)))
    num_runs, t, num_atoms, atom_dim = run_matrix.shape
    num_features = num_atoms * atom_dim
    run_matrix = run_matrix.reshape(num_runs, t, num_features)
    # print('runs reshape: {}'.format(str(run_matrix.shape)))
    run_corr_matrix = np.empty((t, num_features, num_features))
    for i in range(t):
        run_corr_matrix[i] = correlation_matrix(run_matrix[:, i, :])
    run_corr_matrix_0 = np.copy(run_corr_matrix)
    for i in range(t):
        np.fill_diagonal(run_corr_matrix_0[i], 0)
    return run_corr_matrix_0, run_corr_matrix


def correlation_matrix(feature_arr):
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
    results[outlier_matrix(feature_arr)] = 0
    return results


def outlier_matrix(feature_arr):
    no_change = (feature_arr.max(axis=0) - feature_arr.min(axis=0)) < 0.001
    return no_change.reshape(-1, 1) | no_change


def visualize(matrix, fig_out=None, show=False):
    cmap = colors.LinearSegmentedColormap.from_list(
        'cmap', ['blue', 'white', 'red'], 256
    )
    cmap.set_under(color='white')
    cmap.set_over(color='white')
    img = plt.imshow(matrix, cmap=cmap, vmin=-0.95, vmax=0.95, interpolation='nearest', origin='upper')
    plt.colorbar(img, cmap=cmap)
    plt.suptitle = 'mcs={}'.format(fig_out)
    if fig_out:
        plt.savefig(fig_out, format='png', dpi=600)
    if show:
        plt.show()
    plt.close()
