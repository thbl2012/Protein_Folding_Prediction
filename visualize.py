import numpy as np
import os
from matplotlib import pyplot as plt, colors

from item import load_or_make_items
from util import wild_type


def correlation_matrix(item, diag=False):
    t, length, age = (int(item[i]) for i in range(3))
    features = item[3: length+3] / age
    pairwise_prod_mean = item[length+3:].reshape(length, length) / age
    pairwise_mean_prod = features.reshape(-1, 1) * features
    self_corr = np.sqrt(pairwise_prod_mean.diagonal(0) - np.square(features))
    self_corr_matrix = self_corr.reshape(-1, 1) * self_corr
    results = (pairwise_prod_mean - pairwise_mean_prod) / self_corr_matrix
    results[~np.isfinite(results)] = 0
    if not diag:
        np.fill_diagonal(results, 0)
    return results


def visualize(matrix, fig_out=None, show=False):
    cmap = colors.LinearSegmentedColormap.from_list(
        'cmap', ['blue', 'white', 'red'], 256
    )
    img = plt.imshow(matrix, cmap=cmap, interpolation='nearest', origin='upper')
    plt.colorbar(img, cmap=cmap)
    if fig_out:
        plt.savefig(fig_out, format='png', dpi=600)
    if show:
        plt.show()


def trial():
    item = load_or_make_items('run/run_000/items_10000000.npy', 0, 0, 0, 0)[999]
    cm = correlation_matrix(item)
    visualize(cm, fig_out='test_vis.png')
    p = wild_type(30).argsort()
    l = []
    for i in p:
        l.extend((i*3, i*3+1, i*3+2))
    cm = cm[l][:, l]
    plt.close()
    visualize(cm, fig_out='test_vis_reindex.png')


def auto_generate(save_dir):
    os.makedirs('vis', exist_ok=True)
    for filename in os.listdir(save_dir):
        items = load_or_make_items('{}/{}'.format(save_dir, filename), 0, 0, 0, 0)


if __name__ == '__main__':
    trial()
