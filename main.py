import numpy as np
import os
from matplotlib import pyplot as plt, colors

import simulate


def load_chain_from_file(from_file):
    seed = np.load(from_file, allow_pickle=False, fix_imports=False, encoding='bytes')
    return seed.reshape(-1, 3)


def save_chain_to_file(chain, to_file):
    np.save(chain, allow_pickle=False, fix_imports=False)


def get_sample_equil(from_file=None):
    samples_raw = np.load(from_file, allow_pickle=False,
                          fix_imports=False, encoding='bytes')
    return samples_raw.reshape(samples_raw.shape[0], -1, 3)


def save_samples_to_file(samples, to_file):
    np.save(to_file, samples.reshape(len(samples), -1),
            allow_pickle=False, fix_imports=False)


def get_ref_config(samples, ref_method='centroid'):
    if ref_method == 'centroid':
        return simulate.centroid(samples)


def main(
        sim_short_params,
        chain_name='wild_type',
        mode='from_file',
        sample_params=None,
        sim_params=None,
):
    run_dir = 'run'
    ref_config_name = 'ref_config'
    samples_name = 'samples'
    equil_seed_name = 'equil_seed'
    length = 30

    save_dir = '{}/{}'.format(run_dir, chain_name)
    ref_config_path = '{}/{}.npy'.format(save_dir, ref_config_name)
    samples_path = '{}/{}.npy'.format(save_dir, samples_name)
    equil_seed_path = '{}/{}.npy'.format(save_dir, equil_seed_name)

    ref_config = None

    if mode == 'from_file':
        try:
            ref_config = load_chain_from_file(ref_config_path)
        except FileNotFoundError:
            mode = 're_ref'
    if mode == 're_ref':
        try:
            samples = get_sample_equil(samples_path)
            ref_config = simulate.centroid(samples)
            save_chain_to_file(ref_config, ref_config_path)
        except FileNotFoundError:
            mode = 're_sample'
    if mode == 're_sample':
        try:
            if not sample_params:
                raise ValueError('No parameters supplied for re-sampling')
            equil_seed = load_chain_from_file(equil_seed_path)
            samples = simulate.sample_equil(equil_seed, **sample_params)
            save_samples_to_file(samples, samples_path)
        except FileNotFoundError:
            mode = 're_simulate'
    if mode == 're_simulate':
        if not sim_params:
            raise ValueError('No parameters supplied for re-simulation')
        os.makedirs(save_dir, exist_ok=True)
        equil_seed = simulate.simulate_equilibrium(length, chain_type=chain_name, **sim_params)
        save_chain_to_file(equil_seed_path, equil_seed)
        samples = simulate.sample_equil(equil_seed, **sample_params)
        save_samples_to_file(samples, samples_path)
        ref_config = simulate.centroid(samples)
        save_chain_to_file(ref_config, ref_config_path)
    runs = simulate.repeat_simulate_short(ref_config, **sim_short_params)
    # TODO: Correlation matrix create
    pass


def correlation_matrix(feature_arr, zero_diag=True):
    feature_mean = np.mean(feature_arr, axis=0)

    pairwise_prod_matrices = np.empty((feature_arr.shape[0], feature_arr.shape[1]**2))
    for i in range(feature_arr.shape[0]):
        pairwise_prod_matrices[i] = feature_arr[i].reshape(-1, 1) * feature_arr[i]
    pairwise_prod_mean = np.mean(pairwise_prod_matrices, axis=0)
    pairwise_mean_prod = feature_mean.reshape(-1, 1) * feature_mean

    self_corr = np.sqrt(pairwise_prod_mean.diagonal(0) - np.square(feature_mean))
    self_corr_matrix = self_corr.reshape(-1, 1) * self_corr
    results = (pairwise_prod_mean - pairwise_mean_prod) / self_corr_matrix
    results[~np.isfinite(results)] = 0
    if zero_diag:
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


if __name__ == '__main__':
    pass
