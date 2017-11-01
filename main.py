import numpy as np
import os
from matplotlib import pyplot as plt, colors

import simulate


def load_chain_from_file(from_file):
    seed = np.load(from_file, allow_pickle=False, fix_imports=False, encoding='bytes')
    return seed.reshape(-1, 3)


def get_sample_equil(from_file):
    samples_raw = np.load(from_file, allow_pickle=False,
                          fix_imports=False, encoding='bytes')
    return samples_raw.reshape(samples_raw.shape[0], -1, 3)


def save_samples_to_file(samples, to_file):
    np.save(to_file, samples.reshape(len(samples), -1),
            allow_pickle=False, fix_imports=False)


def get_ref_config(samples, ref_method='centroid', ref_config_out=None):
    if ref_method == 'centroid':
        ref_config = simulate.centroid(samples)
    else:
        raise ValueError('Invalid reference method')
    if ref_config_out:
        np.save(ref_config_out, ref_config, allow_pickle=False, fix_imports=False)
    return ref_config


def main(
        chain_name='wild_type',
        mode='re_visualize',
        ref_method='centroid',
        sample_params=None,
        sim_params=None,
        sim_short_params=None,
):
    run_dir = 'run'
    ref_config_name = 'ref_config'
    samples_name = 'samples'
    equil_seed_name = 'equil_seed'
    runs_name = 'runs'
    corr_diag_name_0 = 'corr_diag_0'
    corr_diag_name_1 = 'corr_diag_1'
    fig_dir = 'fig'
    fig_names = ('corr_diag_0', 'corr_diag_1')
    length = 30

    save_dir = '{}/{}'.format(run_dir, chain_name)
    ref_config_path = '{}/{}.npy'.format(save_dir, ref_config_name)
    samples_path = '{}/{}.npy'.format(save_dir, samples_name)
    equil_seed_path = '{}/{}.npy'.format(save_dir, equil_seed_name)
    runs_path = '{}/{}.npy'.format(save_dir, runs_name)
    corr_diag_path_0 = '{}/{}.npy'.format(save_dir, corr_diag_name_0)
    corr_diag_path_1 = '{}/{}.npy'.format(save_dir, corr_diag_name_1)
    fig_paths = ('{}/{}/{}'.format(fig_dir, chain_name, fig_names[0]),
                 '{}/{}/{}'.format(fig_dir, chain_name, fig_names[1]))

    ref_config = None
    samples = None
    equil_seed = None
    corr_matrices = None
    
    if mode == 're_simulate':
        if not sim_params:
            raise ValueError('No parameters supplied for re-simulation')
        os.makedirs(save_dir, exist_ok=True)
        equil_seed = simulate.simulate_equil(length, chain_type=chain_name,
                                             **sim_params, chain_out=equil_seed_path)
        mode = 're_sample'
    if mode == 're_sample':
        if not sample_params:
            raise ValueError('No parameters supplied for re-sampling')
        if equil_seed is None:
            equil_seed = load_chain_from_file(equil_seed_path)
        samples = simulate.sample_equil(equil_seed, **sample_params, samples_out=samples_path)
        mode = 're_ref'
    if mode == 're_ref':
        if samples is None:
            samples = get_sample_equil(samples_path)
        ref_config = get_ref_config(samples, ref_method=ref_method, ref_config_out=ref_config_path)
        mode = 're_short_sim'
    if mode == 're_short_sim':
        if not sim_short_params:
            raise ValueError('No parameters supplied for short re-simulation')
        if ref_config is None:
            ref_config = load_chain_from_file(ref_config_path)
        runs = simulate.repeat_simulate_short(ref_config, **sim_short_params, out=runs_path)
        corr_matrices = run_to_corr_matrices(runs, out_0=corr_diag_path_0, out_1=corr_diag_path_1)

    # Visualization
    assert type(corr_matrices) == (np.ndarray, np.ndarray), 'Invalid mode input'
    for i in range(1):
        os.makedirs(fig_paths[i], exist_ok=True)
        for t in range(corr_matrices[i].shape[0]):
            visualize(
                corr_matrices[i][t],
                fig_out='{}/{}.png'.format(fig_paths[i], t*sim_short_params['save_period']),
                show=False
            )
    return


def run_to_corr_matrices(run_matrix, out_0=None, out_1=None):
    num_runs, t, num_atoms, atom_dim = run_matrix.shape
    num_features = num_atoms * atom_dim
    run_matrix = run_matrix.reshape(num_runs, t, num_features)
    run_corr_matrix = np.empty(t, num_features, num_features)
    for i in range(t):
        run_corr_matrix[i] = correlation_matrix(run_matrix[:, i, :])
    run_corr_matrix_0 = np.copy(run_corr_matrix)
    for i in range(t):
        run_corr_matrix_0[i].fill_diagonal(0)
    if out_1:
        np.save(out_1, run_corr_matrix.reshape(t, num_features**2),
                allow_pickle=False, fix_imports=False)
    if out_0:
        np.save(out_1, run_corr_matrix_0.reshape(t, num_features ** 2),
                allow_pickle=False, fix_imports=False)
    return run_corr_matrix_0, run_corr_matrix


def correlation_matrix(feature_arr):
    feature_mean = np.mean(feature_arr, axis=0)

    pairwise_prod_matrices = np.empty((feature_arr.shape[0], feature_arr.shape[1] ** 2))
    for i in range(feature_arr.shape[0]):
        pairwise_prod_matrices[i] = feature_arr[i].reshape(-1, 1) * feature_arr[i]
    pairwise_prod_mean = np.mean(pairwise_prod_matrices, axis=0)
    pairwise_mean_prod = feature_mean.reshape(-1, 1) * feature_mean

    self_corr = np.sqrt(pairwise_prod_mean.diagonal(0) - np.square(feature_mean))
    self_corr_matrix = self_corr.reshape(-1, 1) * self_corr
    results = (pairwise_prod_mean - pairwise_mean_prod) / self_corr_matrix
    results[~np.isfinite(results)] = 0
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
