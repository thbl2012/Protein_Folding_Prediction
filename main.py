import numpy as np
import os
from matplotlib import pyplot as plt, colors
from atom_chain import AtomChain

import data
import simulate
from util import wild_type


def run_to_corr_matrices(run_matrix, out_0=None, out_1=None):
    print('runs shape: {}'.format(str(run_matrix.shape)))
    num_runs, t, num_atoms, atom_dim = run_matrix.shape
    num_features = num_atoms * atom_dim
    run_matrix = run_matrix.reshape(num_runs, t, num_features)
    print('runs reshape: {}'.format(str(run_matrix.shape)))
    run_corr_matrix = np.empty((t, num_features, num_features))
    for i in range(t):
        run_corr_matrix[i] = correlation_matrix(run_matrix[:, i, :])
    run_corr_matrix_0 = np.copy(run_corr_matrix)
    for i in range(t):
        np.fill_diagonal(run_corr_matrix_0[i], 0)
    if out_1:
        np.save(out_1, run_corr_matrix.reshape(t, num_features**2),
                allow_pickle=False, fix_imports=False)
    if out_0:
        np.save(out_0, run_corr_matrix_0.reshape(t, num_features ** 2),
                allow_pickle=False, fix_imports=False)
    return run_corr_matrix_0, run_corr_matrix


def correlation_matrix(feature_arr):
    print('feature_arr shape: {}'.format(str(feature_arr.shape)))
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
    return results


def visualize(matrix, fig_out=None, show=False):
    cmap = colors.LinearSegmentedColormap.from_list(
        'cmap', ['blue', 'white', 'red'], 256
    )
    img = plt.imshow(matrix, cmap=cmap, interpolation='nearest', origin='upper')
    plt.colorbar(img, cmap=cmap)
    plt.suptitle = 'mcs={}'.format(fig_out)
    if fig_out:
        plt.savefig(fig_out, format='png', dpi=600)
    if show:
        plt.show()
    plt.close()


def main(
        sim_short_charges,
        charge_seq_name='wild_type',
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
    fig_dir = '{}/fig'.format(run_dir)
    fig_names = ('corr_diag_0', 'corr_diag_1')
    length = 30

    data_dir = '{}/data'.format(run_dir)
    ref_config_path = '{}/{}.npy'.format(data_dir, ref_config_name)
    samples_path = '{}/{}.npy'.format(data_dir, samples_name)
    equil_seed_path = '{}/{}.npy'.format(data_dir, equil_seed_name)
    runs_path = '{}/{}.npy'.format(data_dir, runs_name)
    corr_diag_path_0 = '{}/{}.npy'.format(data_dir, corr_diag_name_0)
    corr_diag_path_1 = '{}/{}.npy'.format(data_dir, corr_diag_name_1)
    fig_paths = ('{}/{}/{}'.format(fig_dir, charge_seq_name, fig_names[0]),
                 '{}/{}/{}'.format(fig_dir, charge_seq_name, fig_names[1]))

    ref_config: np.ndarray = None
    samples: np.ndarray = None
    equil_seed: AtomChain = None
    corr_matrices: np.ndarray = None

    if mode == 're_simulate':
        if not sim_params:
            raise ValueError('No parameters supplied for re-simulation')
        os.makedirs(data_dir, exist_ok=True)
        equil_seed = simulate.simulate_equil(length, chain_type=charge_seq_name, **sim_params)
        data.save_chain_to_file(equil_seed, equil_seed_path)
        mode = 're_sample'
    if mode == 're_sample':
        if not sample_params:
            raise ValueError('No parameters supplied for re-sampling')
        if equil_seed is None:
            equil_seed = data.load_chain_from_file(equil_seed_path)
        samples = simulate.sample_equil(equil_seed, **sample_params)
        data.save_samples_to_file(samples, samples_path)
        mode = 're_ref'
    if mode == 're_ref':
        if samples is None:
            samples = data.get_sample_equil(samples_path)
        ref_config = data.get_ref_config(samples, ref_method=ref_method)
        data.save_ref_config(ref_config, ref_config_path)
        mode = 're_short_sim'
    if mode == 're_short_sim':
        if not sim_short_params:
            raise ValueError('No parameters supplied for short re-simulation')
        if ref_config is None:
            ref_config = data.load_ref_config(ref_config_path)
            equil_seed = data.load_chain_from_file(equil_seed_path)
        chain = AtomChain(ref_config, sim_short_charges,
                          spring_const=equil_seed.spring_const,
                          spring_len=equil_seed.spring_len,
                          atom_radius=equil_seed.atom_radius,
                          epsilon=equil_seed.epsilon,
                          boltzmann_const=equil_seed.boltzmann_const,
                          temperature=equil_seed.temperature,
                          rebuild=True)
        runs = simulate.repeat_simulate_short(chain, **sim_short_params, out=runs_path)
        corr_matrices = run_to_corr_matrices(runs, out_0=corr_diag_path_0, out_1=corr_diag_path_1)

    # Visualization
    for i in range(2):
        os.makedirs(fig_paths[i], exist_ok=True)
        for t in range(corr_matrices[i].shape[0]):
            visualize(
                corr_matrices[i][t],
                fig_out='{}/{}.png'.format(fig_paths[i], t * sim_short_params['save_period']),
                show=False
            )
    return


def run(charge_seq, charge_seq_name, mode):
    sim_params = {
        'spring_len': 1,
        'spring_const': 1,
        'atom_radius': 0.5,
        'epsilon': 1,
        'boltzmann_const': 1,
        'temperature': 1,
        'max_dist': 5,
        'start_dist': 1,
        'trial_no': 10000000,
        'plot_period': 100,
        'max_plot_buffer': 10000,
        'verbose': False,
        'colors': None,
        'fig_out': None,
        'print_period': 1000,
    }
    sim_short_params = {
        'max_dist': 5,
        'trial_no': 1000,
        'save_period': 50,
    }
    sample_params = {
        'max_dist': 0.5,
        'num_equiv': 200000,
        'num_sample': 1000,
    }
    main(
        charge_seq,
        charge_seq_name=charge_seq_name,
        mode=mode,
        ref_method='centroid',
        sample_params=sample_params,
        sim_params=sim_params,
        sim_short_params=sim_short_params
    )


if __name__ == '__main__':
    charges = np.full(30, 0)
    for i in range(15):
        charges[i*2+1] = 2
    name = 'alt_0_2'
    run(charges, name, 're_short_sim')
