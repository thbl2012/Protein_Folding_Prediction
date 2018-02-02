import os

from src import simulate, data
from src.atom_chain import AtomChain
from src.correlation import run_to_corr_matrices, visualize
from src.charge_sequences import *
from src import cuda_simulate


def auto_sim(
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

    data_dir = '{}/ref_config'.format(run_dir)
    short_sim_dir = '{}/short_sim'.format(run_dir)
    ref_config_path = '{}/{}.npy'.format(data_dir, ref_config_name)
    samples_path = '{}/{}.npy'.format(data_dir, samples_name)
    equil_seed_path = '{}/{}.npy'.format(data_dir, equil_seed_name)
    runs_path = '{}/{}'.format(short_sim_dir, charge_seq_name)
    corr_diag_path_0 = '{}/{}/{}.npy'.format(short_sim_dir, charge_seq_name, corr_diag_name_0)
    corr_diag_path_1 = '{}/{}/{}.npy'.format(short_sim_dir, charge_seq_name, corr_diag_name_1)
    fig_paths = ('{}/{}/{}'.format(fig_dir, charge_seq_name, fig_names[0]),
                 '{}/{}/{}'.format(fig_dir, charge_seq_name, fig_names[1]))

    ref_config: np.ndarray = None
    samples: np.ndarray = None
    equil_seed: AtomChain = None
    corr_matrices: np.ndarray = None
    runs: np.ndarray = None

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
        os.makedirs(runs_path, exist_ok=True)
        if not sim_short_params:
            raise ValueError('No parameters supplied for short re-simulation')
        if ref_config is None:
            ref_config = data.load_ref_config(ref_config_path)
        if equil_seed is None:
            equil_seed = data.load_chain_from_file(equil_seed_path)
        chain = AtomChain(ref_config, sim_short_charges,
                          spring_const=equil_seed.spring_const,
                          spring_len=equil_seed.spring_len,
                          atom_radius=equil_seed.atom_radius,
                          epsilon=equil_seed.epsilon,
                          boltzmann_const=equil_seed.boltzmann_const,
                          temperature=equil_seed.temperature,
                          rebuild=True)
        runs = simulate.repeat_simulate_short(chain, **sim_short_params)
        data.save_runs(runs, '{}/{}.npy'.format(runs_path, runs_name))
        mode = 're_corr_matrices'
    if mode == 're_corr_matrices':
        if runs is None:
            runs = data.load_runs(runs_path)
        corr_matrices = run_to_corr_matrices(runs)
        np.save(corr_diag_path_0, corr_matrices[0], allow_pickle=False, fix_imports=False)
        np.save(corr_diag_path_1, corr_matrices[1], allow_pickle=False, fix_imports=False)

    # Visualization
    for i in range(2):
        os.makedirs(fig_paths[i], exist_ok=True)
        for t in range(corr_matrices[i].shape[0]):
            visualize(
                corr_matrices[i][t],
                fig_out='{}/{}.png'.format(fig_paths[i], t * sim_short_params['save_period']),
                show=False
            )
    # Save charges
    with open('{}/{}/charges.txt'.format(fig_dir, charge_seq_name), 'w', encoding='utf-8') as txt:
        print('name: ' + charge_seq_name, file=txt)
        print('charges: ', ', '.join(['{:+.2f}'.format(e) for e in sim_short_charges]), sep=': ', file=txt)
    np.save('{}/{}'.format(runs_path, 'charges.npy'), sim_short_charges, allow_pickle=False, fix_imports=False)


def test_auto_sim(
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

    data_dir = '{}/ref_config'.format(run_dir)
    short_sim_dir = '{}/short_sim'.format(run_dir)
    ref_config_path = '{}/{}.npy'.format(data_dir, ref_config_name)
    samples_path = '{}/{}.npy'.format(data_dir, samples_name)
    equil_seed_path = '{}/{}.npy'.format(data_dir, equil_seed_name)
    runs_path = '{}/{}'.format(short_sim_dir, charge_seq_name)
    corr_diag_path_0 = '{}/{}/{}.npy'.format(short_sim_dir, charge_seq_name, corr_diag_name_0)
    corr_diag_path_1 = '{}/{}/{}.npy'.format(short_sim_dir, charge_seq_name, corr_diag_name_1)
    fig_paths = ('{}/{}/{}'.format(fig_dir, charge_seq_name, fig_names[0]),
                 '{}/{}/{}'.format(fig_dir, charge_seq_name, fig_names[1]))

    ref_config: np.ndarray = None
    samples: np.ndarray = None
    equil_seed: AtomChain = None
    corr_matrices: np.ndarray = None
    runs: np.ndarray = None

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
        os.makedirs(runs_path, exist_ok=True)
        if not sim_short_params:
            raise ValueError('No parameters supplied for short re-simulation')
        if ref_config is None:
            ref_config = data.load_ref_config(ref_config_path)
        if equil_seed is None:
            equil_seed = data.load_chain_from_file(equil_seed_path)
        chain = AtomChain(ref_config, sim_short_charges,
                          spring_const=equil_seed.spring_const,
                          spring_len=equil_seed.spring_len,
                          atom_radius=equil_seed.atom_radius,
                          epsilon=equil_seed.epsilon,
                          boltzmann_const=equil_seed.boltzmann_const,
                          temperature=equil_seed.temperature,
                          rebuild=True)

        num_repeat = sim_short_params['num_repeat']
        trial_no = sim_short_params['trial_no']
        save_period = sim_short_params['save_period']
        max_dist = sim_short_params['max_dist']

        vec_num_repeat = np.empty(num_repeat, dtype=np.int32)
        vec_save_period = np.arange(0, trial_no, save_period, dtype=np.int32) + save_period

        runs = cuda_simulate.test_repeat_simulate_short(
            ref_config, sim_short_charges, chain.charge_matrix,
            chain.dist_matrix, chain.spring_const, chain.spring_len,
            chain.atom_radius, chain.epsilon, chain.boltzmann_const,
            chain.temperature, chain.lennard, chain.spring, chain.coulomb,
            chain.energy, max_dist, trial_no, vec_num_repeat, vec_save_period)
        data.save_runs(runs, '{}/{}.npy'.format(runs_path, runs_name))
        mode = 're_corr_matrices'
    if mode == 're_corr_matrices':
        if runs is None:
            runs = data.load_runs(runs_path)
        corr_matrices = run_to_corr_matrices(runs)
        np.save(corr_diag_path_0, corr_matrices[0], allow_pickle=False, fix_imports=False)
        np.save(corr_diag_path_1, corr_matrices[1], allow_pickle=False, fix_imports=False)

    # Visualization
    for i in range(2):
        os.makedirs(fig_paths[i], exist_ok=True)
        for t in range(corr_matrices[i].shape[0]):
            visualize(
                corr_matrices[i][t],
                fig_out='{}/{}.png'.format(fig_paths[i], t * sim_short_params['save_period']),
                show=False
            )
    # Save charges
    with open('{}/{}/charges.txt'.format(fig_dir, charge_seq_name), 'w', encoding='utf-8') as txt:
        print('name: ' + charge_seq_name, file=txt)
        print('charges: ', ', '.join(['{:+.2f}'.format(e) for e in sim_short_charges]), sep=': ', file=txt)
    np.save('{}/{}'.format(runs_path, 'charges.npy'), sim_short_charges, allow_pickle=False, fix_imports=False)


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
        'trial_no': 100,
        'save_period': 5,
        'num_repeat': 10000
    }
    sample_params = {
        'max_dist': 0.5,
        'num_equiv': 200000,
        'num_sample': 1000,
    }
    test_auto_sim(
        charge_seq,
        charge_seq_name=charge_seq_name,
        mode=mode,
        ref_method='centroid',
        sample_params=sample_params,
        sim_params=sim_params,
        sim_short_params=sim_short_params
    )


def main():
    k = 0
    Q = 0
    name = 'test_wild_type_1'
    charges = wild_type_like(q=2, length=30)
    # charges = np.concatenate((full(Q, length=k + 1), wild_type(2., length=30)[k + 1:]), axis=0)
    run(charges, name, 're_short_sim')


if __name__ == '__main__':
    main()
