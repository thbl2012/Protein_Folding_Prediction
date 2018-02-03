import os
import numpy as np
import ipyparallel as ipp
import time
from numba import jit
from uuid import uuid4
from src import simulate, data, charge_sequences as chseq
from src.correlation import run_to_corr_matrices
from src.atom_chain import AtomChain


def get_hours(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)


def get_params(from_file):
    params = {}
    with open(from_file, 'r', encoding='utf-8') as file:
        for line in file:
            token = [s.strip() for s in line.split(':')]
            if len(token) > 1:
                params[token[0]] = np.float(token[1])
    return params


def auto_short_sim(
        charge_seq,
        charge_seq_name,
        model_params,
        ref_config,
        sim_short_params,
        save_dir='data_cnn/data_raw',
):
    unique_id = uuid4()
    corr_diag_path_0 = '{}/{}/diag_0/{}.npy'.format(save_dir, charge_seq_name, unique_id)
    corr_diag_path_1 = '{}/{}/diag_1/{}.npy'.format(save_dir, charge_seq_name, unique_id)
    chain = AtomChain(ref_config, charge_seq,
                      spring_const=model_params['spring_const'],
                      spring_len=model_params['spring_len'],
                      atom_radius=model_params['atom_radius'],
                      epsilon=model_params['epsilon'],
                      boltzmann_const=model_params['boltzmann_const'],
                      temperature=model_params['temperature'],
                      rebuild=True)
    corr_matrices = run_to_corr_matrices(simulate.repeat_simulate_short(chain, **sim_short_params))
    np.save(corr_diag_path_0, corr_matrices[0], allow_pickle=False, fix_imports=False)
    np.save(corr_diag_path_1, corr_matrices[1], allow_pickle=False, fix_imports=False)


def repeat_short_sim(
        charge_seq,
        charge_seq_name,
        model_params,
        ref_config,
        sim_short_params,
        repeat=1000,
        save_dir='data_cnn/data_raw',
):
    os.makedirs('{}/{}/diag_0'.format(save_dir, charge_seq_name), exist_ok=True)
    os.makedirs('{}/{}/diag_1'.format(save_dir, charge_seq_name), exist_ok=True)

    avg_time = 0
    for i in range(1, repeat+1):
        start = time.time()

        auto_short_sim(charge_seq, charge_seq_name, model_params,
                       ref_config, sim_short_params, save_dir=save_dir)

        # Progress report
        running_time = time.time() - start
        avg_time += (running_time - avg_time) / i
        rem_time = get_hours(int(avg_time * (repeat - i)))
        print('{} done out of {}. Time remaining: {}'.format(i, repeat, rem_time))


def parallel_short_sim():
    pass


def main():
    os.chdir('..')
    config = get_params('data_cnn/config/params.cfg')
    ref_config = data.load_ref_config('data_cnn/config/ref_config.npy')

    model_att = ['spring_len', 'spring_const', 'atom_radius', 'epsilon', 'boltzmann_const', 'temperature']
    model_params = {param: config[param] for param in model_att}

    sim_short_att = ['max_dist', 'trial_no', 'save_period', 'num_repeat']
    sim_short_params = {param: config[param] for param in sim_short_att}
    for att in sim_short_att:
        if att != 'max_dist':
            sim_short_params[att] = int(sim_short_params[att])

    repeat_short_sim(
        chseq.predefined['mutant_alt_0p2'],
        'mutant_alt_0p2',
        model_params=model_params,
        ref_config=ref_config,
        sim_short_params=sim_short_params,
        repeat=50,
        save_dir='data_cnn/data_raw',
    )


if __name__ == '__main__':
    main()
