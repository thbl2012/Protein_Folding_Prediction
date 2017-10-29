import multiprocessing as mp
import time

from simulate import simulate_true


def simulate_wrapper(sim_params):
    simulate_true(**sim_params)


def simulate_parallel(num_proc, sim_params):
    # num_proc <= 1000
    params_lst = []
    for i in range(num_proc):
        params = sim_params.copy()
        params.update(run_id=i)
        params_lst.append(params)
    pool = mp.Pool()
    pool.map(simulate_wrapper, params_lst)


def main():
    sim_params = {
        'length': 30,
        'chain_type': 'wild_type',
        'spring_const': 1,
        'spring_len': 0.5,
        'atom_radius': 1,
        'epsilon': 1,
        'boltzmann_const': 1,
        'temperature': 1,
        'max_dist': 0.5,
        'start_dist': 1,
        'trial_no': 10 ** 6,
        'save_period': 1000,
        'records_per_file': 100,
        # 'save_dir': 'D:/Coding Projects/Python/FYP/run',
        'save_dir': '/home/users/a0112184/simulate/run',
        'print_period': 0,
    }
    num_proc = 5
    start = time.time()
    simulate_parallel(num_proc, sim_params)
    print('{} simulations done in {:.2f} minutes'.format(num_proc, (time.time() - start) / 60))


if __name__ == '__main__':
    main()
