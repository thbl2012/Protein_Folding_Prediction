import numpy as np
import random
import pymol

from matplotlib import pyplot

from atom_chain import AtomChain, straight_chain, wild_type


# Storage parameters
STATUS_FILE = 'status.txt'
STT_PREFIX = 'atoms'
FIG_PREFIX = 'fig'


def simulate_equil(
        # Input chain parameters
        length,
        chain_type='wild_type',

        # Model parameters
        spring_len=1.,
        spring_const=1.,
        atom_radius=1.,
        epsilon=1.,
        boltzmann_const=1.,
        temperature=1.,

        # Monte Carlo parameters
        max_dist=1.,
        start_dist=1.,
        trial_no=10000000,

        # Plot parameters
        plot_period=100,
        max_plot_buffer=10000,
        verbose=False,
        colors=None,
        fig_out=None,

        # Info parameters
        print_period=0,
        chain_out=None
):
    # Set up colors
    if not colors:
        colors = {'energy': 'k', 'spring': 'r', 'lennard': 'g', 'coulomb': 'b'}

    # Set up simulation
    charges = None
    if chain_type == 'wild_type':
        charges = wild_type(length)
    chain = AtomChain(
        straight_chain(len(charges), start_dist),
        charges,
        spring_const=spring_const,
        spring_len=spring_len,
        atom_radius=atom_radius,
        epsilon=epsilon,
        boltzmann_const=boltzmann_const,
        temperature=temperature,
        rebuild=True
    )
    accepted_count = 0
    plot_count = 0
    trial_plot = np.empty(max_plot_buffer)
    plots = {'energy': np.empty(max_plot_buffer)}
    if verbose:
        plots.update(spring=np.empty(max_plot_buffer),
                     lennard=np.empty(max_plot_buffer),
                     coulomb=np.empty(max_plot_buffer))
    pyplot.ylabel('energy')
    pyplot.xlabel('trial')
    kwargs = {'linestyle': 'None', 'markersize': 1, 'marker': '+'}

    # Begin simulation
    for t in range(1, trial_no+1):
        accepted = chain.mutate(max_dist)
        accepted_count += accepted
        if t % plot_period == 0:
            for label, plot in plots.items():
                plot[plot_count] = getattr(chain, label)
            trial_plot[plot_count] = t
            plot_count += 1
        if print_period > 0 and t % print_period == 0:
            print(t)
        if plot_count == max_plot_buffer or t == trial_no:
            for label, plot in plots.items():
                plot[plot > 200] = 0
                pyplot.plot(trial_plot, plot, label=label, color=colors[label], **kwargs)
            for _, plot in plots.items():
                plot.fill(0)
            trial_plot.fill(0)
            plot_count = 0

    # Save figure and chain
    if fig_out:
        pyplot.savefig(fig_out, format='png', dpi=600)
    if chain_out:
        np.save(chain_out, chain.atoms, allow_pickle=False, fix_imports=False)
    return chain


def sample_equil(chain, max_dist=1, num_equiv=5000000, num_sample=1000):
    chain = chain.copy()
    samples = np.empty((num_sample, chain.atoms.shape[0], chain.atoms.shape[1]), dtype=np.float64)
    indices = np.array(random.sample(range(num_equiv), num_sample), dtype=np.int32)
    indices.sort()
    i = 0
    for t in range(num_equiv):
        chain.mutate(max_dist)
        if (t+1) % 1000 == 0:
            print(t+1)
        if t == indices[i]:
            samples[i] = chain.atoms
            i += 1
        if i == len(indices):
            break
    return samples


def aligned_dist(p, q):
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    cov = np.dot(p.T, q)
    u, s, v = np.linalg.svd(cov, full_matrices=False)
    rot_dir = np.linalg.det(np.dot(u, v))
    rot = np.dot(u, np.dot(np.array([[1, 0, 0], [0, 1, 0], [0, 0, rot_dir]]), v)).T
    msd = np.sum(np.square(np.dot(p, rot) - q)) / p.shape[0]
    return msd


def centroid(samples):
    ns = samples.shape[0]
    aligned_dist_pairwise = np.empty((ns, ns))
    for i in range(ns):
        for j in range(ns):
            if j == i:
                aligned_dist_pairwise[i][j] = 0
            elif j > i:
                aligned_dist_pairwise[i][j] = aligned_dist(samples[i], samples[j])
            else:
                aligned_dist_pairwise[i][j] = aligned_dist_pairwise[j][i]
    sum_dist = np.sum(aligned_dist_pairwise, axis=0)
    return samples[sum_dist.argmin()]


def simulate_short(
        chain,
        max_dist=1,
        trial_no=1000,
        save_period=50,
):
    chain = chain.copy()
    states = np.empty(
        (trial_no//save_period, chain.atoms.shape[0], chain.atoms.shape[1]),
        dtype=np.float64
    )
    for t in range(1, trial_no+1):
        chain.mutate(max_dist)
        if t % save_period == 0:
            states[t//save_period-1] = chain.atoms
    return states


def repeat_simulate_short(
        chain,
        max_dist=1,
        trial_no=1000,
        num_repeat=1000,
        save_period=50,
        out=None,
):
    runs = np.empty((num_repeat, trial_no//save_period,
                     chain.atoms.shape[0], chain.atoms.shape[1]))
    for i in range(num_repeat):
        runs[i] = simulate_short(chain, max_dist=max_dist, trial_no=trial_no, save_period=save_period)
        print(i+1)
    if out:
        np.save(out, runs.reshape(num_repeat, -1), allow_pickle=False, fix_imports=False)
    return runs


def sandbox():
    sim_params = {
        'length': 30,
        'chain_type': 'wild_type',
        'spring_len': 1.,
        'spring_const': 1.,
        'atom_radius': 0.5,
        'epsilon': 1.,
        'boltzmann_const': 1.,
        'temperature': 1.,
        'max_dist': 3.,
        'start_dist': 1.,
        'trial_no': 10000000,
        'plot_period': 100,
        'max_plot_buffer': 10000,
        'verbose': False,
        'colors': None,
        'fig_out': 'sim.png',
        'print_period': 1000,
        'chain_out': None
    }
    simulate_equil(**sim_params)


if __name__ == '__main__':
    sandbox()
