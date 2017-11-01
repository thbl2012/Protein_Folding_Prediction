import numpy as np
import random
import os

from matplotlib import pyplot

from atom_chain import AtomChain, straight_chain, wild_type
from item import save_items


# Storage parameters
STATUS_FILE = 'status.txt'
STT_PREFIX = 'atoms'
FIG_PREFIX = 'fig'


def simulate_equil(
        # Input chain parameters
        length,
        chain_type='wild_type',

        # Model parameters
        spring_len=1,
        spring_const=1,
        atom_radius=1,
        epsilon=1,
        boltzmann_const=1,
        temperature=1,

        # Monte Carlo parameters
        max_dist=1,
        start_dist=1,
        trial_no=10000000,

        # Plot parameters
        plot_period=10,
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
        temperature=temperature
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
    kwargs = {'linestyle': 'None', 'markersize': 1, 'marker': '_'}

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


def sample_equil(chain, num_equiv=5000000, num_sample=1000, samples_out=None):
    samples = np.empty((num_sample, chain.atoms.shape[0], chain.atoms.shape[1]), dtype=np.float64)
    indices = np.array(random.sample(range(num_equiv), num_sample), dtype=np.int32).sort()
    i = 0
    for t in range(num_equiv):
        chain.mutate()
        if t == indices[i]:
            samples[i] = chain.atoms
            i += 1
    if samples_out:
        np.save(samples_out, samples.reshape(len(samples), -1),
                allow_pickle=False, fix_imports=False)
    return samples


def centroid(samples):
    pass


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
            states[t//save_period] = chain.atoms
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
    if out:
        np.save(out, runs.reshape(num_repeat, -1), allow_pickle=False, fix_imports=False)
    return runs
