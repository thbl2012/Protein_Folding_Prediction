import os
import time
import numpy as np

from matplotlib import pyplot

from atom_chain import AtomChain, straight_chain, wild_type
from item import get_filename, to_item, load_or_make_items, save_items, iadd_items

# Storage parameters
STATUS_FILE = 'status.txt'
STT_PREFIX = 'atoms'
FIG_PREFIX = 'fig'

# Graphic parameters
COLORS = {'energy': 'k', 'spring': 'r', 'lennard': 'g', 'coulomb': 'b'}


def simulate_sandbox(
        length,
        chain_type='wild_type',
        spring_len=1,
        spring_const=1,
        atom_radius=1,
        epsilon=1,
        boltzmann_const=1,
        temperature=1,
        max_dist=1,
        start_dist=1,
        trial_no=10000000,
        plot_period=10,
        max_plot_buffer=10000,
        show=False,
        verbose=False,
        fig_out=None,
        print_period=0,
):
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
        max_dist=max_dist,
        start_dist=start_dist,
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
        accepted = chain.mutate()
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
                pyplot.plot(trial_plot, plot, label=label, color=COLORS[label], **kwargs)
            for _, plot in plots.items():
                plot.fill(0)
            trial_plot.fill(0)
            plot_count = 0

    # Post simulation
    print(accepted_count / trial_no)
    if fig_out:
        pyplot.savefig(fig_out, format='png', dpi=600)
    if show:
        pyplot.show()
    return


def simulate_true(
        length,
        chain_type='wild_type',
        run_id=0,
        spring_len=1,
        spring_const=1,
        atom_radius=1,
        epsilon=1,
        boltzmann_const=1,
        temperature=1,
        max_dist=1,
        start_dist=1,
        trial_no=10000000,
        save_period=1000,
        records_per_file=100,
        save_dir=None,
        print_period=0,
):
    # Set up simulation
    charges = None
    if chain_type == 'wild_type':
        charges = wild_type(length)
    if save_dir is None:
        save_dir = 'D:/Coding Projects/Python/FYP/run'
    os.makedirs('{}/run_{:04d}'.format(save_dir, run_id), exist_ok=True)
    chain = AtomChain(
        straight_chain(len(charges), start_dist),
        charges,
        spring_const=spring_const,
        spring_len=spring_len,
        atom_radius=atom_radius,
        epsilon=epsilon,
        max_dist=max_dist,
        start_dist=start_dist,
        boltzmann_const=boltzmann_const,
        temperature=temperature
    )
    t = 0
    accepted_count = 0
    threshold = save_period * records_per_file
    num_records = trial_no // save_period
    num_files = int(np.ceil(num_records / records_per_file))
    num_features = len(charges) * 3

    # Begin simulation
    for file_count in range(1, num_files+1):
        filename = get_filename(save_dir, run_id, file_count, threshold)
        items = load_or_make_items(filename, num_features, t+save_period,
                                   t+threshold+save_period, save_period)
        for i in range(threshold):
            accepted = chain.mutate()
            accepted_count += accepted
            t += 1
            if t % save_period == 0:
                iadd_items(items[i//save_period], to_item(t, chain.atoms))
            if print_period > 0 and t % print_period == 0:
                print(t)
            if t == trial_no:
                break
        save_items(filename, items)
    return


def simulate_repeat(
        length,
        chain_type='wild_type',
        run_id=0,
        spring_len=1,
        spring_const=1,
        atom_radius=1,
        epsilon=1,
        boltzmann_const=1,
        temperature=1,
        max_dist=1,
        start_dist=1,
        trial_no=10000000,
        save_period=1000,
        records_per_file=100,
        save_dir=None,
        print_period=0,
        num_repeat=1,
):
    for i in range(num_repeat):
        simulate_true(
            length=length,
            chain_type=chain_type,
            run_id=run_id,
            spring_len=spring_len,
            spring_const=spring_const,
            atom_radius=atom_radius,
            epsilon=epsilon,
            boltzmann_const=boltzmann_const,
            temperature=temperature,
            max_dist=max_dist,
            start_dist=start_dist,
            trial_no=trial_no,
            save_period=save_period,
            records_per_file=records_per_file,
            save_dir=save_dir,
            print_period=print_period,
        )
    return


def simulate(
        length,
        chain_type='wild_type',
        run_id=0,
        sandbox=True,
        spring_len=1,
        spring_const=1,
        atom_radius=1,
        epsilon=1,
        boltzmann_const=1,
        temperature=1,
        max_dist=1,
        start_dist=1,
        trial_no=10000000,
        save_period=1000,
        records_per_file=100,
        save_dir=None,
        plot_period=10,
        max_plot_buffer=10000,
        show=False,
        verbose=False,
        fig_out=None,
        print_period=0,
):
    if sandbox:
        simulate_sandbox(
            length,
            chain_type=chain_type,
            spring_len=spring_len,
            spring_const=spring_const,
            epsilon=epsilon,
            boltzmann_const=boltzmann_const,
            temperature=temperature,
            max_dist=max_dist,
            start_dist=start_dist,
            trial_no=trial_no,
            plot_period=plot_period,
            max_plot_buffer=max_plot_buffer,
            show=show,
            verbose=verbose,
            fig_out=fig_out,
            print_period=print_period,
        )
    else:
        simulate_true(
            length,
            chain_type=chain_type,
            run_id=run_id,
            spring_len=spring_len,
            spring_const=spring_const,
            atom_radius=atom_radius,
            epsilon=epsilon,
            boltzmann_const=boltzmann_const,
            temperature=temperature,
            max_dist=max_dist,
            start_dist=start_dist,
            trial_no=trial_no,
            save_period=save_period,
            records_per_file=records_per_file,
            save_dir=save_dir,
            print_period=print_period,
        )
    return


def main():
    model_params = {
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
    }
    plot_params = {
        'plot_period': 100,
        'max_plot_buffer': 100000,
        'verbose': True,
        'fig_out': 'figs/test.png',
    }
    run_params = {
        'trial_no': 10 ** 6,
        'save_period': 1000,
        'records_per_file': 100,
        'save_dir': 'D:/Coding Projects/Python/FYP/run',
        'print_period': 0,
    }
    start = time.time()
    simulate(sandbox=False, **model_params, **plot_params, **run_params)
    print('Simulation done in {:.2f} minutes'.format((time.time() - start) / 60))


if __name__ == '__main__':
    main()
