import os

from matplotlib import pyplot

from atom_chain import *
from item import *

# Storage parameters
STATUS_FILE = 'status.txt'
STT_PREFIX = 'atoms'
FIG_PREFIX = 'fig'

# Graphic parameters
COLORS = {'energy': 'k', 'spring': 'r', 'lennard': 'g', 'coulomb': 'b'}


def simulate_sandbox(
        charges,
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
        fig_out=None
):
    # Set up simulation
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
            if t % 1000 == 0:
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
        charges,
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
):
    # Set up simulation
    os.makedirs('run/run_{:03d}'.format(run_id), exist_ok=True)
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
    num_files = (trial_no-1)//threshold + 1
    num_features = len(charges) * 3

    # Begin simulation
    for file_count in range(1, num_files+1):
        filename = get_filename(run_id, file_count, threshold)
        items = load_or_make_items(filename, num_features, t+save_period,
                                   t+threshold+save_period, save_period)
        for i in range(threshold):
            accepted = chain.mutate()
            accepted_count += accepted
            t += 1
            if t % save_period == 0:
                items[i//save_period] += to_item(t, chain.atoms)
            if t == trial_no:
                break
        save_items(items, filename)
    return


def simulate(
        charges,
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
        plot_period=10,
        max_plot_buffer=10000,
        show=False,
        verbose=False,
        fig_out=None
):
    if sandbox:
        simulate_sandbox(
            charges,
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
            fig_out=fig_out
        )
    else:
        simulate_true(
            charges,
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
            records_per_file=records_per_file
        )
    return


if __name__ == '__main__':
    model_params = {
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
        'trial_no': 10 ** 5,
        'plot_period': 100,
        'max_plot_buffer': 100000,
        'verbose': True,
        'fig_out': 'figs/test.png',
    }
    save_params = {
        'save_period': 100,
        'records_per_file': 100,
    }
    simulate(wild_type(30), sandbox=False, **model_params, **plot_params, **save_params)
