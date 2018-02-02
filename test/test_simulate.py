import numpy as np
from matplotlib import pyplot
from src.atom_chain import AtomChain, straight_chain, wild_type


def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None


def test_simulate_equil(
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
        enable_plot=True,
        plot_window=100,
        plot_period=100,
        # max_plot_buffer=10000,
        # verbose=False,
        colors=None,
        fig_out=None,

        # Info parameters
        print_period=0,
        chain_out=None
):
    # Set up colors
    if not colors:
        colors = {'p': 'b', 'r': 'r'}

    # Set up simulation
    charges = None
    if chain_type == 'wild_type':
        charges = wild_type(length)
    chain = AtomChain(
        straight_chain(len(charges), start_dist=start_dist),
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
    trial_plot = np.empty(plot_window)
    labels = ['acc', 'p', 'r', 'old', 'new', 'd']
    plots = {}
    for label in labels:
        plots[label] = np.empty(plot_window, dtype=np.float)
    kwargs = {'linestyle': '-', 'markersize': 1, 'marker': '+'}

    # Begin simulation
    for t in range(1, trial_no + plot_window + 1):
        stats = chain.test_mutate(max_dist)
        accepted_count += stats['acc']
        if print_period > 0 and t % print_period == 0:
            print(t)
        if not enable_plot or t < trial_no:
            continue
        # Code for plotting
        if t % plot_period == 0:
            if stats['acc'] > 0:
                for label, plot in plots.items():
                    plot[plot_count] = stats[label]
                plot_count += 1
                trial_plot[plot_count] = plot_count
        #
        # if plot_count == max_plot_buffer or t == trial_no:
        #     for label in plots.keys():
        #         plots[label] = plots[label][:plot_count]
        #     trial_plot = trial_plot[:plot_count]
        #
        #     for label, plot in plots.items():
        #         plot[plot > 200] = 200
        #         pyplot.plot(trial_plot, plot, label=label, color=colors[label], **kwargs)
        #     # pyplot.plot(trial_plot, plot, label='diff', color='k', **kwargs)
        #     plot_count = 0
        #     print(trial_plot)

    print(plot_count)

    for label in plots.keys():
        plots[label] = plots[label][:plot_count]
    trial_plot = trial_plot[:plot_count]

    plots['p'][plots['p'] > 1.] = 1.

    fig, axes = pyplot.subplots(nrows=1, ncols=1)
    # ax0 = axes[0]
    # ax0.plot(trial_plot, plots['p'], label='p', color='b', **kwargs)
    # ax0.plot(trial_plot, plots['r'], label='r', color='r', **kwargs)
    # ax0.legend()

    ax1 = axes
    ax1.plot(trial_plot, plots['new'] - plots['old'], color='r', label='diff', **kwargs)
    ax2 = ax1.twinx()
    color_y_axis(ax1, 'r')
    ax2.plot(trial_plot, plots['d'], color='b', label='d', **kwargs)
    color_y_axis(ax2, 'b')
    ax1.legend(loc=0)
    ax2.legend(loc=0)

    # Save figure and chain
    if fig_out:
        fig.savefig(fig_out, format='png', dpi=600)
    if chain_out:
        np.save(chain_out, chain.atoms, allow_pickle=False, fix_imports=False)
    return chain


def main():
    sim_params = {
        'length': 30,
        'chain_type': 'wild_type',
        'spring_len': 10.,
        'spring_const': 1.,
        'atom_radius': 1.,
        'epsilon': 1.,
        'boltzmann_const': 1.,
        'temperature': 1.,
        'max_dist': 3.,
        'start_dist': 10.,
        'trial_no': 1000000,
        'plot_window': 1000,
        'plot_period': 1,
        # 'max_plot_buffer': 100,
        'colors': None,
        'fig_out': 'sim.png',
        'print_period': 10000,
        'chain_out': None
    }
    return test_simulate_equil(**sim_params)


if __name__ == '__main__':
    main()
