from src.bayoptsim import *


def test():
    params = {
        'spring_const': 1.,
        'spring_len': 1.,
        'atom_radius': 1.,
        'epsilon': 1.,
    }
    a = 10
    l = 2
    charges = wild_type(a)
    charge_matrix = pairwise_charges(charges)
    seq1 = straight_chain(l, start_dist=1.)
    seq2 = straight_chain(a+1-l, start_pos=seq1[-1], start_dist=1)[1:]
    seq = np.concatenate((seq1, seq2), axis=0)
    print(seq)
    e1 = hamiltonian(pairwise_dist(seq1), charge_matrix[:l, :l], **params)
    e2 = total_energy(seq1, seq2, charge_matrix[l:, :l], charge_matrix[l:, l:], **params)
    e = hamiltonian(pairwise_dist(seq), charge_matrix, **params)
    print('{:.3f} + {:.3f} = {:.3f} vs {:.3f}'.format(e1, e2, e1+e2, e))


def foo(*x):
    print(x)
    return gpyopt.objective_examples.experiments1d.forrester().f(x[0])


def test_gpyopt():
    bounds = [{'name': 'x', 'type': 'continuous', 'domain': (0, 1)}]
    bo = gpyopt.methods.BayesianOptimization(foo, domain=bounds, acquisition_type='EI', exact_feval=True)
    max_iter = 15
    max_time = 60
    eps = 1e-6
    bo.run_optimization(max_iter, max_time, eps)
    bo.plot_acquisition()


if __name__ == '__main__':
    test()
