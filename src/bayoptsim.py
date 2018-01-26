from matplotlib import use
from numpy.random import seed
import GPyOpt as gpyopt
from scipy.spatial import distance_matrix

from src.util import *

K = 1000


def total_energy(prev_atoms, next_atoms,
                 inter_charge_matrix, intra_charge_matrix,
                 spring_const=1., spring_len=1.,
                 atom_radius=1., epsilon=1.,):
    inter_dist = distance_matrix(next_atoms, prev_atoms)
    # inter_dist[inter_dist < 1e-6] = np.inf
    intra_dist = squareform(pdist(next_atoms), checks=False)
    # intra_dist[intra_dist < 1e-6] = np.inf
    spring = spring_energy(intra_dist, spring_len, spring_const) \
             + spring_const * (inter_dist[0, -1] - spring_len) ** 2
    temp = atom_radius / inter_dist
    temp = temp ** 6
    lennard = lennard_energy(intra_dist, atom_radius, epsilon) \
              + 4 * epsilon * np.sum(temp * temp - temp)
    coulomb1 = coulomb_energy(intra_dist, intra_charge_matrix)
    coulomb2 = np.sum(inter_charge_matrix / inter_dist)
    coulomb = coulomb1 + coulomb2
    return spring + lennard + coulomb + K


def next_opt_chain(prev_atoms, length,
                   inter_charge_matrix, intra_charge_matrix,
                   spring_const=1.0, spring_len=1.0,
                   atom_radius=1.0, epsilon=1.0,
                   search_radius=0.0, search_radius_ratio=1.0,
                   max_iter=10, max_time=60, eps=1e-6, acq='EI'):
    if search_radius <= 0.0:
        search_radius = spring_len * search_radius_ratio

    def f(coords):
        next_atoms = coords.reshape(-1, 3)
        return total_energy(prev_atoms, next_atoms,
                            inter_charge_matrix, intra_charge_matrix,
                            spring_const=spring_const, spring_len=spring_len,
                            atom_radius=atom_radius, epsilon=epsilon)
    bounds = []
    x, y, z = prev_atoms[-1]
    for i in range(1, length+1):
        bounds.append({'name': 'x{:02d}'.format(i),
                       'type': 'continuous',
                       'domain': (x - search_radius*i, x + search_radius*i)})
        bounds.append({'name': 'y{:02d}'.format(i),
                       'type': 'continuous',
                       'domain': (y - search_radius*i, y + search_radius*i)})
        bounds.append({'name': 'z{:02d}'.format(i),
                       'type': 'continuous',
                       'domain': (z - search_radius*i, z + search_radius*i)})

    bo = gpyopt.methods.BayesianOptimization(f, domain=bounds, acquisition_type=acq, exact_feval=True)
    bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
    return bo.x_opt.reshape(-1, 3), bo.fx_opt[0]


def test():
    sim_params = {
        'spring_const': 1., 'spring_len': 1.,
        'atom_radius': 1., 'epsilon': 1.,
    }
    opt_params = {
        'search_radius': 2., 'search_radius_ratio': 1.,
        'max_iter': 100, 'max_time': 300, 'eps': 1e-6, 'acq': 'EI',
    }
    a = 10
    l = 6
    charges = wild_type(a)
    charge_matrix = pairwise_charges(charges)
    seq1 = straight_chain(l, start_dist=1.)
    seq2, f2 = next_opt_chain(seq1, a-l, charge_matrix[l:, :l], charge_matrix[l:, l:], **sim_params, **opt_params)
    seq = np.concatenate((seq1, seq2), axis=0)
    e1 = hamiltonian(pairwise_dist(seq1), charge_matrix[:l, :l], **sim_params)
    e2 = total_energy(seq1, seq2, charge_matrix[l:, :l], charge_matrix[l:, l:], **sim_params)
    e = hamiltonian(pairwise_dist(seq), charge_matrix, **sim_params)
    print(seq)
    print('e2: {:.3f} = f2: {:.3f}'.format(e2, f2[0]))
    print('{:.3f} + {:.3f} = {:.3f} vs {:.3f}'.format(e1, e2, e1 + e2, e))


def optimize_chain(
        # Input chain parameters
        length,
        chain_type='wild_type',

        # Model parameters
        spring_len=1.,
        spring_const=1.,
        atom_radius=1.,
        epsilon=1.,

        # Bayesian Optimizer Parameters
        num_parts=1,
        search_radius=1.,
        search_radius_ratio=1.,
        max_iter=50,
        max_time_per_iter=300,
        eps=1e-6,
        acq='EI',
):
    charges = None
    if chain_type == 'wild_type':
        charges = wild_type(length)
    charge_matrix = pairwise_charges(charges)
    part_length = (length-1) // num_parts
    r = (length-1) % num_parts
    parts = []
    if r > 0:
        parts.append(r)
    if part_length > 0:
        for i in range(num_parts):
            parts.append(part_length)
    prev_atoms = np.zeros((1, 3), dtype=np.float)

    results = []

    for i in range(len(parts)):
        prev_length = prev_atoms.shape[0]
        next_length = parts[i]
        inter_charge_matrix = charge_matrix[prev_length: prev_length + next_length, : prev_length]
        intra_charge_matrix = charge_matrix[prev_length: prev_length + next_length,
                                            prev_length: prev_length + next_length]
        print(prev_length, next_length)
        print(inter_charge_matrix.shape, intra_charge_matrix.shape)
        next_atoms, energy = next_opt_chain(prev_atoms, next_length,
                                            inter_charge_matrix, intra_charge_matrix,
                                            spring_const=spring_const, spring_len=spring_len,
                                            atom_radius=atom_radius, epsilon=epsilon,
                                            search_radius=search_radius,
                                            search_radius_ratio=search_radius_ratio,
                                            max_iter=max_iter, max_time=max_time_per_iter,
                                            eps=eps, acq=acq)
        prev_atoms = np.concatenate((prev_atoms, next_atoms), axis=0)
        results.append(energy)

    print(results[-1])
    print(hamiltonian(pairwise_dist(prev_atoms), charge_matrix, spring_len, spring_const, atom_radius, epsilon))
    result = sum(results) - K*len(results)
    return prev_atoms, result


def main():
    model_params = {
        'spring_len': 1.,
        'spring_const': 1.,
        'atom_radius': 1.,
        'epsilon': 1.,
    }
    optimizer_params = {
        'num_parts': 5,
        'search_radius': 2.,
        'search_radius_ratio': 1.,
        'max_iter': 1000,
        'max_time_per_iter': 500,
        'eps': 1e-6,
        'acq': 'LCB'
    }
    length = 30
    atoms, result = optimize_chain(length, chain_type='wild_type', **model_params, **optimizer_params)
    print(atoms)
    print(result)
    print(hamiltonian(pairwise_dist(atoms), pairwise_charges(wild_type(length)), **model_params))


if __name__ == '__main__':
    main()
