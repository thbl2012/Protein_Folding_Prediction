import numpy as np
from scipy.spatial.distance import pdist, squareform

# def random_distance(s):
#     return Random().uniform(-s, s)


def random_position(p, s):
    return p + s*np.random.RandomState().random_sample(3)*2-s


def straight_chain(length, start_pos=None, start_dist=1.0):
    if start_pos is None:
        start_pos = np.zeros(3)
    chain = np.zeros((length, 3))
    for i in range(length):
        chain[i] = start_pos
        chain[i][2] += start_dist * i
    return chain


def pairwise_dist(atoms):
    return squareform(pdist(atoms))


def inv_pairwise_dist(atoms):
    dist_matrix = pairwise_dist(atoms)
    dist_matrix[dist_matrix == 0] = np.inf
    return 1./dist_matrix


def pairwise_charges(charges):
    return charges.reshape(-1, 1)*charges


def spring_energy(dist_matrix, spring_len, spring_const):
    return np.sum(np.square(np.diagonal(dist_matrix, 1) - spring_len))*spring_const


def lennard_energy(dist_matrix, atom_radius, epsilon):
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_matrix = atom_radius / dist_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = dist_matrix * dist_matrix * dist_matrix * dist_matrix * dist_matrix * dist_matrix
        return np.sum(dist_matrix * dist_matrix - dist_matrix)*epsilon*2


def coulomb_energy(dist_matrix, charge_matrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_matrix = charge_matrix / dist_matrix
        np.fill_diagonal(dist_matrix, 0)
        return np.sum(dist_matrix) / 2


def hamiltonian(dist_matrix, charge_matrix, spring_len, spring_const, atom_radius, epsilon):
    return (spring_energy(dist_matrix, spring_len, spring_const)
            + lennard_energy(dist_matrix, atom_radius, epsilon)
            + coulomb_energy(dist_matrix, charge_matrix))


def wild_type(length):
    charges = np.zeros(length)
    a = 2
    i = 0
    while i < length:
        charges[i] = 0
        charges[i+1] = a
        a = -a
        i += 2
    return charges

