import numpy as np
import simulate
from atom_chain import AtomChain


def save_chain_to_file(chain: AtomChain, to_file):
    chain_params = np.array([chain.spring_const, chain.spring_len,
                             chain.epsilon, chain.atom_radius,
                             chain.boltzmann_const, chain.temperature,
                             chain.spring, chain.lennard, chain.coulomb,
                             chain.energy, chain.last_index,
                             chain.charges.shape[0]])
    chain_data = np.concatenate((chain_params, chain.charges,
                                 chain.atoms.reshape(-1),
                                 chain.charge_matrix.reshape(-1),
                                 chain.dist_matrix.reshape(-1),))
    np.save(to_file, chain_data, allow_pickle=False, fix_imports=False)


def load_chain_from_file(from_file):
    chain_data = np.load(from_file, allow_pickle=False, fix_imports=False, encoding='bytes')
    spring_const, spring_len, epsilon, atom_radius, boltzmann_const, temperature = chain_data[0:6]
    spring, lennard, coulomb, energy, last_index = chain_data[6:11]

    length = int(chain_data[11])
    length_ind = 11
    charges_end = length_ind + length + 1
    atoms_end = charges_end + length*3
    charge_matrix_end = atoms_end + length**2
    dist_matrix_end = charge_matrix_end + length**2
    charges = chain_data[10: length+10]
    atoms = chain_data[charges_end: atoms_end].reshape(-1, 3)
    charge_matrix = chain_data[atoms_end: charge_matrix_end].reshape(length, length)
    dist_matrix = chain_data[charge_matrix_end: dist_matrix_end].reshape(length, length)
    return AtomChain(atoms, charges, spring_const=spring_const, spring_len=spring_len,
                     atom_radius=atom_radius, epsilon=epsilon,
                     boltzmann_const=boltzmann_const, temperature=temperature,
                     charge_matrix=charge_matrix, dist_matrix=dist_matrix,
                     spring=spring, lennard=lennard, coulomb=coulomb,
                     energy=energy, last_index=last_index, rebuild=False
                     )


def get_sample_equil(from_file):
    samples_raw = np.load(from_file, allow_pickle=False,
                          fix_imports=False, encoding='bytes')
    return samples_raw.reshape(samples_raw.shape[0], -1, 3)


def save_samples_to_file(samples, to_file):
    np.save(to_file, samples.reshape(len(samples), -1),
            allow_pickle=False, fix_imports=False)


def save_ref_config(ref_config, to_file):
    np.save(to_file, ref_config, allow_pickle=False, fix_imports=False)


def load_ref_config(from_file):
    return np.load(from_file, allow_pickle=False, fix_imports=False).reshape(-1, 3)


def get_ref_config(samples, ref_method='centroid'):
    if ref_method == 'centroid':
        ref_config = simulate.centroid(samples)
    else:
        raise ValueError('Invalid reference method')
    return ref_config
