import numpy as np
from numba import guvectorize

from src.atom_chain import AtomChain


def test_simulate_short(
        atoms, charges,
        charge_matrix, dist_matrix,
        spring_const, spring_len,
        atom_radius, epsilon,
        boltzmann_const, temperature,
        lennard, spring, coulomb,
        energy,
        max_dist=1.,
        trial_no=1000,
        save_period=np.arange(50, 1050, 50, dtype=np.int32),
):
    chain = AtomChain(
        atoms, charges,
        spring_const=spring_const,
        spring_len=spring_len,
        atom_radius=atom_radius,
        epsilon=epsilon,
        boltzmann_const=boltzmann_const,
        temperature=temperature,
        dist_matrix=np.copy(dist_matrix),
        charge_matrix=charge_matrix,
        spring=spring,
        lennard=lennard,
        coulomb=coulomb,
        energy=energy,
        rebuild=False
    )
    states = np.empty(
        (save_period.shape[0], atoms.shape[0], atoms.shape[1]),
        dtype=np.float64
    )
    i = 0
    for t in range(1, trial_no+1):
        rand = np.random.RandomState()
        i = rand.randint(0, len(atoms))
        p = random_position(atoms[i], max_dist)

        # Compute energy difference
        v = self.get_trial_dist_vector(p)
        spring_diff = self.spring_diff(i, v)
        lennard_diff = self.lennard_diff(i, v)
        coulomb_diff = self.coulomb_diff(i, v)
        energy_diff = spring_diff + lennard_diff + coulomb_diff

        # Generate acceptance decision
        accepted = 0
        if rand.uniform(0, 1) < np.exp(- energy_diff / (self.boltzmann_const * self.temperature)):
            self.update_position(i, p, v)
            self.energy += energy_diff
            self.spring += spring_diff
            self.lennard += lennard_diff
            self.coulomb += coulomb_diff
            accepted = 1
        self.last_index = i
        if t == save_period[i]:
            states[i] = chain.atoms
            i += 1
    return states


signature = 'void(float64[:,:], float64[:], float64[:,:], float64[:,:]'
signature += ', float64' * 11
signature += ', int32, int32[:], int32[:], float64[:,:,:,:])'
layout = '(na,al),(na),(na,na),(na,na)' + ',()'*12 + ',(nr),(ns)->(nr,ns,na,al)'


@guvectorize([signature], layout, target='cuda')
def test_repeat_simulate_short(
        atoms, charges,
        charge_matrix, dist_matrix,
        spring_const, spring_len,
        atom_radius, epsilon,
        boltzmann_const, temperature,
        lennard, spring, coulomb,
        energy, max_dist,
        trial_no, vec_num_repeat, vec_save_period,
        runs,
):
    for i in range(vec_num_repeat.shape[0]):
        runs[i] = test_simulate_short(atoms, charges,
                                      charge_matrix, dist_matrix,
                                      spring_const, spring_len,
                                      atom_radius, epsilon,
                                      boltzmann_const, temperature,
                                      lennard, spring, coulomb,
                                      energy, max_dist=max_dist,
                                      trial_no=trial_no, save_period=vec_save_period)
