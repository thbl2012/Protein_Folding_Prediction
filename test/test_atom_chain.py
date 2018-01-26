from atom_chain import *


def str_chain(chain):
    return '[{:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(
        chain.spring, chain.lennard, chain.coulomb, chain.energy)


def random_position(p, s):
    rd = np.random.RandomState()
    rd.seed(0)
    return p + s*rd.random_sample(3)*2-s


def lennard_energy(dist_matrix, atom_radius, epsilon):
    with np.errstate(divide='ignore', invalid='ignore'):
        # print('>> dist_matrix')
        # print(dist_matrix)
        dist_matrix = atom_radius / dist_matrix
        # print('>> dist_matrix = atom_radius / dist_matrix')
        # print(dist_matrix)
        np.fill_diagonal(dist_matrix, 0)
        # print('>> np.fill_diagonal(dist_matrix, 0)')
        # print(dist_matrix)
        dist_matrix = dist_matrix * dist_matrix * dist_matrix * dist_matrix * dist_matrix * dist_matrix
        # print('>> dist_matrix = dist_matrix * dist_matrix * dist_matrix * dist_matrix * dist_matrix * dist_matrix')
        # print(dist_matrix)
        dist_matrix = dist_matrix * dist_matrix - dist_matrix
        # print('>> dist_matrix = dist_matrix * dist_matrix - dist_matrix')
        # print(dist_matrix)
        return np.sum(dist_matrix)*epsilon*2


def test_chain():
    l = 30
    return AtomChain(
        straight_chain(l, start_dist=10.),
        wild_type(l),
        spring_len=10.,
        spring_const=1.,
        atom_radius=10.,
        epsilon=1.,
        boltzmann_const=np.inf,
        temperature=np.inf,
    )


def test_trial_dist_vect():
    chain = test_chain()
    p = np.array([0, 0, 0])
    dv = chain.get_trial_dist_vector(p)
    # print(dv)


def test_dist_matrix():
    chain = test_chain()
    p = random_position(chain.atoms[0], 3.)
    chain.update_position(0, p, chain.get_trial_dist_vector(p))
    mat1 = chain.dist_matrix
    mat1[0][0] = 0
    mat2 = pairwise_dist(chain.atoms)
    print(mat1)
    print()
    print(mat2)
    print()
    print(np.sum((mat1 - mat2)**2))


def test_lennard_diff():
    chain = test_chain()
    print('>> initial lennard: {}'.format(chain.lennard))
    p = random_position(chain.atoms[0], 3.)
    vect = chain.get_trial_dist_vector(p)
    diff = chain.lennard_diff(0, vect)
    print('>> lennard diff: {}'.format(diff))
    print('>> new lennard by diff: {}'.format(diff + chain.lennard))
    chain.update_position(0, p, vect)
    chain.recompute()
    print('>> new lennard recomputed: {}'.format(chain.lennard))
    print()
    print(chain.dist_matrix)
    print('>> new lenard recomputation:')
    print(lennard_energy(chain.dist_matrix, chain.atom_radius, chain.epsilon))


def coulomb_energy(dist_matrix, charge_matrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_matrix = charge_matrix / dist_matrix
        np.fill_diagonal(dist_matrix, 0)
        return np.sum(dist_matrix) / 2


def test_coulomb_diff():
    chain = test_chain()
    print('>> initial coulomb: {}'.format(chain.coulomb))
    print('>> initial coulomb recomputed: {}'.format(coulomb_energy(chain.dist_matrix, chain.charge_matrix)))
    p = random_position(chain.atoms[0], 3.)
    vect = chain.get_trial_dist_vector(p)
    diff = chain.coulomb_diff(0, vect)
    print('>> coulomb diff: {}'.format(diff))
    print('>> new coulomb by diff: {}'.format(diff + chain.coulomb))
    chain.update_position(0, p, vect)
    chain.recompute()
    print('>> new coulomb recomputed: {}'.format(chain.coulomb))
    print()
    print(chain.dist_matrix)
    print('>> new coulomb recomputation:')
    print(coulomb_energy(chain.dist_matrix, chain.charge_matrix))


def test_hamiltonian():
    chain = test_chain()


def main():
    chain = test_chain()
    print(str_chain(chain))
    for i in range(100):
        acc = chain.mutate(3.)
        if acc:
            str1 = str_chain(chain)
            chain.recompute()
            str2 = str_chain(chain)
            print(str1, str2, sep='    ')


if __name__ == '__main__':

    main()
