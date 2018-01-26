from src.data import load_chain_from_file


def print_atoms_to_pdb(to_pdb, atoms, temp, atom_type='H'):
    with open(to_pdb, 'w', encoding='utf-8') as file:
        for i in range(atoms.shape[0]):
            print(get_pdb_line(atom_no=i+1, atom_name=atom_type, res_name='PRO',
                               chain_id='', res_no=i+1, x=atoms[i, 0],
                               y=atoms[i, 1], z=atoms[i, 2], occup=0.0,
                               temp=temp, symb=atom_type), file=file)
    return


def get_pdb_line(
        atom_no, atom_name, res_name,
        chain_id, res_no, x, y, z,
        occup, temp, symb,
):
    s1 = 'ATOM  {:5d} {:4s} {:>3s} {:1s}{:4d}    '.format(
        atom_no, atom_name, res_name, chain_id, res_no
    )
    s2 = '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:1s}'.format(
        x, y, z, occup, temp, symb
    )
    return s1 + s2


def main():
    run_dir = 'run/data'
    equil_seed_name = 'equil_seed'
    equil_seed_path = '{}/{}.npy'.format(run_dir, equil_seed_name)
    chain = load_chain_from_file(equil_seed_path)
    print_atoms_to_pdb('atoms.pdb', chain.atoms, temp=1.0, atom_type='H')


if __name__ == '__main__':
    main()
