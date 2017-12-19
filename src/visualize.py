from data import load_chain_from_file


def get_pdb_from_atoms(atoms):
    name = 'atoms.pdb'
    with open(name, 'w', encoding='utf-8') as file:
        # print("HEADER    MY ATOMS                    08-NOV-17   1A31", file=file)
        # print("TITLE     MY ATOMS", file=file)
        # print("SEQRES   1 A    9  PRO PRO GLY PRO PRO GLY PRO PRO GLY", file=file)
        # print("SEQRES   1 B    6  PRO PRO GLY PRO PRO GLY", file=file)
        # print("SEQRES   1 C    6  PRO PRO GLY PRO PRO GLY", file=file)
        for i in range(atoms.shape[0]):
            print('ATOM {} C PRO A 1 {:.3f} {:.3f} {:.3f} 1.00 1.00 C'.format(i+1, *atoms[i]), file=file)
    return name


def main():
    chain_name = 'wild_type'
    run_dir = 'run'
    equil_seed_name = 'equil_seed'
    save_dir = '{}/{}'.format(run_dir, chain_name)
    equil_seed_path = '{}/{}.npy'.format(save_dir, equil_seed_name)
    atoms = load_chain_from_file(equil_seed_path)
    filename = get_pdb_from_atoms(atoms)
    return filename


if __name__ == '__main__':
    main()
