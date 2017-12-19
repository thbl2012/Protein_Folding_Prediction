from typing import Optional

from typing.io import IO

from deprecated.old_simulate import *


def read_chain(file):
    if not file:
        file = '{}/atoms.txt'.format(os.getcwd())
    try:
        file = open(file, 'r')
        atoms = []
        for line in file:
            atoms.append(tuple(map(lambda x: float(x), re.split(',', line))))
        file.close()
    except FileNotFoundError:
        atoms = straight_chain(30)
    return AtomChain(atoms, wild_type())


def save_chain(atoms, filename):
    with open(filename, 'w') as file:  # type: Optional[IO[str]]
        for a in atoms:
            print(*a, sep=',', file=file)


def auto_simulate():
    try:
        status_file = open(STATUS_FILE, 'r')
        status = status_file.read().splitlines()
        last_status = status[0]
        last_num = int(status[1])
        status_file.close()
    except FileNotFoundError:
        last_status = None
        last_num = 0
    chain = read_chain(last_status)
    output = '{}_{:04d}.png'.format(FIG_PREFIX, last_num)
    simulate(chain, 10000000, plot_period=100, max_plot_buffer=100000, fig_out=output)
    filename = '{}_{:04d}.txt'.format(STT_PREFIX, last_num)
    save_chain(chain.atoms, filename)
    with open(STATUS_FILE, 'w') as status_file:
        print(filename, file=status_file)
        print(last_num+1, file=status_file)