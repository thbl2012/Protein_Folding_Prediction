import numpy as np
import itertools as itt


def get_charge(name='wild_type', length=30, **kwargs):
    if name == 'wild_type':
        charges = np.empty(length)
        a = 2
        i = 0
        while i < length:
            charges[i] = 0
            charges[i + 1] = a
            a = -a
            i += 2
        return charges
    elif name == 'mutant':
        charges = get_charge('wild_type', length=length)
        charges[1] = 0
        return charges
    elif name == 'all_zeros':
        return np.zeros(length)
    elif name == 'all_twos':
        return np.full(length, 2)
    elif name == 'full':
        q = kwargs['q']
        return np.full
    else:
        raise ValueError('Charge sequence undefined')


def full(q=2., length=30, **kw):
    return cyclic((q,), length=length, **kw)


def wild_type_like(q=2., length=30, **kw):
    return cyclic((0, q, 0, -q), length=length, **kw)


def cyclic(qs, length=30, **kw):
    seq = np.empty(length, dtype=np.float)
    cyc = itt.cycle(qs)
    for i in range(length):
        seq[i] = next(cyc)
    return seq


def step(steps, **kw):
    if isinstance(steps, dict):
        steps = steps.items()
    steps = sorted(steps, key=lambda e: e[1])
    seq = np.empty(steps[-1][1], dtype=np.float)
    for i in range(len(steps)):
        q, cur = steps[i]
        prev = 0 if i == 0 else steps[i-1][1]
        seq[prev:cur] = q
    return seq


predefined = dict(
    wild_type=wild_type_like(q=2., length=30),
    mutant_alt_0p2=cyclic((0., 2.), length=30),
    mutant_p2_to_0=cyclic((0., 0., 0., -2.), length=30),
    mutant_n2_to_0=cyclic((0., 2., 0., 0.), length=30),
    mutant_all_0=full(q=0., length=30),
    mutant_all_p2=full(q=2., length=30),
)
