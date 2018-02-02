import numpy as np
from numba import guvectorize


@guvectorize(['void(float64[:], float64[:])'], '(n)->(n)', target='cuda', nopython=True)
def foo(a, b):
    for i in range(a.shape[0]):
        b[i] = a[i] + 1. # np.random.random()


if __name__ == '__main__':
    print(foo(np.arange(1, 10, 2, dtype=np.float64)))
