import multiprocessing as mp
import pandas as pd
from scipy.spatial.distance import *
from scipy.spatial import distance_matrix
from scipy.linalg import svd as spsvd
from sklearn.metrics.pairwise import euclidean_distances

from util import inv_pairwise_dist
from deprecated.item import *


def dist_matrix_scipy_1(npl):
    return squareform(pdist(npl))


def dist_matrix_scipy_2(npl):
    return distance_matrix(npl, npl)


def dist_matrix_sklearn(npl):
    return euclidean_distances(npl)


def dist_matrix_np(npl):
    n, m = npl.shape
    dists = np.zeros((n, n))
    for d in range(m):
        d_dim = npl[:, d]
        dists += np.square(d_dim - d_dim[:, None])
    return np.sqrt(dists)


def auto_test(opt=None):
    if opt:
        eval('dist_matrix_{}(np.array(straight_chain(30, 1)))'.format(opt))
    else:
        eval('dist_matrix(straight_chain(30, 1))')


def lennard1(dom, m):
    with np.errstate(divide='ignore', invalid='ignore'):
        m = dom/m
        m[~np.isfinite(m)] = 0
        return np.sum(np.power(m, 12) - np.power(m, 6))/2


def lennard2(dom, m):
    with np.errstate(divide='ignore', invalid='ignore'):
        m = dom/m
        m[~np.isfinite(m)] = 0
        m **= 6
        return np.sum(np.power(m, 2) - m)/2


def lennard3(dom, m):
    m1 = squareform(m)
    m1 **= -1
    m1 *= dom
    m1 **= 6
    return np.sum(m1*m1 - m1)


def lennard4(dom, m):
    m1 = (dom/squareform(m))
    m1 = m1*m1*m1*m1*m1*m1
    return np.sum(m1*m1 - m1)


def item(dom, x):
    if x == 0:
        return 0
    else:
        y = (dom/x)**6
        return y*y - y


def lennard5(dom, m):
    with np.errstate(divide='ignore', invalid='ignore'):
        m = dom/m
        np.fill_diagonal(m, 0)
        m = m*m*m*m*m*m
        return np.sum(m*m - m)/2


def lennard6(dom, m):
    with np.errstate(divide='ignore', invalid='ignore'):
        m = dom/m
        np.fill_diagonal(m, 0)
        m = m*m*m*m*m*m
        return (np.sum(m*m) - np.sum(m))/2


f = np.random.random_sample(90)
p = f * f.reshape(-1, 1)
fn = 'test_print.out'


def print1():
    with open(fn, 'w', encoding='utf-8') as file:
        print(*f, *p.reshape(-1), sep=',', file=file)


def print2():
    with open(fn, 'w', encoding='utf-8') as file:
        print(','.join([','.join([repr(x) for x in f]), ','.join([repr(x) for x in p.reshape(-1)])]), file=file)


def print3():
    with open(fn, 'w', encoding='utf-8') as file:
        print(','.join([repr(x) for x in f]), ','.join([repr(x) for x in p.reshape(-1)]), sep=',', file=file)


def print4():
    with open(fn, 'w', encoding='utf-8') as file:
        print(','.join([repr(x) for x in np.concatenate((f, p.reshape(-1)))]), file=file)


def load_np_fromfile():
    return np.fromfile('test_load_fromfile.csv', dtype=np.float64, sep=',')


def load_np_loadtxt():
    return np.loadtxt('test_load.csv', dtype=np.float64, delimiter=',')


def load_pd_read_csv():
    return pd.read_csv('test_load.csv', dtype=np.float64, delimiter=',', header=None).values


def load_np_load():
    return np.load('test_load.npy', allow_pickle=False, fix_imports=False, encoding='bytes')


def get_concat_items():
    a = np.random.random(1000)
    b = np.random.random(9000)
    return a, b


def concat1(a, b):
    return np.concatenate((np.array([1, 2, 3]), a, b))


def concat2(a, b, la, lb):
    x = np.empty(la + lb + 3)
    x[:1] = 1
    x[1:2] = 2
    x[2:3] = 3
    x[3:(la+3)] = a
    x[(la+3):(la+lb+3)] = b
    return x


def repeat(s, num=1):
    print(s * num)


def irepeat(args):
    s, kw = args
    repeat(s, kw.get('num'))


def multiproc_kwarg():
    pool = mp.Pool(3)
    kw = {'num': 5}
    pool.map(irepeat, [('a', kw), ('b', kw), ('c', kw)])


def svd_scipy(a):
    return spsvd(a, full_matrices=False, check_finite=False)


def svd_numpy(a):
    return np.linalg.svd(a, full_matrices=False)


def inv_dist_mat_1(a):
    return inv_pairwise_dist(a)


def inv_dist_mat_2(a):
    dist_mat = np.empty((a.shape[0], a.shape[0]))
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            dist_mat[i, j] = np.sum(np.square(a[i] - a[j]))
    np.fill_diagonal(dist_mat, np.inf)
    return np.invsqrt(dist_mat)

