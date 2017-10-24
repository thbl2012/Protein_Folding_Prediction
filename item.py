import numpy as np


# Deprecated
class Item:
    def __init__(self, length=0, t=0, age=0, data=None):
        if data is not None:
            self.sums, self.pairwise_prod_sums = data
            self.length = len(self.sums)
            self.age = age
            self.t = t
        else:
            self.sums = np.zeros(length)
            self.pairwise_prod_sums = np.zeros((length, length))
            self.length = length
            self.age = 0
            self.t = t

    def __add__(self, item):
        if self.length != item.length:
            raise TypeError('Unable to add: Item lengths do not match')
        elif self.t != item.t:
            raise TypeError('Unable to add: Item times do not match')
        result = Item(length=self.length, t=self.t)
        result.sums = self.sums + item.sums
        result.pairwise_prod_sums = self.pairwise_prod_sums + item.pairwise_prod_sums
        result.age = self.age + item.age
        return result

    def __iadd__(self, item):
        if self.length != item.length:
            raise TypeError('Unable to add: Item lengths do not match')
        elif self.t != item.t:
            raise TypeError('Unable to add: Item times do not match')
        self.age += item.age
        self.sums += item.sums
        self.pairwise_prod_sums += item.pairwise_prod_sums
        return self

    def __repr__(self):
        return 'Item(t={}, age={}, data=({}, {}))'.format(
            self.t,
            self.age,
            repr(self.sums),
            repr(self.pairwise_prod_sums)
        )

    def __str__(self):
        return ','.join([
            repr(self.t),
            repr(self.age),
            repr(self.length),
            ','.join([repr(x) for x in self.sums]),
            ','.join([repr(x) for x in self.pairwise_prod_sums])
        ])


def get_filename(run_id, file_count, threshold):
    return 'run_{:03d}/items_{:07d}.npy'.format(run_id, file_count * threshold)


def to_item(t, atoms):
    features = atoms.reshape(-1)
    pairwise_prods = features * features.reshape(-1, 1)
    return np.concatenate((np.array((t, len(features), 1)), features, pairwise_prods))


def save_items(filename, items):
    np.save(filename, items.T, allow_pickle=False, fix_imports=False)


def load_or_make_items(filename, length, start_t, end_t, save_period):
    try:
        items = np.load(filename, allow_pickle=False, fix_imports=False, encoding='bytes').T
    except FileNotFoundError:
        item_length = length * length + length + 3
        list_t = np.arange(start_t, end_t, save_period)
        items = np.zeros((len(list_t), item_length), dtype=np.float64)
        items[:, 1].fill(length)
        items[:, 0] = list_t
    return items


def get_t(item):
    return item[0]


def get_age(item):
    return item[2]


def get_len(item):
    return item[1]


def set_t(item, t):
    item[0] = t


def set_age(item, age):
    item[2] = age


def set_len(item, length):
    item[1] = length


def iadd_items(item1, item2):
    # item1 += item2
    assert get_t(item1) == get_t(item2), 'Unable to add, t are different'
    assert get_len(item1) == get_len(item2), 'Unable to add, lengths are different'
    item1[2:] += item2[2:]
