import numpy as np


def is_conflict_free_gusfield_and_get_two_columns_in_coflicts(I, na_value):
    def sort_bin(a):
        b = np.transpose(a)
        b_view = np.ascontiguousarray(b).view(
            np.dtype((np.void, b.dtype.itemsize * b.shape[1]))
        )
        idx = np.argsort(b_view.ravel())[::-1]
        c = b[idx]
        return np.transpose(c), idx

    Ip = I.copy()
    Ip[Ip == na_value] = 0
    O, idx = sort_bin(Ip)
    # TODO: delete duplicate columns
    # print(O, '\n')
    Lij = np.zeros(O.shape, dtype=int)
    for i in range(O.shape[0]):
        maxK = 0
        for j in range(O.shape[1]):
            if O[i, j] == 1:
                Lij[i, j] = maxK
                maxK = j + 1
    # print(Lij, '\n')
    Lj = np.amax(Lij, axis=0)
    # print(Lj, '\n')
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            if O[i, j] == 1:
                if Lij[i, j] != Lj[j]:
                    return False, (idx[j], idx[Lj[j] - 1])
    return True, (None, None)


def get_effective_matrix(I, delta01, delta_na_to_1, change_na_to_0=False):
    x = np.array(I + delta01, dtype=np.int8)
    if delta_na_to_1 is not None:
        na_indices = delta_na_to_1.nonzero()
        x[na_indices] = (
            1  # should have been (but does not accept): x[na_indices] = delta_na_to_1[na_indices]
        )
    if change_na_to_0:
        x[np.logical_and(x != 0, x != 1)] = 0
    return x
