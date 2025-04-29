#cython: boundscheck=False, wraparound=False, nonecheck=False
# distutils: language = c++

import numpy as np
import cython

from utils import is_conflict_free_gusfield_and_get_two_columns_in_coflicts

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from libc.limits cimport INT_MIN, INT_MAX
cimport numpy as cnp

cnp.import_array()

DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t
ctypedef pair[pair[int, int], pair[int, int]] pair_of_pairs_t

cdef vector[pair_of_pairs_t] get_conflict_edgelist(cnp.ndarray[DTYPE_t, ndim=2] A) noexcept:
    """Given a matrix A of mutations and cell samples, return an edgelist of
    corresponding to a vertex cover instance of the matrix A. I.e., edges occur
    between zeros in conflict.
    """
    cdef int n = A.shape[1]
    cdef int m = A.shape[0]
    cdef vector[pair_of_pairs_t] edge_list 

    cdef int p, q
    cdef int row1, row2
    cdef int has_one_one = 0
    cdef vector[int] zero_ones, one_zeros

    for p in range(n):
        for q in range(p+1, n):
            for i in range(m):
                if A[i, p] and A[i, q]:
                    has_one_one = 1
                elif A[i, p] == 1 and A[i, q] == 0:
                    one_zeros.push_back(i)
                elif A[i, p] == 0 and A[i, q] == 1:
                    zero_ones.push_back(i)

            if has_one_one == 1:
                for row1 in zero_ones:
                    for row2 in one_zeros:
                        edge_list.push_back(
                                pair_of_pairs_t(pair[int, int](row1, p), pair[int, int] (row2, q))
                        )
                        # edge_list.push_back(pair[int, int](row2, q))
            one_zeros.clear()
            zero_ones.clear()
            has_one_one = 0

    print("Edge list size: ", len(edge_list))
    return edge_list

cdef int min_unweighted_vertex_cover_from_edgelist(vector[pair_of_pairs_t] edge_list) noexcept:
    """For unweightred case, no need to use local ratio techniques used in
    networkx.algorithms.min_weighted_vertex_cover().
    """
    # Will hash the pairs manualy
    cdef unordered_set[int] cover

    # TODO: Use a random device and swaps to shuffle the array instead of calling to numpy
    cdef int num_edges = edge_list.size()
    cdef cnp.ndarray[DTYPE_t, ndim=1] ixs = np.random.permutation(num_edges)
    cdef int i, lrow, lcol, rrow, rcol
    cdef pair[int, int] lpair, rpair
    cdef int lhash, rhash
    for i in range(num_edges):
        # u, v = edge_list[i]
        lpair = edge_list[ixs[i]].first
        rpair = edge_list[ixs[i]].second
        lhash = lpair.first * num_edges + lpair.second
        rhash = rpair.first * num_edges + rpair.second
        if cover.find(lhash) == cover.end() or cover.find(rhash) == cover.end():
            cover.insert(lhash)
            cover.insert(rhash)

    if cover.size() % 2 == 0:
        return cover.size() // 2
    return cover.size() // 2 + 1

cdef (int, vector[pair[int, int]]) vertex_cover_ub_greedy(cnp.ndarray[DTYPE_t, ndim=2] A) noexcept:
    cdef int m = A.shape[0], n = A.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=1] ixs = np.random.permutation(n)

    cdef int i, j, p, q, num_flips, row, col
    cdef int has_one_one = 0
    cdef vector[int] one_zeros, zero_ones
    cdef vector[pair[int, int]] flips

    for p in range(n):
        for q in range(p+1, n):
            for i in range(m):
                if A[i, p] and A[i, q]:
                    has_one_one = 1
                elif A[i, p] and not A[i, q]:
                    one_zeros.push_back(i)
                elif not A[i, p] and A[i, q]:
                    zero_ones.push_back(i)

            if has_one_one and zero_ones.size() > 0 and one_zeros.size() > 0:
                # Resolve p, q conflict by flipping the cheaper of the two
                # TODO: Implement randomized later if desired
                if zero_ones.size() < one_zeros.size():
                    for i in zero_ones:
                        A[i, p] = 1
                        flips.push_back(pair[int, int](i, p))
                else:
                    for i in one_zeros:
                        A[i, q] = 1
                        flips.push_back(pair[int, int](i, q))

            has_one_one = 0
            one_zeros.clear()
            zero_ones.clear()

    num_flips = flips.size()

    # TODO: Implement me in cython
    is_cf, col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(A, 3)

    # Undo the matrix flips
    for i in range(flips.size()):
        row = flips[i].first
        col = flips[i].second
        A[row, col] = 0

    if is_cf:
        return num_flips, flips
    else:
        return -1, flips

def vertex_cover_init(cnp.ndarray[DTYPE_t, ndim=2] A):
    cdef int m = A.shape[0], n = A.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=1] ixs = np.random.permutation(n)

    cdef int i, j, p, q, num_flips, row, col
    cdef int has_one_one = 0
    cdef vector[int] one_zeros, zero_ones
    cdef vector[pair[int, int]] flips
    cdef int done = 0

    while not done:
        for p in range(n):
            for q in range(p+1, n):
                for i in range(m):
                    if A[i, p] and A[i, q]:
                        has_one_one = 1
                    elif A[i, p] and not A[i, q]:
                        one_zeros.push_back(i)
                    elif not A[i, p] and A[i, q]:
                        zero_ones.push_back(i)

                if has_one_one and zero_ones.size() > 0 and one_zeros.size() > 0:
                    # Resolve p, q conflict by flipping the cheaper of the two
                    # TODO: Implement randomized later if desired
                    if zero_ones.size() < one_zeros.size():
                        for i in zero_ones:
                            A[i, p] = 1
                            flips.push_back(pair[int, int](i, p))
                    else:
                        for i in one_zeros:
                            A[i, q] = 1
                            flips.push_back(pair[int, int](i, q))

                has_one_one = 0
                one_zeros.clear()
                zero_ones.clear()

        # TODO: Implement me in cython
        is_cf, col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(A, 3)
        if is_cf:
            done = 1

    num_flips = flips.size()

    return flips


# TODO: Wrap the above functions in a get_bounds() function
def get_bounds(cnp.ndarray[DTYPE_t, ndim=2] A, int iterations = 1):
    cdef int best_lb = 0
    cdef int best_ub = INT_MAX
    cdef int i, lb, greedy_ub
    cdef vector[pair_of_pairs_t] edge_list = get_conflict_edgelist(A)
    cdef vector[pair[int, int]] flips
    for i in range(iterations):
        # Have the second returned value be the upper bound
        lb = min_unweighted_vertex_cover_from_edgelist(edge_list)
        best_lb = max(lb, best_lb)

        greedy_ub, flips = vertex_cover_ub_greedy(A)
        if greedy_ub >= 0:
            best_ub = min(greedy_ub, best_ub)

    return best_lb, best_ub, flips
