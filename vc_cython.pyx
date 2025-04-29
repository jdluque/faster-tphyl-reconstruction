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

def get_conflict_edgelist(cnp.ndarray[DTYPE_t, ndim=2] A):
    """Given a matrix A of mutations and cell samples, return an edgelist of
    corresponding to a vertex cover instance of the matrix A. I.e., edges occur
    between zeros in conflict.
    """
    cdef int n = A.shape[1]
    cdef int m = A.shape[0]
    cdef vector[pair[int, int]] edge_list 

    cdef int p, q
    cdef int row1, row2
    cdef int has_one_one
    cdef vector[int] zero_ones, one_zeros

    for p in range(n):
        for q in range(p+1, n):
            for i in range(m):
                if A[i, p] and A[i, q]:
                    has_one_one = 1
                elif A[i, p] and not A[i, q]:
                    one_zeros.push_back(i)
                elif not A[i, p] and A[i, q]:
                    zero_ones.push_back(i)

            if not has_one_one:
                continue

            for row1 in zero_ones:
                for row2 in one_zeros:
                    edge_list.push_back(pair[int, int](row1, p))
                    edge_list.push_back(pair[int, int](row2, q))
            one_zeros.clear()
            zero_ones.clear()

    return edge_list

cdef int min_unweighted_vertex_cover_from_edgelist(vector[pair[int, int]] edge_list):
    """For unweightred case, no need to use local ratio techniques used in
    networkx.algorithms.min_weighted_vertex_cover().
    """
    cdef unordered_set[int] cover

    # TODO: Use a random device and swaps to shuffle the array instead of calling to numpy
    cdef int num_edges = edge_list.size()
    cdef cnp.ndarray[DTYPE_t, ndim=1] ixs = np.random.permutation(num_edges)
    cdef int i, u, v
    for i in range(num_edges):
        # u, v = edge_list[i]
        u = edge_list[ixs[i]].first
        v = edge_list[ixs[i]].second
        if cover.find(u) == cover.end() or cover.find(v) == cover.end():
            cover.insert(u)
            cover.insert(v)

    if cover.size() % 2 == 0:
        return cover.size() // 2
    return cover.size() // 2 + 1


def vertex_cover_ub_greedy(cnp.ndarray[DTYPE_t, ndim=2] A):
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
        return int(num_flips)
    else:
        print("not conflict free :(")
        return float("inf")

# TODO: Wrap the above functions in a get_bounds() functiondef get_bounds(A: np.ndarray, iterations: int = 1):
def get_bounds(cnp.ndarray[DTYPE_t, ndim=2] A, int iterations = 1):
    cdef int best_lb = 0
    cdef int best_ub = INT_MAX
    cdef int i, lb, greedy_ub
    cdef vector[pair[int, int]] edge_list = get_conflict_edgelist(A)
    for i in range(iterations):
        # Have the second returned value be the upper bound
        lb = min_unweighted_vertex_cover_from_edgelist(edge_list)
        print("lb found ", lb)
        best_lb = max(lb, best_lb)

        greedy_ub = vertex_cover_ub_greedy(A)
        best_ub = min(greedy_ub, best_ub)

    return best_lb, best_ub
