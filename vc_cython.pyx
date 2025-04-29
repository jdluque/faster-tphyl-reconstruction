#cython: boundscheck=False, wraparound=False, nonecheck=False
# distutils: language = c++

import numpy as np
import cython

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp import cmath
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

def min_unweighted_vertex_cover_from_edgelist(edge_list: list):
    """For unweightred case, no need to use local ratio techniques used in
    networkx.algorithms.min_weighted_vertex_cover().
    """
    cdef unordered_set[int] cover

    cdef cnp.ndarray[DTYPE_t, ndim=1] ixs = np.random.permutation(len(edge_list))
    cdef int i, u, v
    for i in ixs:
        u, v = edge_list[i]
        if cover.find(u) == cover.end() or cover.find(v) == cover.end():
            continue
        cover.insert(u)
        cover.insert(v)

    if cover.size() % 2 == 0:
        return cover.size() / 2
    return cover.size() / 2 + 1


def vertex_cover_pp_from_edgelist(edge_list):
    """Returns
    1. a lower bound on the number of bit flips required to make A a
    perfect phylogeny by solving a related weighted vertex cover instance.
    2. a set of (i,j) indices of bits flipped.
    """
    vc = min_unweighted_vertex_cover_from_edgelist(edge_list)
    flipped_bits = len(vc)
    return np.ceil(flipped_bits / 2), vc

