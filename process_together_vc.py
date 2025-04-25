"""This module contains helpers to compute a lower bound on the required number
of bit flips for a perfect phyolgeny by solving a related vertex cover instance.
"""

import itertools as it
import time

import networkx as nx
import numpy as np
import pandas as pd


def make_graph(A):
    """Given a matrix A of mutations and cell samples, return a graph G where
    nodes are zeros of A, and edges occur between zeros that appear in a
    three-gametes rule violation together.
    """
    m, n = A.shape
    edge_list = []
    for p, q in it.combinations(range(n), 2):
        col_p, col_q = A[:, p], A[:, q]
        has_one_one = any(np.logical_and(col_p, col_q))
        if not has_one_one:
            continue
        zero_ones = np.nonzero(np.logical_and(~col_p, col_q))[0]
        one_zeros = np.nonzero(np.logical_and(col_p, ~col_q))[0]
        for row1 in zero_ones:
            for row2 in one_zeros:
                # For every 10 and 01 in conflict, at least one is (fractionally) flipped
                edge_list.append(((row1, p), (row2, q)))

    G = nx.Graph(edge_list)
    return G


def min_unweighted_vertex_cover(G: nx.Graph):
    """For unweightred case, no need to use local ratio techniques used in
    networkx.algorithms.min_weighted_vertex_cover().
    """
    cover = set()
    edges = np.random.permutation(list(G.edges()))
    for u, v in edges:
        u = tuple(u)
        v = tuple(v)
        if u in cover or v in cover:
            continue
        cover.add(u)
        cover.add(v)
    return cover


def vertex_cover_pp(G):
    """Returns
    1. a lower bound on the number of bit flips required to make A a
    perfect phylogeny by solving a related weighted vertex cover instance.
    2. a set of (i,j) indices of bits flipped.
    """
    vc = min_unweighted_vertex_cover(G)
    flipped_bits = len(vc)
    return np.ceil(flipped_bits / 2), vc


def vetex_cover_process_edges_together(A: np.ndarray, get_small_vc=True):
    num_ones_og = A.sum()
    # Don't clobber the original matrix
    B = np.copy(A.astype(np.bool))
    n = B.shape[1]
    for p in range(n):
        for q in range(p + 1, n):
            col_p = B[:, p]
            col_q = B[:, q]
            has_one_one = any(col_p & col_q)
            if not has_one_one:
                continue
            zero_ones = np.logical_and(~col_p, col_q)
            one_zeros = np.logical_and(col_p, ~col_q)
            # print(B[zero_ones, p].sum())
            # print(B[one_zeros, q].sum())

            # Resolve p, q conflict by flipping the cheaper of the two
            num_zero_ones = zero_ones.sum()
            num_one_zeros = one_zeros.sum()

            # print(f"{num_zero_ones=} {num_one_zeros=}")
            if not (num_one_zeros and num_zero_ones):
                continue

            flip_zero_ones = num_zero_ones < num_one_zeros
            if not get_small_vc:
                flip_zero_ones = not flip_zero_ones
            if flip_zero_ones:
                # assert B[zero_ones, p].sum() == 0
                # print("col before assignment", B[zero_ones, p])
                B[zero_ones, p] = True
                # print("col after assignment", B[zero_ones, p])
                # print(f"Flipped {one_zeros.sum()} entries")
            else:
                # assert B[one_zeros, q].sum() == 0
                # print("col before assignment", B[one_zeros, q])
                B[one_zeros, q] = True
                # print("col after assignment", B[one_zeros, q])
                # print(f"Flipped {zero_ones.sum()} entries")
            # print(B.sum(), " ", num_ones_og)
            assert B.sum() != num_ones_og

    return B.sum() - num_ones_og


if __name__ == "__main__":
    df = pd.read_csv("example/data2.SC", sep="\t", index_col=0)
    # df = pd.read_csv("real/melanoma20_clean.tsv", sep="\t", index_col=0)
    df.reset_index(drop=True, inplace=True)
    df = (df == 1).astype(np.bool)
    A = df.to_numpy(dtype=np.bool)
    start = time.time()
    G = make_graph(A)
    graph_build_time = time.time() - start
    print(f"{np.sum(A)=}")
    print(A.shape)
    best_lb = 0
    best_ub = float("inf")

    best_lb_tog = 0
    best_ub_tog = float("inf")
    NUM_ITS = 50
    for i in range(NUM_ITS):
        lb, flipped_bits = vertex_cover_pp(G)
        ub = len(flipped_bits)
        best_lb = max(lb, best_lb)
        best_ub = min(ub, best_ub)

        get_small_vc = i < NUM_ITS // 2
        other_num_flipped_bits = vetex_cover_process_edges_together(A, get_small_vc)
        other_lb = np.ceil(other_num_flipped_bits / 2)
        best_lb_tog = max(other_lb, best_lb_tog)
        best_ub_tog = min(other_num_flipped_bits, best_ub_tog)

    end = time.time()
    print(f"Runtime: {end - start:.3f} s")
    print(f"It takes at least {best_lb} bit flips to turn A into a perfect phylogeny.")
    print(f"It takes at most {best_ub} bit flips to turn A into a perfect phylogeny.")

    print("\nBy flipping together...")
    print(
        f"It takes at least {best_lb_tog} bit flips to turn A into a perfect phylogeny."
    )
    print(
        f"It takes at most {best_ub_tog} bit flips to turn A into a perfect phylogeny."
    )
