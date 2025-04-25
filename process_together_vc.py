"""This module contains helpers to compute a lower bound on the required number
of bit flips for a perfect phyolgeny by solving a related vertex cover instance.
"""

import itertools as it
import time

import networkx as nx
import numpy as np
import pandas as pd

from utils import is_conflict_free_gusfield_and_get_two_columns_in_coflicts


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


def vetex_cover_ub_greedy(A: np.ndarray, randomized=False):
    num_ones_og = A.sum()
    # Don't clobber the original matrix
    B = np.copy(A.astype(np.bool))
    n = B.shape[1]
    cols = np.random.permutation(n)
    for p, q in it.combinations(cols, 2):
        col_p = B[:, p]
        col_q = B[:, q]
        has_one_one = any(col_p & col_q)
        if not has_one_one:
            continue
        zero_ones = np.logical_and(~col_p, col_q)
        one_zeros = np.logical_and(col_p, ~col_q)

        # Resolve p, q conflict by flipping the cheaper of the two
        num_zero_ones = zero_ones.sum()
        num_one_zeros = one_zeros.sum()

        # print(f"{num_zero_ones=} {num_one_zeros=}")
        if not (num_one_zeros and num_zero_ones):
            continue

        # print(f"Before: {zero_ones.sum()} and {one_zeros.sum()}")
        if randomized:
            probability_flip_zero_ones = num_zero_ones / (num_zero_ones + num_one_zeros)
            flip_zero_ones = np.random.rand() < probability_flip_zero_ones
        else:
            flip_zero_ones = num_zero_ones < num_one_zeros

        if flip_zero_ones:
            # if num_zero_ones < num_one_zeros:
            B[zero_ones, p] = True
        else:
            B[one_zeros, q] = True

    is_cf, col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(B, 3)
    if is_cf:
        return B.sum() - num_ones_og
    else:
        print("not conflict free :(")
        return float("inf")


def vetex_cover_process_edges_together(A: np.ndarray):
    """NOTE: Better lower bounds with vertex_cover_pp and better upper bounds
    and vetex_cover_ub_greedy.
    """
    num_ones_og = A.sum()
    # Don't clobber the original matrix
    B = np.copy(A.astype(np.bool))
    n = B.shape[1]
    for p in range(n):
        for q in range(p + 1, n):
            cur_B_sum = B.sum()
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

            # Drop excess indeces. Only want to flip min(num 10s, num 01s) from
            # each column

            # print(f"Before: {zero_ones.sum()} and {one_zeros.sum()}")
            if num_zero_ones > num_one_zeros:
                ixs_to_drop = np.random.choice(
                    np.nonzero(zero_ones)[0],
                    size=num_zero_ones - num_one_zeros,
                    replace=False,
                )
                zero_ones[ixs_to_drop] = False
            elif num_zero_ones < num_one_zeros:
                ixs_to_drop = np.random.choice(
                    np.nonzero(one_zeros)[0],
                    size=num_one_zeros - num_zero_ones,
                    replace=False,
                )
                one_zeros[ixs_to_drop] = False

            assert zero_ones.sum() == one_zeros.sum()
            # print(f"Flipping {zero_ones.sum()} and {one_zeros.sum()}")
            B[zero_ones, p] = True
            B[one_zeros, q] = True

            assert cur_B_sum + 2 * zero_ones.sum() == B.sum()
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

    best_ub_greedy = float("inf")

    NUM_ITS = 200
    for i in range(NUM_ITS):
        lb, flipped_bits = vertex_cover_pp(G)
        ub = len(flipped_bits)
        best_lb = max(lb, best_lb)
        best_ub = min(ub, best_ub)

        other_num_flipped_bits = vetex_cover_process_edges_together(A)
        other_lb = np.ceil(other_num_flipped_bits / 2)
        best_lb_tog = max(other_lb, best_lb_tog)
        best_ub_tog = min(other_num_flipped_bits, best_ub_tog)

        greedy_ub = vetex_cover_ub_greedy(A)
        best_ub_greedy = min(greedy_ub, best_ub_greedy)

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

    print("\nUsing greedy...")
    print(
        f"It takes at most {best_ub_greedy} bit flips to turn A into a perfect phylogeny."
    )
