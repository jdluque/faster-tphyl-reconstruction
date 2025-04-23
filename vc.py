"""This module contains helpers to compute a lower bound on the required number
of bit flips for a perfect phyolgeny by solving a related vertex cover instance.
"""

import itertools as it
from collections import defaultdict

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
        pq_pair_counts = defaultdict(list)
        # Count the number of 01s, 10s, and 11s
        for i in range(m):
            if A[i, p] and not A[i, q]:
                pq_pair_counts[1, 0].append(i)
            elif not A[i, p] and A[i, q]:
                pq_pair_counts[0, 1].append(i)
            elif A[i, p] and A[i, q]:
                pq_pair_counts[1, 1].append(i)
        # Tally up the number of flips required to fix (p,q) violations
        count11 = len(pq_pair_counts[1, 1])
        count01 = len(pq_pair_counts[0, 1])
        count10 = len(pq_pair_counts[1, 0])
        # Add an edge between the zeros of each 01 and each 10 in conflict with one another
        if count11 and count01 and count10:
            for i1 in pq_pair_counts[0, 1]:
                for i2 in pq_pair_counts[1, 0]:
                    edge_list.append(((i1, p), (i2, q)))
    G = nx.Graph(edge_list)
    m, n = A.shape
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
    assert len(cover) >= 40, cover
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


if __name__ == "__main__":
    df = pd.read_csv("example/data2.SC", sep="\t", index_col=0)
    df.reset_index(drop=True, inplace=True)
    df = (df == 1).astype(np.bool)
    A = df.to_numpy(dtype=np.bool)
    G = make_graph(A)
    print(f"{np.sum(A)=}")
    print(A.shape)
    best_lb = 0
    best_ub = float("inf")
    for _ in range(5):
        lb, flipped_bits = vertex_cover_pp(G)
        # print(f"{lb=}")
        ub = len(flipped_bits)
        best_lb = max(lb, best_lb)
        best_ub = min(ub, best_ub)
    print(f"It takes at least {best_lb} bit flips to turn A into a perfect phylogeny.")
    print(f"It takes at most {best_ub} bit flips to turn A into a perfect phylogeny.")
