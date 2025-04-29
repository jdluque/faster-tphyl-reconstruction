import time

import numpy as np
import pandas as pd
from vc_cython import (
    get_bounds,
    get_conflict_edgelist,
    vertex_cover_ub_greedy,
)

import process_together_vc
from vc import make_graph

if __name__ == "__main__":
    df = pd.read_csv("example/data2.SC", sep="\t", index_col=0)
    # df = pd.read_csv("real/melanoma20_clean.tsv", sep="\t", index_col=0)
    df.reset_index(drop=True, inplace=True)
    df = (df == 1).astype(np.bool)
    A = df.to_numpy(dtype=np.int64)
    start = time.time()
    print(f"{np.sum(A)=}")
    print(A.shape)
    best_lb = 0
    best_ub = float("inf")

    cython_graph_build_time = time.time()
    edge_list = get_conflict_edgelist(A)
    print(f"Cython Runtime: {time.time() - cython_graph_build_time:.5f} s")

    numpy_graph_build_time = time.time()
    edge_list = process_together_vc.get_conflict_edgelist(A.astype(np.bool))
    print(f"Python vectorized Runtime: {time.time() - numpy_graph_build_time:.5f} s")

    nx_graph_build_time = time.time()
    edge_list = make_graph(A.astype(np.bool))
    print(f"Networkx Runtime: {time.time() - nx_graph_build_time:.5f} s")

    # edge_list = list(edge_list)
    # get_vc_time = time.time()
    # lb = min_unweighted_vertex_cover_from_edgelist(edge_list)
    # print(f"Cython get vertex cover Runtime: {time.time() - get_vc_time:.5f} s")
    # print(f"{lb=}")

    get_vc_ub = time.time()
    ub = vertex_cover_ub_greedy(A)
    print(f"Python greedy vertex cover ub Runtime: {time.time() - get_vc_ub:.5f} s")
    print(f"{ub=}")

    get_vc_ub = time.time()
    lb, ub = get_bounds(A)
    print(f"Python get_bounds Runtime: {time.time() - get_vc_ub:.5f} s")
    print(f"{lb=} {ub=}")

    NUM_ITS = 8
    print(f"running {NUM_ITS} iterations")

    # lb, ub = get_bounds(A, NUM_ITS)
    # print(f"Best lb {lb}")
    # print(f"Best ub {ub}")
