import time

import numpy as np
import pandas as pd
from vc_cython import (
    get_bounds,
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

    numpy_graph_build_time = time.time()
    edge_list = process_together_vc.get_conflict_edgelist(A.astype(np.bool))
    print(
        f"Python vectorized build edgelist Runtime: {time.time() - numpy_graph_build_time:.5f} s"
    )

    nx_graph_build_time = time.time()
    edge_list = make_graph(A.astype(np.bool))
    print(f"Networkx build edgelist Runtime: {time.time() - nx_graph_build_time:.5f} s")

    NUM_ITS = 1
    print(f"running {NUM_ITS} iterations")

    get_vc_ub = time.time()
    lb, ub, flips = get_bounds(A, iterations=NUM_ITS)
    print(f"Python get_bounds Runtime: {time.time() - get_vc_ub:.5f} s")
    print(f"{lb=} {ub=}")
