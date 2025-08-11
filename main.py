#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, Erfan Sadeqi Azer and Farid Rashidi Mehrabadi All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# =========================================================================================
# Written by : Erfan Sadeqi Azer (esadeqia@iu.edu)
#              and Farid Rashidi Mehrabadi (frashidi@iu.edu)
# Last Update: Jan 20, 2020
# =========================================================================================

import copy
import csv
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd

from bnb import solve_by_BnB
from utils import is_conflict_free_gusfield_and_get_two_columns_in_coflicts


def now():
    return f"[{datetime.now().strftime('%m/%d %H:%M:%S')}]"


def printf(s):
    print(f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] ", end="")
    print(s, flush=True)


def is_conflict_free(D):
    conflict_free = True
    for p in range(D.shape[1]):
        for q in range(p + 1, D.shape[1]):
            oneone = False
            zeroone = False
            onezero = False
            for r in range(D.shape[0]):
                if D[r, p] == 1 and D[r, q] == 1:
                    oneone = True
                if D[r, p] == 0 and D[r, q] == 1:
                    zeroone = True
                if D[r, p] == 1 and D[r, q] == 0:
                    onezero = True
            if oneone and zeroone and onezero:
                conflict_free = False
                return conflict_free
    return conflict_free


def count_flips(I, O, na_value):
    flips_0_1 = 0
    flips_1_0 = 0
    flips_na_0 = 0
    flips_na_1 = 0
    n, m = I.shape
    for i in range(n):
        for j in range(m):
            if I[i, j] == 0 and O[i, j] == 1:
                flips_0_1 += 1
            elif I[i, j] == 1 and O[i, j] == 0:
                flips_1_0 += 1
            elif I[i, j] == na_value and O[i, j] == 0:
                flips_na_0 += 1
            elif I[i, j] == na_value and O[i, j] == 1:
                flips_na_1 += 1
    return flips_0_1, flips_1_0, flips_na_0, flips_na_1


def infer_na_value(x):
    vals = set(np.unique(x))
    all_vals = copy.copy(vals)
    vals.remove(0)
    vals.remove(1)
    if len(vals) > 0:
        assert len(vals) == 1, (
            "Unable to infer na: There are more than three values:" + repr(all_vals)
        )
        return vals.pop()
    return None


def draw_tree(filename):
    add_cells = True

    from collections import Counter

    import networkx as nx
    from networkx.drawing.nx_agraph import to_agraph

    def contains(col1, col2):
        for i in range(len(col1)):
            if not col1[i] >= col2[i]:
                return False
        return True

    df = pd.read_csv(filename, sep="\t", index_col=0)
    splitter_mut = "\n"
    matrix = df.values
    names_mut = list(df.columns)

    i = 0
    while i < matrix.shape[1]:
        j = i + 1
        while j < matrix.shape[1]:
            if np.array_equal(matrix[:, i], matrix[:, j]):
                matrix = np.delete(matrix, j, 1)
                x = names_mut.pop(j)
                names_mut[i] += splitter_mut + x
                j -= 1
            j += 1
        i += 1

    rows = matrix.shape[0]
    cols = matrix.shape[1]
    dimensions = np.sum(matrix, axis=0)
    indices = np.argsort(dimensions)
    dimensions = np.sort(dimensions)
    names_mut = [names_mut[indices[i]] for i in range(cols)]

    G = nx.DiGraph(dpi=300)
    G.add_node(cols)
    G.add_node(cols - 1)
    G.add_edge(cols, cols - 1, label=names_mut[cols - 1])
    node_mud = {}
    node_mud[names_mut[cols - 1]] = cols - 1

    i = cols - 2
    while i >= 0:
        if dimensions[i] == 0:
            break
        attached = False
        for j in range(i + 1, cols):
            if contains(matrix[:, indices[j]], matrix[:, indices[i]]):
                G.add_node(i)
                G.add_edge(node_mud[names_mut[j]], i, label=names_mut[i])
                node_mud[names_mut[i]] = i
                attached = True
                break
        if not attached:
            G.add_node(i)
            G.add_edge(cols, i, label=names_mut[i])
            node_mud[names_mut[i]] = i
        i -= 1

    clusters = {}
    for node in G:
        if node == cols:
            # G._node[node]['label'] = '<<b>germ<br/>cells</b>>'
            G._node[node]["fontname"] = "Helvetica"
            G._node[node]["width"] = 0.4
            G._node[node]["style"] = "filled"
            G._node[node]["penwidth"] = 3
            G._node[node]["fillcolor"] = "gray60"
            continue
        untilnow_mut = []
        sp = nx.shortest_path(G, cols, node)
        for i in range(len(sp) - 1):
            untilnow_mut += G.get_edge_data(sp[i], sp[i + 1])["label"].split(
                splitter_mut
            )
        untilnow_cell = df.loc[
            (df[untilnow_mut] == 1).all(axis=1)
            & (df[[x for x in df.columns if x not in untilnow_mut]] == 0).all(axis=1)
        ].index
        if len(untilnow_cell) > 0:
            clusters[node] = f"<b>{', '.join(untilnow_cell)}</b>"
        else:
            clusters[node] = "––"

        if add_cells:
            if "––" not in clusters[node]:
                G._node[node]["fillcolor"] = "#80C4DF"
            else:
                G._node[node]["fillcolor"] = "gray90"
            G._node[node]["label"] = clusters[node]
        else:
            G._node[node]["label"] = ""
            G._node[node]["shape"] = "circle"
        G._node[node]["fontname"] = "Helvetica"
        G._node[node]["width"] = 0.4
        G._node[node]["style"] = "filled"
        G._node[node]["penwidth"] = 2.5
    i = 1
    for k, v in clusters.items():
        if v == "––":
            clusters[k] = i * "––"
            i += 1

    for node in G:
        if node != cols:
            num = 0
            paths = nx.shortest_path(G, source=cols, target=node)
            for i in range(len(paths) - 1):
                x = paths[i]
                y = paths[i + 1]
                num += len(G[x][y]["label"].split(splitter_mut))
            G._node[node]["label"] = (
                f"<[{node}]  " + G._node[node]["label"] + f"  ({num})>"
            )
        else:
            G._node[node]["label"] = f"<[{node}]  germ cells>"

    data = G.edges.data("label")
    outputpath = filename[: -len(".CFMatrix")]
    for u, v, l in data:
        ll = l.split(splitter_mut)
        genes = [x.split(".")[0] for x in ll]
        a = Counter(genes)
        a = a.most_common()
        lll = list(set([x.split(".")[0] for x in ll]))
        G.add_edge(u, v, label=splitter_mut.join(lll))
        print(
            f"[{u}]->[{v}]: {' '.join(ll)}", file=open(f"{outputpath}.mutsAtEdges", "a")
        )
        G.add_edge(u, v, label=f" {len(ll)}")

    header = ""
    temp = df.columns[(df == 0).all(axis=0)]
    if len(temp) > 0:
        header += f"Became Germline: {len(temp)}<br/>"

    H = nx.relabel_nodes(G, clusters)
    html = """<{}>""".format(header)
    H.graph["graph"] = {
        "label": html,
        "labelloc": "t",
        "resolution": 300,
        "fontname": "Helvetica",
        "fontsize": 8,
    }
    H.graph["node"] = {"fontname": "Helvetica", "fontsize": 12}
    H.graph["edge"] = {"fontname": "Helvetica", "fontsize": 12, "penwidth": 2}

    mygraph = to_agraph(H)
    mygraph.layout(prog="dot")
    mygraph.draw(f"{outputpath}.png")


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="A Fast Branch and Bound Algorithm for the Perfect Tumor Phylogeny Reconstruction Problem"
    )
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        dest="i",
        type=str,
        default=None,
        required=True,
        help="Path to single-cell data matrix file",
    )
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "-o",
        "--output",
        dest="o",
        type=str,
        default=".",
        required=False,
        help="Output directory [default: %(default)s]",
    )
    optional.add_argument(
        "-b",
        "--bounding",
        dest="b",
        type=int,
        default=1,
        required=False,
        help="Bounding algorithm (1, 12) [default: %(default)s]",
    )
    optional.add_argument(
        "-t",
        "--drawTree",
        action="store_true",
        dest="t",
        required=False,
        help="Draw output tree with Graphviz [required Graphviz to be installed on the system]",
    )
    optional.add_argument(
        "-l",
        "--log",
        type=str,
        default="experiments.log",
        required=False,
        help="Path for log file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        format="%(asctime)s : %(name)-8s : %(levelname)-6s : %(message)s",
        level=logging.INFO,
    )

    df_input = pd.read_csv(args.i, delimiter="\t", index_col=0)
    df_input = df_input.replace("?", 3)
    df_input = df_input.astype(int)
    matrix_input = df_input.values
    na_value = infer_na_value(matrix_input)
    logger.info("using args %s", args)

    # NOTE: should match bounding_algs in bnb.solve_by_BnB
    alg_name = [
        "TwoSat full",
        "TwoSat compact",
        "LP GLOP full rewrite",
        "LP PDLP partial rewrite",
        "LP Gurobi partial rewrite",
        "LP PDLP full rewrite",
        "Vertex Cover",
        "Maximum Weight Matching",
        "Hybrid LP Gurobi partial rewrite",
        "Hybrid LP PDLP full rewrite",
        "Hybrid LP PDLP partial rewrite",
        "Hybrid Vertex Cover",
        "Hybrid Extended LP PDLP",
    ][args.b - 1]
    logger.info("Bounding algorithm: %s", alg_name)

    printf(f"Size: {matrix_input.shape}")
    printf(f"NAValue: {na_value}")
    printf(f"#Zeros: {len(np.where(matrix_input == 0)[0])}")
    printf(f"#Ones: {len(np.where(matrix_input == 1)[0])}")
    printf(f"#NAs: {len(np.where(matrix_input == na_value)[0])}")
    start_time = time.time()

    matrix_output = matrix_input.copy()
    flips, bnb_instance = solve_by_BnB(matrix_input, na_value, args.b - 1)
    for k in flips:
        matrix_output[k] = 1
    matrix_output[np.where(matrix_output == na_value)] = 0

    end_time = time.time()
    printf(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    flips_0_1, flips_1_0, flips_na_0, flips_na_1 = count_flips(
        matrix_input, matrix_output, na_value
    )
    cf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
        matrix_output, None
    )
    printf(f"#0->1: {flips_0_1}")
    printf(f"#1->0: {flips_1_0}")
    printf(f"#na->0: {flips_na_0}")
    printf(f"#na->1: {flips_na_1}")
    printf(f"isDone: {cf}")

    if args.o and cf:
        df_output = pd.DataFrame(matrix_output)
        df_output.columns = df_input.columns
        df_output.index = df_input.index
        df_output.index.name = "cellIDxmutID"
        filename = os.path.splitext(os.path.basename(args.i))[0]
        if not os.path.exists(args.o):
            os.makedirs(args.o)
        file = os.path.join(args.o, filename)
        df_output.to_csv(f"{file}.CFMatrix", sep="\t")

        # TODO: Set write_csvs as arg flag?
        write_csvs = True
        if write_csvs:
            if not os.path.exists("experiments"):
                os.makedirs("experiments")
            best_nodes_csv_path = "experiments/best_nodes.csv"
            bounds_csv_path = "experiments/bounds.csv"
            times_csv_path = "experiments/times.csv"
            logger.info(
                "Writing to csvs %s, %s, and %s",
                best_nodes_csv_path,
                bounds_csv_path,
                times_csv_path,
            )

            with open(best_nodes_csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)

                if not os.path.exists(best_nodes_csv_path) or not os.path.getsize(
                    best_nodes_csv_path
                ):
                    writer.writerow(
                        ["algorithm", "dataset", "objective", "seconds since start"]
                    )
                for obj, explored_time in bnb_instance.best_nodes:
                    writer.writerow([alg_name, filename, obj, round(explored_time, 3)])

            with open(bounds_csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not os.path.exists(bounds_csv_path) or not os.path.getsize(
                    bounds_csv_path
                ):
                    writer.writerow(
                        ["algorithm", "dataset", "count", "min", "max", "mean"]
                    )
                writer.writerow(
                    [
                        alg_name,
                        filename,
                        round(bnb_instance.boundingAlg.num_lower_bounds, 3),
                        round(bnb_instance.boundingAlg.lb_min_time, 3),
                        round(bnb_instance.boundingAlg.lb_max_time, 3),
                        round(
                            bnb_instance.boundingAlg.lb_time_total
                            / bnb_instance.boundingAlg.num_lower_bounds,
                            3,
                        ),
                    ]
                )

            with open(times_csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not os.path.exists(times_csv_path) or not os.path.getsize(
                    times_csv_path
                ):
                    writer.writerow(["algorithm", "dataset", "seconds", "finished"])
                writer.writerow(
                    [alg_name, filename, round(end_time - start_time, 3), cf]
                )

    if args.o and args.t and cf:
        draw_tree(f"{file}.CFMatrix")
        printf(f"The output phylogenetic tree is in '{args.o}' directory!")
