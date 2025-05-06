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
import logging
import time

import numpy as np
import pybnb
import scipy.sparse as sp

from abstract import BoundingAlgAbstract
from linear_programming_bounding import LinearProgrammingBounding
from LPBoundGurobi import LinearProgrammingBoundingGurobi
from MWM_bounding import DynamicMWMBounding
from twosat import make_constraints_np_matrix, make_twosat_model_from_np, twosat_solver
from utils import (
    get_effective_matrix,
    is_conflict_free_gusfield_and_get_two_columns_in_coflicts,
)
from vertex_cover_bounding import VertexCoverBounding

logger = logging.getLogger(__name__)


def solve_by_BnB(matrix_in, na_value, which_bounding):
    """Use TwoSatBounding to run BnB algorithm."""
    bounding_algs = [
        TwoSatBounding(
            heuristic_setting=None,
            n_levels=2,
            compact_formulation=False,
            na_value=na_value,
        ),  # Real Data
        TwoSatBounding(
            heuristic_setting=[True, True, False, True, True],
            n_levels=1,
            compact_formulation=True,
            na_value=na_value,
        ),  # Simulation
        # TODO: Preserve the old algorithms orders so that the data remains
        # relevant. Create new bounding algorithms for the hybrid algorithms.
        LinearProgrammingBounding("GLOP", hybrid=False),
        LinearProgrammingBounding("PDLP", hybrid=False, branch_on_full_lp=False),
        LinearProgrammingBoundingGurobi(hybrid=False),
        LinearProgrammingBounding("PDLP", hybrid=False, branch_on_full_lp=True),
        VertexCoverBounding(5),
        DynamicMWMBounding(na_value=na_value),
        LinearProgrammingBoundingGurobi(
            hybrid=True,
            heuristic_setting=[True, True, False, True, True],
            n_levels=1,
            compact_formulation=True,
        ),
        LinearProgrammingBounding(
            "PDLP",
            hybrid=True,
            branch_on_full_lp=True,
            heuristic_setting=[True, True, False, True, True],
            n_levels=1,
            compact_formulation=True,
        ),
        LinearProgrammingBounding(
            "PDLP",
            hybrid=True,
            branch_on_full_lp=False,
            heuristic_setting=[True, True, False, True, True],
            n_levels=1,
            compact_formulation=True,
        ),
    ]
    result = bnb_solve(
        matrix_in, bounding_algorithm=bounding_algs[which_bounding], na_value=na_value
    )
    matrix_output = result[0]
    flips = []
    zero_one_flips = np.where((matrix_in != matrix_output) & (matrix_in != na_value))
    for i in range(len(zero_one_flips[0])):
        flips.append((zero_one_flips[0][i], zero_one_flips[1][i]))
    na_one_flips = np.where((matrix_output == 1) & (matrix_in == na_value))
    for i in range(len(na_one_flips[0])):
        flips.append((na_one_flips[0][i], na_one_flips[1][i]))

    bnb_instance = result[2]
    return flips, bnb_instance


class TwoSatBounding(BoundingAlgAbstract):
    def __init__(
        self,
        priority_version=-1,
        cluster_rows=False,
        cluster_cols=False,
        only_descendant_rows=False,
        na_value=None,
        heuristic_setting=None,
        n_levels=2,
        eps=0,
        compact_formulation=False,
    ):
        """
        :param priority_version:
        """
        assert not cluster_rows, "Not implemented yet"
        assert not cluster_cols, "Not implemented yet"
        assert not only_descendant_rows, "Not implemented yet"

        super().__init__()

        self.priority_version = priority_version

        self.na_support = True
        self.na_value = na_value
        self.matrix = None
        self._times = None
        self.next_lb = None
        self.heuristic_setting = heuristic_setting
        self.n_levels = n_levels
        self.eps = eps  # only for upperbound
        self.compact_formulation = compact_formulation
        self.cluster_rows = cluster_rows
        self.cluster_cols = cluster_cols
        self.only_descendant_rows = only_descendant_rows
        self.num_lower_bounds = 1

        # NOTE: defined for compatibility with LinearProgrammingBounding in solve_by_BnB
        self.last_lp_feasible_delta = None

    def get_name(self):
        params = [
            type(self).__name__,
            self.priority_version,
            self.heuristic_setting,
            self.n_levels,
            self.eps,
            self.compact_formulation,
        ]
        params_str = map(str, params)
        return "_".join(params_str)

    def reset(self, matrix):
        self.matrix = matrix  # todo: make the model here and do small alterations later

        # self.na_value = infer_na_value(matrix)
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    def get_init_node(self):
        # def twosat_solver(matrix, cluster_rows=False, cluster_cols=False, only_descendant_rows=False,
        #                   na_value=None, leave_nas_if_zero=False, return_lb=False, heuristic_setting=None,
        #                   n_levels=2, eps=0, compact_formulation=True):
        #     pass

        node = pybnb.Node()
        init_node_time = time.time()
        solution, model_time, opt_time, lb = twosat_solver(
            self.matrix,
            cluster_rows=self.cluster_rows,
            cluster_cols=self.cluster_cols,
            only_descendant_rows=self.only_descendant_rows,
            na_value=self.na_value,
            leave_nas_if_zero=True,
            return_lb=True,
            heuristic_setting=None,
            n_levels=self.n_levels,
            eps=self.eps,
            compact_formulation=self.compact_formulation,
        )
        self._times["model_preparation_time"] += model_time
        self._times["optimization_time"] += opt_time

        nodedelta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 0))
        init_node_time = time.time() - init_node_time
        # assert False, (
        #     f"next lower bound {lb} with found initial node in {init_node_time} with {nodedelta.count_nonzero()} flips"
        # )
        node_na_delta = sp.lil_matrix(
            np.logical_and(solution == 1, self.matrix == self.na_value)
        )
        logger.info("Time to compute init node: %s ", self._times)
        node.state = (
            nodedelta,
            True,
            None,
            nodedelta.count_nonzero(),
            self.get_state(),
            node_na_delta,
        )
        node.queue_priority = self.get_priority(
            till_here=-1, this_step=-1, after_here=-1, icf=True
        )
        self.next_lb = lb
        return node

    def get_bound(self, delta, delta_na=None):
        # make this dynamic when more nodes were getting explored
        if self.next_lb is not None:
            lb = self.next_lb
            self.next_lb = None
            return lb
        self._extraInfo = None
        current_matrix = get_effective_matrix(self.matrix, delta, delta_na)
        has_na = np.any(current_matrix == self.na_value)

        bound_time = time.time()
        model_time = time.time()
        return_value = make_constraints_np_matrix(
            current_matrix,
            n_levels=self.n_levels,
            na_value=self.na_value,
            compact_formulation=self.compact_formulation,
        )
        F, map_f2ij, zero_vars, na_vars, hard_constraints, col_pair = (
            return_value.F,
            return_value.map_f2ij,
            return_value.zero_vars,
            return_value.na_vars,
            return_value.hard_constraints,
            return_value.col_pair,
        )

        if col_pair is not None:
            icf = False
        elif return_value.complete_version:
            icf = True
        else:
            icf = None  # not sure
        rc2 = make_twosat_model_from_np(
            hard_constraints,
            F,
            zero_vars,
            na_vars,
            eps=0,
            heuristic_setting=self.heuristic_setting,
            compact_formulation=self.compact_formulation,
        )

        model_time = time.time() - model_time
        self._times["model_preparation_time"] += model_time

        opt_time = time.time()
        variables = rc2.compute()
        opt_time = time.time() - opt_time
        self._times["optimization_time"] += opt_time
        bound_time = time.time() - bound_time
        self.lb_time_total += bound_time
        self.lb_min_time = min(bound_time, self.lb_min_time)
        self.lb_max_time = max(bound_time, self.lb_max_time)

        result = 0
        for var_ind in range(len(variables)):
            if (
                variables[var_ind] > 0
                and abs(variables[var_ind]) in map_f2ij
                and self.matrix[map_f2ij[abs(variables[var_ind])]] == 0
            ):
                result += 1

        assert has_na or ((result == 0) == (col_pair is None)), f"{result}_{col_pair}"
        self._extraInfo = {
            "icf": icf,
            "one_pair_of_columns": col_pair,
        }
        ret = result + delta.count_nonzero()
        self.num_lower_bounds += 1
        # If we have a precomputed bound from get_init_node, use it
        if self.next_lb is not None:
            lb = self.next_lb
            self.next_lb = None
            return lb
        return ret

    def get_priority(self, till_here, this_step, after_here, icf=False):
        if icf:
            return self.matrix.shape[0] * self.matrix.shape[1] + 10
        else:
            sgn = np.sign(self.priority_version)
            pv_abs = self.priority_version * sgn
            if pv_abs == 1:
                return sgn * (till_here + this_step + after_here)
            elif pv_abs == 2:
                return sgn * (this_step + after_here)
            elif pv_abs == 3:
                return sgn * (after_here)
            elif pv_abs == 4:
                return sgn * (till_here + after_here)
            elif pv_abs == 5:
                return sgn * (till_here)
            elif pv_abs == 6:
                return sgn * (till_here + this_step)
            elif pv_abs == 7:
                return 0
        assert False, "get_priority did not return anything!"

    def notify_new_best_node(self, node, current):
        bound = node.state[3]
        logger.info("New best node with bound: %.2f", bound)


class BnB(pybnb.Problem):
    def __init__(self, I, boundingAlg: BoundingAlgAbstract, na_value=None):
        self.na_value = na_value
        self.has_na = np.any(I == self.na_value)
        self.I = I
        self.delta = sp.lil_matrix(I.shape, dtype=np.int8)  # this can be coo_matrix too
        self.boundingAlg = boundingAlg
        self.delta_na = None
        if self.has_na:
            assert boundingAlg.na_support, (
                "Input has N/A coordinates but bounding algorithm doesn't support it."
            )
            self.delta_na = sp.lil_matrix(
                I.shape, dtype=np.int8
            )  # the coordinates with na that are decided to be 1
        (
            self.icf,
            self.colPair,
        ) = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(self.I, na_value)
        self.boundingAlg.reset(I)
        self.node_to_add = self.boundingAlg.get_init_node()
        self.bound_value = self.boundingAlg.get_bound(self.delta)
        self.start_time = time.time()
        self.best_nodes = []  # list of (node objective, time explored)

    def sense(self):
        return pybnb.minimize

    def objective(self):
        if self.icf:
            return self.delta.count_nonzero()
        else:
            return pybnb.Problem.infeasible_objective(self)

    def bound(self):
        return self.bound_value

    def save_state(self, node):
        node.state = (
            self.delta,
            self.icf,
            self.colPair,
            self.bound_value,
            self.boundingAlg.get_state(),
            self.delta_na,
        )

    def load_state(self, node):
        (
            self.delta,
            self.icf,
            self.colPair,
            self.bound_value,
            boundingAlgState,
            self.delta_na,
        ) = node.state
        self.boundingAlg.set_state(boundingAlgState)

    def get_current_matrix(self):
        return get_effective_matrix(self.I, self.delta, self.delta_na)

    def branch(self):
        if self.icf:
            return

        need_for_new_nodes = True
        if self.node_to_add is not None:
            newnode = self.node_to_add
            self.node_to_add = None
            if (
                newnode.state[0].count_nonzero() == self.bound_value
            ):  # current_obj == lb => no need to explore
                need_for_new_nodes = False
            assert newnode.queue_priority is not None, (
                "Right before adding a node its priority in the queue is not set!"
            )
            yield newnode

        if need_for_new_nodes:
            p, q = self.colPair
            nf01 = None
            current_matrix = self.get_current_matrix()
            for col, colp in [(q, p), (p, q)]:
                node = pybnb.Node()
                nodedelta = copy.deepcopy(self.delta)
                node_na_delta = copy.deepcopy(self.delta_na)
                col1 = np.array(current_matrix[:, col], dtype=np.int8).reshape(-1)
                col2 = np.array(current_matrix[:, colp], dtype=np.int8).reshape(-1)
                rows01 = np.nonzero(np.logical_and(col1 == 0, col2 == 1))[0]
                rows21 = np.nonzero(np.logical_and(col1 == self.na_value, col2 == 1))[0]
                if (
                    len(rows01) + len(rows21) == 0
                ):  # nothing has changed! Dont add new node
                    continue
                nodedelta[rows01, col] = 1
                nf01 = nodedelta.count_nonzero()
                if self.has_na:
                    node_na_delta[rows21, col] = 1
                    new_bound = self.boundingAlg.get_bound(nodedelta, node_na_delta)
                    lp_feasible_delta = self.boundingAlg.last_lp_feasible_delta
                else:
                    new_bound = self.boundingAlg.get_bound(nodedelta)
                    lp_feasible_delta = self.boundingAlg.last_lp_feasible_delta

                # Upper bound based code (Add a new rounded node)
                if lp_feasible_delta is not None:
                    feasible_node = pybnb.Node()
                    nf01 = lp_feasible_delta.count_nonzero()

                    node_icf = True
                    node_col_pair = None
                    node_bound_value = nf01  # feasible node has known cost

                    feasible_node.state = (
                        lp_feasible_delta,
                        node_icf,
                        node_col_pair,
                        node_bound_value,
                        self.boundingAlg.get_state(),
                        self.delta_na,
                    )
                    feasible_node.queue_priority = self.boundingAlg.get_priority(
                        till_here=nf01,
                        this_step=0,
                        after_here=0,
                        icf=True,
                    )

                    yield feasible_node

                node_icf, nodecol_pair = None, None
                extra_info = self.boundingAlg.get_extra_info()

                if extra_info is not None:
                    if "icf" in extra_info:
                        node_icf = extra_info["icf"]
                    if "one_pair_of_columns" in extra_info:
                        nodecol_pair = extra_info["one_pair_of_columns"]
                if node_icf is None:
                    x = get_effective_matrix(self.I, nodedelta, node_na_delta)
                    node_icf, nodecol_pair = (
                        is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
                            x, self.na_value
                        )
                    )

                node_bound_value = max(self.bound_value, new_bound)
                node.state = (
                    nodedelta,
                    node_icf,
                    nodecol_pair,
                    node_bound_value,
                    self.boundingAlg.get_state(),
                    node_na_delta,
                )
                node.queue_priority = self.boundingAlg.get_priority(
                    till_here=nf01 - len(rows01),
                    this_step=len(rows01),
                    after_here=new_bound - nf01,
                    icf=node_icf,
                )
                assert node.queue_priority is not None, (
                    "Right before adding a node its priority in the queue is not set!"
                )
                yield node

    def notify_new_best_node(self, node, current):
        bound = node.state[3]
        cur_time = time.time() - self.start_time
        self.best_nodes.append((bound, cur_time))
        logger.info(
            "New best node with bound: %.2f found after %.2f s", bound, cur_time
        )


def bnb_solve(matrix, bounding_algorithm, na_value=None):
    problem1 = BnB(matrix, bounding_algorithm, na_value=na_value)
    solver = pybnb.Solver()
    results1 = solver.solve(problem1, queue_strategy="custom", log=None)
    if results1.solution_status != "unknown":
        returned_delta = results1.best_node.state[0]
        returned_delta_na = results1.best_node.state[-1]
        returned_matrix = get_effective_matrix(
            matrix, returned_delta, returned_delta_na, change_na_to_0=True
        )
    else:
        returned_matrix = np.zeros((1, 1))
    # print("results1.nodes:  ", results1.nodes)
    logger.info(
        "Number of lower bounds computed: %d", problem1.boundingAlg.num_lower_bounds
    )
    num_lbs = problem1.boundingAlg.num_lower_bounds
    avg_model_prep_time = (
        model_prep_time := problem1.boundingAlg._times["model_preparation_time"]
    ) / num_lbs
    avg_model_opt_time = (
        model_opt_time := problem1.boundingAlg._times["optimization_time"]
    ) / num_lbs
    print(f"{model_prep_time=:.5f} with {avg_model_prep_time=:.5f}")
    print(f"{model_opt_time=:.5f} with {avg_model_opt_time=:.5f}")

    return returned_matrix, results1.termination_condition, problem1
