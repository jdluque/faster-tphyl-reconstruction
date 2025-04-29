import networkx as nx
import numpy as np
import copy
import pybnb
import scipy.sparse as sp

from utils import is_conflict_free_gusfield_and_get_two_columns_in_coflicts
from abstract import BoundingAlgAbstract

def simple_alg(x, mx_iter = 50):
    sol = x.copy()
    for ind in range(mx_iter):
        icf, col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(sol)
        # print(icf, col_pair)
        if icf:
            return True, sol
        col1 = sol[:, col_pair[0]]
        col2 = sol[:, col_pair[1]]
        rows01 = np.nonzero(np.logical_and(col1 == 0, col2 == 1))[0]
        # print(ind, len(rows01))
        sol[rows01, col_pair[0]] = 1
    return False, sol


class DynamicMWMBounding(BoundingAlgAbstract):
    def __init__(self, ascending_order=False, na_value=None, priority_version=-2, pass_info=False, make_ub=False):
        """
        :param ratio:
        :param ascending_order: if True the column pair with max weight is chosen in extra info
        """
        super().__init__()
        self.matrix = None
        self.G = None
        self._extra_info = {}
        self.ascending_order = ascending_order
        self.na_value = na_value
        self.priority_version = priority_version
        self.pass_info = pass_info
        self.make_ub = make_ub
        self.na_support = True if na_value is not None else False

        # NOTE: defined for compatibility with LinearProgrammingBounding in solve_by_BnB
        self.last_lp_feasible_delta = None

        # Debug variables
        self.num_lower_bounds = 0

    def get_name(self):
        params = [type(self).__name__,
                  self.ascending_order,
                  self.na_value,
                  self.priority_version,
                  self.pass_info,
                  self.make_ub
                  ]
        params_str = map(str, params)
        return "_".join(params_str)

    def reset(self, matrix):
        self.matrix = matrix
        self.G = nx.Graph()
        for p in range(self.matrix.shape[1]):
            for q in range(p + 1, self.matrix.shape[1]):
                self.calc_min0110_for_one_pair_of_columns(p, q, self.matrix)
        self._times = {"model_preparation_time": 0, "optimization_time": 0}

    def get_init_node(self):
        if not self.make_ub:
            return None
        node = pybnb.Node()
        icf, solution = simple_alg(self.matrix, mx_iter=500)

        nodedelta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 0))
        node_na_delta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == self.na_value))
        node.state = (nodedelta, icf, None, nodedelta.count_nonzero(), self.get_state(), node_na_delta)
        print("-----------", nodedelta.count_nonzero())
        node.queue_priority = 100000
        return node

    def get_extra_info(self):
        if self.pass_info:
            return self._extra_info
        else:
            return None

    def calc_min0110_for_one_pair_of_columns(self, p, q, current_matrix):
        found_one_one = False
        number_of_zero_one = 0
        number_of_one_zero = 0
        for r in range(current_matrix.shape[0]):
            if current_matrix[r, p] == 1 and current_matrix[r, q] == 1:
                found_one_one = True
            if current_matrix[r, p] == 0 and current_matrix[r, q] == 1:
                number_of_zero_one += 1
            if current_matrix[r, p] == 1 and current_matrix[r, q] == 0:
                number_of_one_zero += 1
        if self.G.has_edge(p, q):
            self.G.remove_edge(p, q)
        if found_one_one:
            self.G.add_edge(p, q, weight=min(number_of_zero_one, number_of_one_zero))

    def get_bound(self, delta, na_delta=None):

        self.num_lower_bounds += 1
        self._extraInfo = None

        current_matrix = self.matrix + delta
        old_g = copy.deepcopy(self.G)
        flips_mat = np.transpose(delta.nonzero())
        flipped_cols_set = set(flips_mat[:, 1])
        for q in flipped_cols_set:  # q is a changed column
            for p in range(self.matrix.shape[1]):
                if p < q:
                    self.calc_min0110_for_one_pair_of_columns(p, q, current_matrix)
                elif q < p:
                    self.calc_min0110_for_one_pair_of_columns(q, p, current_matrix)

        best_pairing = nx.max_weight_matching(self.G)

        sign = 1 if self.ascending_order else -1

        opt_pair_value = delta.shape[0] * delta.shape[1] * (-sign)  # either + inf or - inf
        opt_pair = None
        lb = 0
        for a, b in best_pairing:
            lb += self.G[a][b]["weight"]
            if self.G[a][b]["weight"] * sign > opt_pair_value * sign and self.G[a][b]["weight"] > 0:
                opt_pair_value = self.G[a][b]["weight"]
                opt_pair = (a, b)
        self.G = old_g
        self._extra_info = {"icf": (lb == 0), "one_pair_of_columns": opt_pair if lb > 0 else None}
        return lb + flips_mat.shape[0]

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