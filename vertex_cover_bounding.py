import copy
import logging
import time

import numpy as np
import pybnb
import scipy.sparse as sp
from vc_cython import get_bounds, vertex_cover_init

from abstract import BoundingAlgAbstract
from twosat import twosat_solver
from utils import get_effective_matrix

logger = logging.getLogger(__name__)


class VertexCoverBounding(BoundingAlgAbstract):
    def __init__(
        self,
        hybrid,
        num_iterations=1,
        priority_version=-1,
        na_value=None,
        # TwoSatBoudning params for hybrid algorithm
        cluster_rows=False,
        cluster_cols=False,
        only_descendant_rows=False,
        heuristic_setting=None,
        n_levels=2,
        eps=0,
        compact_formulation=False,
    ):
        """Initialize the Linear Programming Bounding algorithm.

        Args:
            priority_version: Controls node priority in the branch and bound tree
            na_value: Value representing missing data in the matrix
        """
        super().__init__()

        self.matrix = None  # Input Matrix
        self._extra_info = None  # Additional information from bounding
        self._extraInfo = {}  # For compatibility with the abstract class
        self._times = {}  # Store timing information
        self.na_support = False  # Not supporting NA values yet
        self.na_value = na_value
        self.next_lb = None  # Store precomputed lower bound from get_init_node
        self.priority_version = priority_version  # Controls node priority calculation
        self.model_state = None  # State to store/restore

        # Debug variables
        # This is the variable which holds an leaf node's number of flips
        # This gets populated when the vertex cover instance leads to a perfect
        # phylogeny and hence we can try to update the incumbent best node
        # TODO: Refactor this variable name
        self.last_lp_feasible_delta = None

        self.hybrid = hybrid

        self.num_iterations = num_iterations

        # TwoSatBounding params; to be used by the hybrid algorithm
        self.heuristic_setting = heuristic_setting
        self.n_levels = n_levels
        self.eps = eps  # only for upperbound
        self.compact_formulation = compact_formulation
        self.cluster_rows = cluster_rows
        self.cluster_cols = cluster_cols
        self.only_descendant_rows = only_descendant_rows
        self.num_lower_bounds = 1

    def get_name(self):
        """Return a string identifier for this bounding algorithm."""
        params = [
            type(self).__name__,
            self.priority_version,
        ]
        params_str = map(str, params)
        return "_".join(params_str)

    def reset(self, matrix):
        """Reset the bounding algorithm with a new matrix.

        Args:
            matrix: The input matrix for the problem
        """
        assert self.na_value is None, "N/A is not implemented yet"
        self.matrix = matrix
        self._times = {"model_preparation_time": 0, "optimization_time": 0}
        self.model_state = None

    def get_init_node(self):
        logger.info("Sampling %d vertex covers", self.num_iterations)
        if self.hybrid:
            return self.twosat_based_get_init_node()
        else:
            return self.vc_based_get_init_node()

    def twosat_based_get_init_node(self):
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
        node_na_delta = sp.lil_matrix(
            np.logical_and(solution == 1, self.matrix == self.na_value)
        )
        logger.info("Time to compute init node: %s ", self._times)
        initial_best_node_objective = nodedelta.count_nonzero()
        node.state = (
            nodedelta,
            True,
            None,
            initial_best_node_objective,
            self.get_state(),
            node_na_delta,
        )
        node.queue_priority = self.get_priority(
            till_here=-1, this_step=-1, after_here=-1, icf=True
        )
        self.next_lb = lb

        return node

    def vc_based_get_init_node(self):
        """Create and return an initial node with a solution from the LP relaxation.

        Returns:
            A pybnb.Node object with initial solution
        """
        node = pybnb.Node()

        # Matrix to become conflict free
        current_matrix = np.copy(self.matrix)

        init_node_time = time.time()
        model_time_start = time.time()

        flips = vertex_cover_init(current_matrix)

        init_node_time = time.time() - init_node_time
        # Create delta matrix (flips of 0→1)
        # I don't need this because vertex_cover_init already flips the matrix entries
        # TODO: Make the current_matrix flips more efficient
        nodedelta = sp.lil_matrix(np.logical_and(current_matrix == 1, self.matrix == 0))

        # Set node state
        node.state = (
            nodedelta,  # Delta matrix (flips)
            True,  # Is conflict-free flag
            None,  # Column pair (None if conflict-free)
            nodedelta.count_nonzero(),  # Current objective value
            self.get_state(),  # Algorithm state
            None,  # NA delta (not implemented)
        )

        logger.info(
            "Computed best node in get_init_node() with %d flips", node.state[3]
        )

        # Set priority for this node
        node.queue_priority = self.get_priority(
            till_here=-1, this_step=-1, after_here=-1, icf=True
        )

        return node

    def get_bound(self, delta, na_delta=None):
        """Calculate a lower bound on the number of flips needed.

        Args:
            delta: Sparse matrix with flipped entries
            na_delta: NA entries to be flipped (not implemented)

        Returns:
            Lower bound value
        """
        self.num_lower_bounds += 1
        # If we have a precomputed bound from get_init_node, use it
        if self.next_lb is not None:
            lb = self.next_lb
            self.next_lb = None
            self.last_lp_feasible_delta = None
            return lb

        # TODO: Switch to a smaller dtype; will need to propagate this change to vc_cython.pyx
        effective_matrix = get_effective_matrix(self.matrix, delta, na_delta).astype(
            np.int64
        )

        vc_time = time.time()
        lb, ub, flips = get_bounds(effective_matrix)
        vc_time = time.time() - vc_time
        self.lb_time_total += vc_time
        self.lb_min_time = min(vc_time, self.lb_min_time)
        self.lb_max_time = max(vc_time, self.lb_max_time)

        if ub >= 0:
            # TODO: Optimize this assignment
            for i, j in flips:
                effective_matrix[i, j] = 1
            self.last_lp_feasible_delta = sp.lil_matrix(
                np.logical_and(effective_matrix == 1, self.matrix == 0),
            )
        else:
            self.last_lp_feasible_delta = None

        rv = lb + delta.count_nonzero()
        return rv

    def get_state(self):
        """Get the current state of the bounding algorithm.

        Returns:
            State dictionary or None
        """
        return self.model_state

    def set_state(self, state):
        """Restore the state of the bounding algorithm.

        Args:
            state: State dictionary or None
        """
        self.model_state = state

    def get_extra_info(self):
        """Get extra information from the bounding algorithm.

        Returns:
            Dictionary with extra information
        """
        return copy.copy(self._extraInfo)

    def get_priority(self, till_here, this_step, after_here, icf=False):
        """Calculate the priority of a node for the branch and bound queue.

        Args:
            till_here: Flips made so far
            this_step: Flips made in this step
            after_here: Estimated flips still needed
            icf: Is conflict-free flag

        Returns:
            Priority value (higher values are processed first)
        """
        if icf:
            # If conflict-free, give very high priority
            return self.matrix.shape[0] * self.matrix.shape[1] + 10
        else:
            # Otherwise, use priority_version to determine strategy
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
            else:
                # Default: prioritize by estimated flips needed
                return -after_here

    def get_times(self):
        """Get timing information.

        Returns:
            Dictionary with timing information
        """
        return self._times
