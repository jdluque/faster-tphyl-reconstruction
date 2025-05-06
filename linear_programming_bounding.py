import copy
import logging
import time

import numpy as np
import pybnb
import scipy.sparse as sp
from ortools.linear_solver.python import model_builder

from abstract import BoundingAlgAbstract
from linear_programming import get_linear_program, get_linear_program_from_col_subset
from twosat import twosat_solver
from utils import (
    get_effective_matrix,
    is_conflict_free_gusfield_and_get_two_columns_in_coflicts,
)

logger = logging.getLogger(__name__)


class LinearProgrammingBounding(BoundingAlgAbstract):
    def __init__(
        self,
        solver_name,
        hybrid,
        branch_on_full_lp=True,
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

        self.solver_name = solver_name  # LP solver
        self.matrix = None  # Input Matrix
        # Linear Program solver and variables
        self.linear_program = None
        self.linear_program_vars = None
        self._extra_info = None  # Additional information from bounding
        self._extraInfo = {}  # For compatibility with the abstract class
        self._times = {}  # Store timing information
        self.na_support = False  # Not supporting NA values yet
        self.na_value = na_value
        self.next_lb = None  # Store precomputed lower bound from get_init_node
        self.priority_version = priority_version  # Controls node priority calculation
        self.model_state = None  # State to store/restore
        self.branch_on_full_lp = branch_on_full_lp
        # Used to compute incumbent "upper bounds" after rounding LP solutions
        self.last_lp_feasible_delta = None

        # Whether to use the max weight 2-sat bounding algorithm on the initial node
        self.hybrid = hybrid

        # Debug variables
        self.num_lower_bounds = 0

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
            # TODO: Add other params?
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
        """Create and return an initial node with a solution from the LP relaxation.

        Returns:
            A pybnb.Node object with initial solution
        """
        if self.hybrid:
            return self.twosat_based_get_init_node()
        else:
            return self.lp_based_get_init_node()

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

    def lp_based_get_init_node(self):
        node = pybnb.Node()

        # Matrix to become conflict free
        current_matrix = np.copy(self.matrix)

        init_node_time = time.time()
        model_time_start = time.time()

        self.linear_program, self.linear_program_vars = get_linear_program(
            current_matrix
        )

        model_time = time.time() - model_time_start
        self._times["model_preparation_time"] += model_time

        model, vars = self.linear_program, self.linear_program_vars
        while True:
            # Start timing model preparation
            solver = model_builder.Solver(self.solver_name)

            # Solve and time optimization
            opt_time_start = time.time()
            status = solver.solve(model)
            opt_time = time.time() - opt_time_start
            self._times["optimization_time"] += opt_time

            if status != model_builder.SolveStatus.OPTIMAL:
                # If no optimal solution, return None
                return None

            # Store the LP objective value for future bound calculations
            if self.next_lb is None:
                self.next_lb = np.ceil(solver.objective_value)
                logger.info("In get_init_node(): lower bound: %.2f", self.next_lb)

            # Round solution to get a binary matrix
            rounded_columns = set()
            for i, j in vars:
                # NOTE: Some solvers may give 0.5 - epsilon
                if solver.value(vars[i, j]) >= 0.499:
                    if not current_matrix[i, j]:
                        rounded_columns.add(j)
                    current_matrix[i, j] = 1

            # Check if the solution is conflict-free
            icf, col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
                current_matrix, self.na_value
            )

            if icf:
                break
            if not (rounded_columns or icf):
                logger.error(
                    "No columns were rounded but the matrix is not conflict free. The algorithm is stuck. This can happen due to numerical issues with the LP solver (%s). Consider using a different LP solver or reducing the error tolerance.",
                    self.solver_name,
                )
                raise Exception(
                    "No columns were rounded but the matrix is not conflict free. The algorithm is stuck. This can happen due to numerical issues with the LP solver."
                )
            logger.info(
                "Rounded solution had conflicts -- resolving LP and re-rounding"
            )

            model_time_start = time.time()
            # Prepare model for another iteration
            model, vars = get_linear_program_from_col_subset(
                current_matrix, rounded_columns
            )
            self._times["model_preparation_time"] += time.time() - model_time_start

        init_node_time = time.time() - init_node_time
        # Create delta matrix (flips of 0→1)
        nodedelta = sp.lil_matrix(np.logical_and(current_matrix == 1, self.matrix == 0))

        logger.info("Completed init node: objective_value= %.2f", self.next_lb)
        logger.info(f"{self._times=}")

        # Set node state
        node.state = (
            nodedelta,  # Delta matrix (flips)
            icf,  # Is conflict-free flag
            col_pair,  # Column pair (None if conflict-free)
            nodedelta.count_nonzero(),  # Current objective value
            self.get_state(),  # Algorithm state
            None,  # NA delta (not implemented)
        )

        # Set priority for this node
        node.queue_priority = self.get_priority(
            till_here=-1, this_step=-1, after_here=-1, icf=icf
        )

        return node

    def compute_lp_bound(self, branch_on_full_lp, delta, na_delta=None):
        """Helper method to compute LP bound for a given delta.

        Args:
            delta: Sparse matrix with flipped entries
            na_delta: NA entries to be flipped (not implemented)
            full_lp: bool
                Whether to use the linear program with constraints for all
                conflicts present in the matrix after flipping delta, or
                whether to re-use the original LP, which contains only initial
                conflicts.

        Returns:
            Lower bound value
        """
        # Create effective matrix
        current_matrix = get_effective_matrix(self.matrix, delta, na_delta)

        # Save extra info for branching decisions
        is_conflict_free, conflict_col_pair = (
            is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
                current_matrix, self.na_value
            )
        )
        self._extraInfo = {
            "icf": is_conflict_free,
            "one_pair_of_columns": conflict_col_pair,
        }

        bound_time = time.time()
        # If the current matrix is already conflict free, there is no need to
        # do anything else
        if is_conflict_free:
            self.last_lp_feasible_delta = None
            return np.ceil(objective_value)

        # Start timing model preparation
        model_time_start = time.time()

        if branch_on_full_lp:
            self.linear_program, self.linear_program_vars = get_linear_program(
                current_matrix
            )
        else:
            # Instead of getting a brand new linear_program, recycle the initial one
            for i, j in zip(*np.nonzero(delta)):
                if (i, j) in self.linear_program_vars:
                    self.linear_program_vars[i, j].lower_bound = 1

        # Record model preparation time
        model_time = time.time() - model_time_start
        self._times["model_preparation_time"] += model_time

        # Solve and time optimization
        opt_time_start = time.time()

        solver = model_builder.Solver(self.solver_name)
        status = solver.solve(self.linear_program)

        opt_time = time.time() - opt_time_start
        self._times["optimization_time"] += opt_time

        if status != model_builder.SolveStatus.OPTIMAL:
            logger.error(
                "The LP does not have an optimal solution. This should not be possible."
            )
            return float("inf")  # Return infinity as a bound

        if branch_on_full_lp:
            objective_value = solver.objective_value + delta.count_nonzero()
        else:
            objective_value = solver.objective_value
            # Can clone the model -- or better yet -- set the lower bounds back to 0
            for i, j in zip(*np.nonzero(delta)):
                if (i, j) in self.linear_program_vars:
                    self.linear_program_vars[i, j].lower_bound = 0

        # Get upper bound (from rounded LP solution)
        # NOTE: This is the other option
        # feasible_delta = self.get_initial_upper_bound(delta, max_rounds=10)

        # Round LP solution, accumulating in the no-longer needed current_matrix
        for (i, j), variable in self.linear_program_vars.items():
            if solver.value(variable) >= 0.499:
                current_matrix[i, j] = 1

        # Check if the rounded matrix is conflict free
        is_cf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
            current_matrix, self.na_value
        )

        bound_time = time.time() - bound_time
        self.lb_time_total += bound_time
        self.lb_min_time = min(bound_time, self.lb_min_time)
        self.lb_max_time = max(bound_time, self.lb_max_time)

        # Create delta matrix (flips of 0→1)
        if not is_cf:  # has conflicts
            rounded_delta_matrix = None
        else:
            rounded_delta_matrix = sp.lil_matrix(
                np.logical_and(current_matrix == 1, self.matrix == 0),
            )

        # Update with rounded matrix
        self.last_lp_feasible_delta = rounded_delta_matrix

        # Return the bound (LP objective includes existing flips)
        return np.ceil(objective_value)

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

        # Otherwise compute the bound using LP
        return self.compute_lp_bound(
            branch_on_full_lp=self.branch_on_full_lp, delta=delta, na_delta=na_delta
        )

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

    def get_priority(
        self, till_here, this_step, after_here, icf=False
    ):  # TODO: Do I need this
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
