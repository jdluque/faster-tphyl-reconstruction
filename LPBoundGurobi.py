import copy
import time

import gurobipy as gp
import numpy as np
import pybnb
import scipy.sparse as sp
from gurobipy import GRB

from abstract import BoundingAlgAbstract
from linear_programming import get_linear_program_gurobi
from utils import (
    get_effective_matrix,
    is_conflict_free_gusfield_and_get_two_columns_in_coflicts,
)


class LinearProgrammingBoundingGurobi(BoundingAlgAbstract):
    def __init__(self, priority_version=-1, na_value=None):
        """Initialize the Linear Programming Bounding algorithm.

        Args:
            priority_version: Controls node priority in the branch and bound tree
            na_value: Value representing missing data in the matrix
        """
        self.matrix = None  # Input Matrix
        # Linear Program solver and variables
        self.linear_program = gp.Model()
        self.linear_program_vars = None
        self._extra_info = None  # Additional information from bounding
        self._extraInfo = {}  # For compatibility with the abstract class
        self._times = {}  # Store timing information
        self.na_support = False  # Not supporting NA values yet
        self.na_value = na_value
        self.next_lb = None  # Store precomputed lower bound from get_init_node
        self.priority_version = priority_version  # Controls node priority calculation
        self.model_state = None  # State to store/restore

        # Debug variables
        self.num_lower_bounds = 0

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
        node = pybnb.Node()

        # Start timing model preparation
        model_time_start = time.time()
        model, vars = get_linear_program_gurobi(self.matrix)
        self.linear_program = model
        self.linear_program_vars = vars

        # Model getting
        # self.linear_program.Params.LogFile = "gurobi.log"
        # self.linear_program.Params.LogToConsole = False
        self.linear_program.Params.OutputFlag = 0

        # Record model preparation time
        model_time = time.time() - model_time_start
        self._times["model_preparation_time"] += model_time

        # Solve and time optimization
        opt_time_start = time.time()
        model.optimize()
        opt_time = time.time() - opt_time_start
        self._times["optimization_time"] += opt_time

        if model.Status != gp.GRB.OPTIMAL:
            # If no optimal solution, return None
            return None

        # Round solution to get a binary matrix
        solution = np.copy(self.matrix)
        # TODO: Will need to add a check here to ensure the matrix is conflict free. Else re-run the LP and round the new solution.
        for i, j in vars:
            # NOTE: Some solvers may give 0.5 - epsilon
            if vars[i, j].X >= 0.499:
                solution[i, j] = 1

        # Check if the solution is conflict-free
        icf, col_pair = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
            solution, self.na_value
        )
        assert icf, "Initial node must be conflict free"

        # Create delta matrix (flips of 0→1)
        nodedelta = sp.lil_matrix(np.logical_and(solution == 1, self.matrix == 0))

        # Store the LP objective value for future bound calculations
        self.next_lb = np.ceil(model.getObjective().getValue())
        print("In init node: objective_value=", self.next_lb)

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

    def compute_lp_bound(self, delta, na_delta=None):
        """Helper method to compute LP bound for a given delta.

        Args:
            delta: Sparse matrix with flipped entries
            na_delta: NA entries to be flipped (not implemented)

        Returns:
            Lower bound value
        """
        # Create effective matrix
        current_matrix = get_effective_matrix(self.matrix, delta, na_delta)

        # Start timing model preparation
        model_time_start = time.time()

        # Instead of getting a brand new linear_program, recycle the initial one
        for i, j in zip(*delta.nonzero()):
            self.linear_program.setAttr("LB", self.linear_program_vars[i, j], 1)

        # Record model preparation time
        model_time = time.time() - model_time_start
        self._times["model_preparation_time"] += model_time

        # Solve and time optimization
        opt_time_start = time.time()

        self.linear_program.optimize()
        # solver = model_builder.Solver(self.solver_name)
        if self.linear_program.Status != GRB.OPTIMAL:
            print(
                "Linear Programming Bounding: The problem does not have an optimal solution."
            )
            return float("inf")

        opt_time = time.time() - opt_time_start
        self._times["optimization_time"] += opt_time

        # Can clone the model -- or better yet -- set the lower bounds back to 0
        for i, j in zip(*delta.nonzero()):
            self.linear_program.setAttr("LB", self.linear_program_vars[i, j], 0)

        # Save extra info for branching decisions (TODO: Is this needed)
        is_conflict_free, conflict_col_pair = (
            is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
                current_matrix, self.na_value
            )
        )
        self._extraInfo = {
            "icf": is_conflict_free,
            "one_pair_of_columns": conflict_col_pair,
        }

        # Return the bound (LP objective includes existing flips)
        return np.ceil(self.linear_program.getObjective().getValue())

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
            return lb

        # Otherwise compute the bound using LP
        return self.compute_lp_bound(delta, na_delta)

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
