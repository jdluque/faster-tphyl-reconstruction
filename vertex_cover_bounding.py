from abstract import BoundingAlgAbstract


class VertexCoverBounding(BoundingAlgAbstract):
    def __init__(self, solver_name, priority_version=-1, na_value=None):
        """Initialize the Linear Programming Bounding algorithm.

        Args:
            priority_version: Controls node priority in the branch and bound tree
            na_value: Value representing missing data in the matrix
        """
        self.solver_name = solver_name  # LP solver
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
        self.num_lower_bounds = 0
        # This is the variable which holds an leaf node's number of flips
        # This gets populated when the vertex cover instance leads to a perfect
        # phylogeny and hence we can try to update the incumbent best node
        # TODO: Refactor this variable name
        self.last_lp_feasible_delta = None

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
                print(f"Lower bound {self.next_lb}")

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
            print("Rounded solution had conflicts")

            model_time_start = time.time()
            # Prepare model for another iteration
            model, vars = get_linear_program_from_col_subset(
                current_matrix, rounded_columns
            )
            self._times["model_preparation_time"] += time.time() - model_time_start

        init_node_time = time.time() - init_node_time
        # Create delta matrix (flips of 0→1)
        nodedelta = sp.lil_matrix(np.logical_and(current_matrix == 1, self.matrix == 0))

        # assert False, (
        #     f"Done finding conflict free matrix with {nodedelta.count_nonzero()} flips in {init_node_time} s"
        # )

        print("Completed init node: objective_value=", self.next_lb)
        print(f"{self._times=}")

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

    # @DeprecationWarning  # Don't use this
    # def get_initial_upper_bound(self, delta, max_rounds=10):
    #     """Helper method to compute the upper bound based on rounded LP
    #
    #     Args:
    #         delta: Sparse matrix with flipped entries
    #         max_rounds: maximum number of rounds allowed
    #
    #     Returns:
    #         Sparse delta matrix of added mutations
    #     """
    #     for attempt in range(max_rounds):  # FIX THIS SOLVE
    #         solver, current_matrix = self.linear_program.get_solver_and_matrix(
    #             delta, na_delta=None
    #         )
    #
    #         if solver.Solve() != pywraplp.Solver.OPTIMAL:
    #             print("Warning: LP did not solve to optimality on attempt", attempt)
    #             continue  # Try again
    #
    #         # Round LP solution
    #         rounded_matrix = np.copy(current_matrix)
    #         for (i, j), var_index in self.linear_program_vars.items():
    #             val = solver.Value(self.linear_program.var_from_index(var_index))
    #             rounded_matrix[i, j] = 1 if val >= 0.499 else 0
    #
    #         # Check if the rounded matrix is conflict free
    #         is_cf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
    #             rounded_matrix, self.na_value
    #         )
    #
    #         if is_cf:
    #             # Return the corresponding sparse delta matrix
    #             delta_matrix = sp.lil_matrix(
    #                 np.logical_and(rounded_matrix == 1, self.matrix == 0)
    #             )
    #             return delta_matrix
    #
    #     print("Warning: Failed to find conflict-free rounded matrix within max rounds.")
    #     return None

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
            print(
                "Linear Programming Bounding: The problem does not have an optimal solution."
            )
            return float("inf")  # Return infinity as a bound

        objective_value = solver.objective_value
        # Can clone the model -- or better yet -- set the lower bounds back to 0
        for i, j in zip(*delta.nonzero()):
            self.linear_program_vars[i, j].lower_bound = 0

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

        # Get upper bound (from rounded LP solution)

        # NOTE: This is the other option
        # feasible_delta = self.get_initial_upper_bound(delta, max_rounds=10)

        # Round LP solution
        # NOTE: use current_matrix to accumulate the rounded solution since we
        # do not need it anymore
        # TODO: Fix the indexing into LP variables
        for (i, j), variable in self.linear_program_vars.items():
            if solver.value(variable) >= 0.499:
                current_matrix[i, j] = 1

        # Check if the rounded matrix is conflict free
        is_cf, _ = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(
            current_matrix, self.na_value
        )

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
