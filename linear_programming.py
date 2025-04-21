import gurobipy as gp
import numpy as np
from ortools.linear_solver.python.model_builder import ModelBuilder, Variable


def get_linear_program_gurobi(matrix) -> tuple[gp.Model, gp.tupledict]:
    # WARN: Can use type np.bool only because there are no na values
    matrix = matrix.astype(np.bool)
    m, n = matrix.shape

    model = gp.Model()
    # Create variables
    zero_coords = zip(*(~matrix).nonzero())
    x = model.addVars(zero_coords)

    # Create constraints
    for p in range(n):
        for q in range(p + 1, n):
            col_p, col_q = matrix[:, p], matrix[:, q]
            has_one_one = np.any(np.logical_and(col_p, col_q))
            if not has_one_one:
                continue
            zero_ones = np.nonzero(np.logical_and(~col_p, col_q))[0]
            one_zeros = np.nonzero(np.logical_and(col_p, ~col_q))[0]
            for row1 in zero_ones:
                for row2 in one_zeros:
                    # For every 10 and 01 in conflict, at least one is (fractionally) flipped
                    model.addLConstr(x[row1, p] + x[row2, q] >= 1)
    # All created variables are correspond to zeros in the matrix
    model.setObjective(gp.quicksum(x), gp.GRB.MINIMIZE)

    return model, x


def get_linear_program(
    matrix,
) -> tuple[ModelBuilder, dict[tuple[int, int], Variable]]:
    model = ModelBuilder()
    # WARN: Can use type np.bool only because there are no na values
    matrix = matrix.astype(np.bool)
    m, n = matrix.shape

    # Create variables
    vars = {}
    for i, j in zip(*(~matrix).nonzero()):
        # NOTE: Using infinity _could_ lead to optimizations. It is unclear if it does in our particular case.
        vars[(i, j)] = model.new_var(0, 1, False, f"x_{i}_{j}")

    # Create constraints
    for p in range(n):
        for q in range(p + 1, n):
            col_p, col_q = matrix[:, p], matrix[:, q]
            # NOTE: np.any() does not short-circuit
            has_one_one = any(np.logical_and(col_p, col_q))
            if not has_one_one:
                continue
            zero_ones = np.nonzero(np.logical_and(~col_p, col_q))[0]
            one_zeros = np.nonzero(np.logical_and(col_p, ~col_q))[0]
            for row1 in zero_ones:
                for row2 in one_zeros:
                    # For every 10 and 01 in conflict, at least one is (fractionally) flipped
                    model.add_linear_constraint(vars[row1, p] + vars[row2, q], 1)

    # All created variables are correspond to zeros in the matrix
    model.minimize(sum(var for var in vars.values()))

    return model, vars
