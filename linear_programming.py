from itertools import combinations

import gurobipy as gp
import numpy as np
import ortools.linear_solver.python.model_builder as mb
from ortools.linear_solver.python.model_builder import ModelBuilder


def get_linear_program_from_col_subset_gurobi(
    matrix: np.ndarray,
    rounded_columns: set,
) -> tuple[gp.Model, gp.tupledict[tuple[int, int], gp.Var]]:
    # WARN: Can use type np.bool only because there are no na values
    matrix = matrix.astype(np.bool)
    m, n = matrix.shape

    model = gp.Model()
    # Create variables
    zero_coords = zip(*(~matrix).nonzero())
    x = model.addVars(zero_coords)

    # Create constraints
    for p in rounded_columns:
        for q in range(n):
            if p == q:
                continue

            col_p, col_q = matrix[:, p], matrix[:, q]

            has_one_one = np.any(np.logical_and(col_p, col_q))
            if not has_one_one:
                continue

            zero_ones = np.nonzero(np.logical_and(~col_p, col_q))[0]
            one_zeros = np.nonzero(np.logical_and(col_p, ~col_q))[0]

            # For every 10 and 01 in conflict, at least one is (fractionally) flipped
            for row1 in zero_ones:
                for row2 in one_zeros:
                    model.addLConstr(x[row1, p] + x[row2, q] >= 1)

    # All created variables correspond to zeros in the matrix
    model.setObjective(gp.quicksum(x), gp.GRB.MINIMIZE)

    return model, x


def get_linear_program_from_col_subset(
    matrix: np.ndarray,
    rounded_columns: set,
) -> tuple[mb.Model, dict[tuple[int, int], mb.Variable]]:
    # WARN: Can use type np.bool only because there are no na values
    matrix = matrix.astype(np.bool)
    m, n = matrix.shape

    model = mb.Model()
    # Create variables
    vars = {}
    for i, j in zip(*(~matrix).nonzero()):
        # NOTE: Using infinity _could_ lead to optimizations. It is unclear if it does in our particular case.
        vars[i, j] = model.new_var(0, 1, False, None)

    # Create constraints
    for p in rounded_columns:
        for q in range(p + 1, n):
            if p == q:
                continue
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


def get_linear_program_gurobi(
    matrix,
) -> tuple[gp.Model, gp.tupledict[tuple[int, int], gp.Var]]:
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


def add_constraints_for_newly_rounded_cols(
    model: gp.Model,
    vars: gp.tupledict[tuple[int, int], gp.Var],
    matrix: np.ndarray,
    rounded_columns: set[int],
) -> list[gp.Constr | gp.Var]:
    """When rounding to get an initial solution, it seems easier to rebuild the
    LP: just adding the new constraints results in too large of an LP (given
    that we should have removed lots of previous constraints and variables from
    the rounding).
    """
    new_items: list[gp.Var | gp.Constr] = []

    for p, q in combinations(rounded_columns, 2):
        col_p, col_q = matrix[:, p], matrix[:, q]

        has_one_one = np.any(np.logical_and(col_p, col_q))
        if not has_one_one:
            continue

        zero_ones = np.nonzero(np.logical_and(~col_p, col_q))[0]
        one_zeros = np.nonzero(np.logical_and(col_p, ~col_q))[0]

        for row1 in zero_ones:
            for row2 in one_zeros:
                # The zeros might not have been in a conflict before so we need a new variable
                if (row1, p) in vars:
                    x = vars[row1, p]
                else:
                    x = model.addVar()
                    vars[row1, p] = x
                    new_items.append(x)
                if (row2, q) in vars:
                    y = vars[row2, q]
                else:
                    y = model.addVar()
                    vars[row2, q] = y
                    new_items.append(y)

                # For every 10 and 01 in conflict, at least one is (fractionally) flipped
                new_items.append(model.addLConstr(x + y >= 1))

    return new_items


def get_linear_program(
    matrix,
) -> tuple[ModelBuilder, dict[tuple[int, int], mb.Variable]]:
    model = ModelBuilder()
    # WARN: Can use type np.bool only because there are no na values
    matrix = matrix.astype(np.bool)
    m, n = matrix.shape

    # Create variables
    vars = {}
    for i, j in zip(*(~matrix).nonzero()):
        # NOTE: Using infinity _could_ lead to optimizations. It is unclear if it does in our particular case.
        vars[i, j] = model.new_var(0, 1, False, None)

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
