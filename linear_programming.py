import numpy as np
from ortools.linear_solver import pywraplp


def get_linear_program(
    matrix,
) -> tuple[pywraplp.Solver, pywraplp.Objective, dict]:
    solver = pywraplp.Solver.CreateSolver("GLOP")
    # WARN: Can use type np.bool only because there are no na values
    matrix = matrix.astype(np.bool)
    m, n = matrix.shape

    # Create variables
    vars = {}
    for i in range(m):
        for j in range(n):
            # Only matrix zeros should create variables
            if not matrix[i, j]:
                # NOTE: Using infinity _could_ lead to optimizations. It is unclear if it does in our particular case.
                vars[f"x_{i}_{j}"] = solver.NumVar(0, solver.infinity(), f"x_{i}_{j}")

    # Create constraints
    for p in range(n):
        for q in range(p + 1, n):
            col_p, col_q = matrix[:, p], matrix[:, q]
            has_one_one = np.any(np.logical_and(col_p, col_q))
            if not has_one_one:
                continue
            zero_ones = np.logical_and(~col_p, col_q)
            one_zeros = np.logical_and(col_p, ~col_q)
            for r1 in zero_ones.nonzero()[0]:
                for r2 in one_zeros.nonzero()[0]:
                    # For every 10 and 01 in conflict, at least one is (fractionally) flipped
                    solver.Add(vars[f"x_{r1}_{p}"] + vars[f"x_{r2}_{q}"] >= 1)
    # All created variables are correspond to zeros in the matrix
    objective = solver.Objective()
    for var in vars.values():
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()

    return solver, objective, vars
