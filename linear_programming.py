from ortools.linear_solver import pywraplp


def get_linear_program(
    matrix,
) -> tuple[pywraplp.Solver, pywraplp.Objective, dict]:
    solver = pywraplp.Solver.CreateSolver("GLOP")

    m = matrix.shape[0]  # rows
    n = matrix.shape[1]  # cols

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
            # Figure out 10s and 01s and whether there is a 11
            zero_ones = []
            one_zeros = []
            has_one_one = False
            for i in range(m):
                x, y = matrix[i, p], matrix[i, q]
                if not x and y:
                    zero_ones.append(i)
                elif x and not y:
                    one_zeros.append(i)
                elif x and y:
                    has_one_one = True

            # For every 10 and 01 in conflict, at least one is (fractionally) flipped
            if has_one_one:
                for r1 in zero_ones:
                    for r2 in one_zeros:
                        solver.Add(vars[f"x_{r1}_{p}"] + vars[f"x_{r2}_{q}"] >= 1)

    # All created variables are correspond to zeros in the matrix
    objective = solver.Objective()
    for var in vars.values():
        objective.SetCoefficient(var, 1)
    objective.SetMinimization()

    return solver, objective, vars
