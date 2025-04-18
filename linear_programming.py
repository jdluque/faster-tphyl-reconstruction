import numpy as np
import scipy.sparse as sp
from ortools.linear_solver.python.model_builder import ModelBuilder


def get_linear_program(
    matrix,
) -> tuple[ModelBuilder, dict]:
    model = ModelBuilder()
    # WARN: Can use type np.bool only because there are no na values
    matrix = matrix.astype(np.bool)
    m, n = matrix.shape

    # Create variables
    vars = {}
    for flat_ix, (i, j) in enumerate(zip(*(0 == matrix).nonzero())):
        # NOTE: Using infinity _could_ lead to optimizations. It is unclear if it does in our particular case.
        model.new_var(0, 1, False, f"x_{i}_{j}")
        vars[(i, j)] = flat_ix

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
                    left_zero = model.var_from_index(vars[r1, p])
                    right_zero = model.var_from_index(vars[r2, q])
                    model.add_linear_constraint(left_zero + right_zero, 1)

    # All created variables are correspond to zeros in the matrix
    model.minimize(sum(model.var_from_index(i) for i in range(len(vars))))

    return model, vars


def get_linear_program_from_delta(
    matrix: np.ndarray,
    delta: sp.lil_matrix,
    model: ModelBuilder,
    vars: dict,
) -> ModelBuilder:
    """Neutralize each LP variable corresponding to a flipped bit in delta by
    making its lower bound 1.
    """
    # Neutralize variables flipped thus far
    for i, j in zip(*delta.nonzero()):
        var_ix = vars[(i, j)]
        model.var_from_index(var_ix).lower_bound = 1
        # TODO: Remove this check before deploying
        assert model.var_from_index(var_ix).lower_bound == 1

    return model
