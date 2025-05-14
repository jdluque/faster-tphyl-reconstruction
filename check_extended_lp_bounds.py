"""This is an example of where the lp relaxation of the phiscs (2019) lp achieved a greater lower bound than the initial conflicts only lp."""

import numpy as np
from ortools.linear_solver.python import model_builder

from linear_programming import get_extended_linear_program, get_linear_program

A = np.array(
    [
        # k1
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        # k2
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        # k3
        [1, 0, 1],
        # k4
        [0, 1, 1],
    ]
)
m1, vars1 = get_linear_program(A)
m2, vars2 = get_extended_linear_program(A)

for model in [m1, m2]:
    solver = model_builder.Solver("GLOP")
    solver.solve(model)
    print(f"{solver.objective_value=}")
