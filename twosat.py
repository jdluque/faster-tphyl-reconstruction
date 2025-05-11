import itertools
import time

import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

rec_num = 0


def twosat_solver(
    matrix,
    cluster_rows=False,
    cluster_cols=False,
    only_descendant_rows=False,
    na_value=None,
    leave_nas_if_zero=False,
    return_lb=False,
    heuristic_setting=None,
    n_levels=2,
    eps=0,
    compact_formulation=False,
):
    global rec_num
    rec_num += 1
    assert not cluster_rows, "Not implemented yet"
    assert not cluster_cols, "Not implemented yet"
    assert not only_descendant_rows, "Not implemented yet"
    model_time = 0
    opt_time = 0
    start_time = time.time()

    return_value = make_constraints_np_matrix(
        matrix,
        n_levels=n_levels,
        na_value=na_value,
        compact_formulation=compact_formulation,
    )
    model_time += time.time() - start_time
    F, map_f2ij, zero_vars, na_vars, hard_constraints, col_pair = (
        return_value.F,
        return_value.map_f2ij,
        return_value.zero_vars,
        return_value.na_vars,
        return_value.hard_constraints,
        return_value.col_pair,
    )

    if col_pair is not None:
        icf = False
    elif return_value.complete_version:
        icf = True
    else:
        icf = None

    final_output = None
    lower_bound = 0
    if icf:
        final_output, total_time = matrix.copy(), 0
    else:
        start_time = time.time()
        rc2 = make_twosat_model_from_np(
            hard_constraints,
            F,
            zero_vars,
            na_vars,
            eps,
            heuristic_setting,
            compact_formulation=compact_formulation,
        )
        model_time += time.time() - start_time

        a = time.time()
        variables = rc2.compute()
        b = time.time()
        opt_time += b - a
        output_matrix = matrix.copy()
        output_matrix = output_matrix.astype(np.int8)

        for var_ind in range(len(variables)):
            if (
                0 < variables[var_ind] and variables[var_ind] in map_f2ij
            ):  # if 0 or 2 make it one
                output_matrix[map_f2ij[variables[var_ind]]] = 1
                if matrix[map_f2ij[variables[var_ind]]] != na_value:
                    lower_bound += 1
        # I don't change 2s to 0s here keep them 2 for next time

        # For recursion I set off all sparsification parameters
        # Also I want na->0 to stay na for the recursion regardless of original input for leave_nas_if_zero
        # I am also not passing eps here to wrap up the recursion soon

        Orec, rec_model_time, rec_opt_time = twosat_solver(
            output_matrix,
            na_value=na_value,
            heuristic_setting=None,
            n_levels=n_levels,
            leave_nas_if_zero=True,
            compact_formulation=compact_formulation,
        )
        model_time += rec_model_time
        opt_time += rec_opt_time

        if not leave_nas_if_zero:
            Orec[Orec == na_value] = 0
        final_output = Orec

    if return_lb:
        return final_output, model_time, opt_time, lower_bound
    else:
        return final_output, model_time, opt_time


def make_constraints_np_matrix(
    matrix,
    constraints=None,
    n_levels=2,
    na_value=None,
    row_coloring=None,
    col_coloring=None,
    probability_threshold=None,
    fn_rate=None,
    column_intersection=None,
    compact_formulation=True,
):
    """
    Returns a "C x 2 x 2" matrix where C is the number of extracted constraints each constraints is of the form:
    ((r1, c1), (r2, c2)) and correspond to Z_{r1, c1} or Z{r2, c2}
    :param matrix: A binary matrix cellsXmutations
    :param constraints: If not None instead of evaluating the whole matrix it will only look at potential constraints
    :param level: The type of constraints to add
    :param na_value:
    :param row_coloring: Only constraints that has the same row coloring will be used
    :param col_coloring: Only constraints that has the same column coloring will be used
    :param probability_threshold:
    :param fn_rate:
    :return:
    """
    # todo: Take decendence analysis out of here?
    # todo: how to reuse constraints input
    from collections import namedtuple

    assert (probability_threshold is None) == (fn_rate is None)
    descendance_analysis = probability_threshold is not None
    assert 1 <= n_levels <= 2, "not implemented yet"

    # means none of scarification ideas have been used
    complete_version = all_None(
        row_coloring, col_coloring, probability_threshold, fn_rate
    )

    soft_cnst_num = 0
    hard_constraints = [[] for _ in range(n_levels)]  # an empty list each level
    if descendance_analysis:
        # dictionary for lazy calculation of decadence:
        descendent_dict = dict()

    # variables for each zero
    F = -np.ones(matrix.shape, dtype=np.int64)
    num_var_F = 0
    map_f2ij = dict()
    zero_vars = list()
    na_vars = list()
    if compact_formulation:
        B_vars_offset = matrix.shape[0] * matrix.shape[1] + 1
        num_var_B = 0
        map_b2ij = dict()
        if n_levels >= 2:
            C_vars_offset = B_vars_offset + matrix.shape[1] * matrix.shape[1] + 1
            num_var_C = 0
            map_c2ij = dict()

    col_pair = None
    pair_cost = 0

    if column_intersection is None:
        column_intersection = calculate_column_intersections(matrix, row_by_row=True)
        # column_intersection = calculate_column_intersections(matrix, for_loop=True)
    for p in range(matrix.shape[1]):
        for q in range(p + 1, matrix.shape[1]):
            if column_intersection[p, q]:  # p and q has intersection
                # todo: check col_coloring here
                r01 = np.nonzero(
                    np.logical_and(
                        zero_or_na(matrix[:, p], na_value=na_value), matrix[:, q] == 1
                    )
                )[0]
                r10 = np.nonzero(
                    np.logical_and(
                        matrix[:, p] == 1, zero_or_na(matrix[:, q], na_value=na_value)
                    )
                )[0]
                cost = min(len(r01), len(r10))
                if cost > pair_cost:  # keep best pair to return as auxiliary info
                    # print("------------", cost, (p, q), len(r01), len(r10), column_intersection[p, q])
                    col_pair = (p, q)
                    pair_cost = cost
                if cost > 0:  # don't do anything if one of r01 or r10 is empty
                    if (
                        not compact_formulation
                    ):  # len(r01) * len(r10) many constraints will be added
                        for a, b in itertools.product(r01, r10):
                            # todo: check row_coloring
                            for row, col in [
                                (a, p),
                                (b, q),
                            ]:  # make sure the variables for this are made
                                var_list = (
                                    zero_vars if matrix[row, col] == 0 else na_vars
                                )
                                num_var_F = make_sure_variable_exists(
                                    F, row, col, num_var_F, map_f2ij, var_list, na_value
                                )
                            hard_constraints[0].append(
                                [[a, p], [b, q]]
                            )  # at least one of them should be flipped
                    else:  # compact formulation: (r01 + r10) number of new constraints will be added
                        # define new B variable
                        b_pq = B_vars_offset + num_var_B
                        num_var_B += 1
                        for row_list, col, sign in zip((r01, r10), (p, q), (1, -1)):
                            for row in row_list:
                                var_list = (
                                    zero_vars if matrix[row, col] == 0 else na_vars
                                )
                                num_var_F = make_sure_variable_exists(
                                    F, row, col, num_var_F, map_f2ij, var_list, na_value
                                )
                                hard_constraints[0].append([row, col, b_pq, sign])
                                # this will be translated to (Z_ap or (sign)B_pq)
            elif n_levels >= 2:
                r01 = np.nonzero(
                    np.logical_and(
                        zero_or_na(matrix[:, p], na_value=na_value), matrix[:, q] == 1
                    )
                )[0]
                r10 = np.nonzero(
                    np.logical_and(
                        matrix[:, p] == 1, zero_or_na(matrix[:, q], na_value=na_value)
                    )
                )[0]
                cost = min(len(r01), len(r10))
                if cost > 0:  # don't do anything if one of r01 or r10 is empty
                    if not compact_formulation:
                        # len(r01) * len(r10) * (len(r01) * len(r10)) many constraints will be added
                        x = np.empty((r01.shape[0] + r10.shape[0], 2), dtype=np.int64)
                        x[: len(r01), 0] = r01
                        x[: len(r01), 1] = p
                        x[-len(r10) :, 0] = r10
                        x[-len(r10) :, 1] = q

                        for a, b, ind in itertools.product(r01, r10, range(x.shape[0])):
                            for row, col in [
                                (a, p),
                                (b, q),
                                (x[ind, 0], x[ind, 1]),
                            ]:  # make sure the variables for this are made
                                # print(row, col)
                                var_list = (
                                    zero_vars if matrix[row, col] == 0 else na_vars
                                )
                                num_var_F = make_sure_variable_exists(
                                    F, row, col, num_var_F, map_f2ij, var_list, na_value
                                )
                            row = [[a, p], [b, q], [x[ind, 0], x[ind, 1]]]
                            if not np.array_equal(
                                row[0], row[2]
                            ) and not np.array_equal(row[1], row[2]):
                                hard_constraints[1].append(
                                    [[a, p], [b, q], [x[ind, 0], x[ind, 1]]]
                                )
                    else:  #  if compact_formulation: 2(r01 + r10) will be added
                        # define two new C variable
                        c_pq0 = C_vars_offset + num_var_C
                        num_var_C += 1
                        c_pq1 = C_vars_offset + num_var_C
                        num_var_C += 1
                        for row_list, col, sign in zip((r01, r10), (p, q), (1, -1)):
                            for row in row_list:
                                var_list = (
                                    zero_vars if matrix[row, col] == 0 else na_vars
                                )
                                num_var_F = make_sure_variable_exists(
                                    F, row, col, num_var_F, map_f2ij, var_list, na_value
                                )
                                if sign == 1:
                                    hard_constraints[1].append([row, col, c_pq0, c_pq1])
                                    # this will be translated to (~Z_ap or ~c_pq0 or ~c_pq1)
                                    # and (Z_ap or c_pq0)
                                else:
                                    hard_constraints[1].append([row, col, c_pq1, c_pq0])
                                    # this will be translated to (~Z_ap or ~c_pq0 or ~c_pq1) (the same)
                                    # and (Z_ap or c_pq1) (different)

    # todo: when using this make sure to put an if to say if the model is small and
    return_type = namedtuple(
        "ReturnType",
        "F map_f2ij zero_vars na_vars hard_constraints col_pair complete_version",
    )
    for ind in range(n_levels):
        hard_constraints[ind] = np.array(hard_constraints[ind], dtype=np.int64)
    return return_type(
        F, map_f2ij, zero_vars, na_vars, hard_constraints, col_pair, complete_version
    )


def make_twosat_model_from_np(
    constraints,
    F,
    zero_vars,
    na_vars,
    eps=None,
    heuristic_setting=None,
    compact_formulation=True,
):
    if eps is None:
        eps = 1 / (len(zero_vars) + len(na_vars))

    if heuristic_setting is None:
        rc2 = RC2(WCNF())
    else:
        assert len(heuristic_setting) == 5
        rc2 = RC2(
            WCNF(),
            adapt=heuristic_setting[0],
            exhaust=heuristic_setting[1],
            incr=heuristic_setting[2],
            minz=heuristic_setting[3],
            trim=heuristic_setting[4],
        )

    if not compact_formulation:
        # hard constraints Z_a,p or Z_b,q
        for constr_ind in range(constraints[0].shape[0]):
            constraint = constraints[0][constr_ind]
            a, p, b, q = constraint.flat
            # print(constraint, F.shape)
            # print(a, p, b, q)
            rc2.add_clause([F[a, p], F[b, q]])
        if len(constraints) >= 2:
            # hard constraints Z_a,p or Z_b,q or -Z_c,d
            for constr_ind in range(constraints[1].shape[0]):
                constraint = constraints[1][constr_ind]
                a, p, b, q, c, d = constraint.flat
                # print(a, p, b, q, c, d)
                rc2.add_clause([F[a, p], F[b, q], -F[c, d]])
    else:
        # hard constraints Z_a,p or (sign) b_pq
        for constr_ind in range(constraints[0].shape[0]):
            constraint = constraints[0][constr_ind]
            row, col, b_pq, sign = constraint.flat
            rc2.add_clause([F[row, col], sign * b_pq])
        if len(constraints) >= 2:
            # hard constraints Z_a,p or Z_b,q or -Z_c,d
            for constr_ind in range(constraints[1].shape[0]):
                constraint = constraints[1][constr_ind]
                row, col, c_pq0, c_pq1 = constraint.flat
                # if Z_rc is True at least one of p, q should become active
                # E.g., c_pq0 be False
                rc2.add_clause([-F[row, col], -c_pq0, -c_pq1])
                # if c_pq0 is False then Z_rc has to be flipped
                rc2.add_clause([F[row, col], c_pq0])

    # soft constraints for zero variables
    for var in zero_vars:
        rc2.add_clause([-var], weight=1)

    if eps > 0:
        # soft constraints for zero variables
        for var in na_vars:
            rc2.add_clause([-var], weight=eps)

    return rc2


def all_None(*args):
    return args.count(None) == len(args)


def calculate_column_intersections(matrix, for_loop=False, row_by_row=False):
    ret = np.empty((matrix.shape[1], matrix.shape[1]), dtype=np.bool)
    mask_1 = matrix == 1

    if for_loop:
        for p in range(matrix.shape[1]):
            # even though the diagonals are not necessary, I keep it for ease of debugging
            for q in range(p, matrix.shape[1]):
                ret[p, q] = np.any(np.logical_and(mask_1[:, p], mask_1[:, q]))
                ret[q, p] = ret[p, q]
    elif row_by_row:
        ret[:, :] = 0
        for r in range(matrix.shape[0]):
            one_columns = mask_1[r]
            ret[np.ix_(one_columns, one_columns)] = True
    return ret


def zero_or_na(vec, na_value=-1):
    return np.logical_or(vec == 0, vec == na_value)


def make_sure_variable_exists(
    memory_matrix, row, col, num_var_F, map_f2ij, var_list, na_value
):
    if memory_matrix[row, col] < 0:
        num_var_F += 1
        map_f2ij[num_var_F] = (row, col)
        memory_matrix[row, col] = num_var_F
        var_list.append(num_var_F)
    return num_var_F
