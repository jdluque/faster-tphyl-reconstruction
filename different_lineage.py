import numpy as np


def diff_lineage_from_cols(A, p, q):
    """Check if columns p and q are in different-lineage relationships for
    matrix A.
    """
    col_p_A = A[:, p]
    col_q_A = A[:, q]
    z_A = col_p_A - col_q_A
    return np.all(np.isin([-1, 1], z_A))


def different_lineage(A: np.ndarray, B_gt: np.ndarray):
    assert A.shape == B_gt.shape, "Input matrices do not have equal dimensions."

    n, m = A.shape

    diff_lineage_pairs_in_gt = 0
    diff_lineage_pairs_in_computed_sol_and_gt = 0
    for p in range(m):
        for q in range(p + 1, m):
            diff_lineage_in_B = diff_lineage_from_cols(B_gt, p, q)
            if diff_lineage_in_B:
                diff_lineage_pairs_in_gt += 1
                diff_lineage_in_A = diff_lineage_from_cols(A, p, q)
                if diff_lineage_in_A:
                    diff_lineage_pairs_in_computed_sol_and_gt += 1

    # rv = diff_lineage_pairs_in_computed_sol_and_gt / diff_lineage_pairs_in_gt
    return diff_lineage_pairs_in_computed_sol_and_gt, diff_lineage_pairs_in_gt
    # assert rv <= 1
    # return rv


if __name__ == "__main__":
    # Read the data in and compute the different lineage metrics
    pass
