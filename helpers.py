import numpy as np

# Utility function to check if two columns have conflicts
def is_conflict(X, p, q):
    """Check if a columns p and q of X have conflicts."""
    col_p = X[:, p]
    col_q = X[:, q]
    return np.any(col_p & col_q) and np.any(~col_p & col_q) and np.any(col_p & ~col_q)