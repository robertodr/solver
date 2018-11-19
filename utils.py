import numpy as np
from numpy import linalg as LA


def quadratic_form(A, b, x):
    """
    Compute quadratic form associated to Ax=b.

    Warning
    -------
    The matrix A is assumed to be symmetric, positive-definite
    """
    # Compute quadratic form 1/2 xAx - xb
    return (0.5 * np.einsum('i,ij,j->', x, A, x) - np.einsum('i,i->', x, b))


def relative_error_to_reference(x, x_ref):
    return LA.norm(x - x_ref) / LA.norm(x_ref)


def compute_residual(A, b, x):
    return b - np.einsum('ij,j->i', A, x)


def compute_residual_norm(A, b, x):
    return LA.norm(compute_residual(A, b, x))
