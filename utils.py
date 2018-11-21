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


class DIIS:
    """
    A very plain implementation of Pulay's DIIS
    """

    def __init__(self, max_history):
        self._max_history = max_history
        self._xs = []
        self._rs = []

    def _prune(self):
        # Check precondition
        assert len(self._xs) == len(self._rs)
        if len(self._xs) > self._max_history:
            self._xs.pop(0)
            self._rs.pop(0)

    def _form_B(self):
        # Check precondition
        assert len(self._xs) == len(self._rs)
        history = len(self._xs)

        B = np.empty((history + 1, history + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for i, ri in enumerate(self._rs):
            for j, rj in enumerate(self._rs):
                if j > i: continue
                val = np.einsum('i,i', ri, rj)
                B[i, j] = val
                B[j, i] = val

        # Normalize B
        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        return B

    def _form_rhs(self):
        # Check precondition
        assert len(self._xs) == len(self._rs)
        history = len(self._xs)

        # Build rhs vector
        rhs = np.zeros(history + 1)
        rhs[-1] = -1

        return rhs

    def append(self, x, r):
        self._xs.append(x)
        self._rs.append(r)

    def extrapolate(self):
        self._prune()
        B = self._form_B()
        rhs = self._form_rhs()
        cs = LA.solve(B, rhs)

        # Form linear combination, excluding last element of coefficients
        return np.einsum('i,i...->...', cs[:-1], self._xs)
