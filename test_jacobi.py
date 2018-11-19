from functools import partial
from operator import ge, le
from typing import Callable, Dict, List, Tuple

import numpy as np
import zarr
from fixpoint import *
from numpy import linalg as LA
from utils import *


def jacobi(A, b, rtol=1.0e-8, etol=1.0e-8, max_it=25, x_0=None):
    if x_0 is None:
        x = np.zeros_like(b)
    else:
        x = x_0

    it = 0
    # Compute pseudoenergy
    E_old = 0.0
    E_new = quadratic_form(A, b, x)
    DeltaE = E_new - E_old
    rnorm = compute_residual_norm(A, b, x)
    print('Iteration #    Residual norm         Delta E')
    # Report at start
    print('    {:4d}        {:.5E}       {:.5E}'.format(
        it, rnorm, abs(DeltaE)))
    while it < max_it:
        # Update solution vector
        x = jacobi_step(A, b, x)
        # Compute residual
        rnorm = compute_residual_norm(A, b, x)
        # Compute new pseudoenergy
        E_new = quadratic_form(A, b, x)
        DeltaE = E_new - E_old
        E_old = E_new

        it += 1

        # Check convergence
        if rnorm < rtol and abs(DeltaE) < etol:
            # Print last statistics before breaking out
            print('    {:4d}        {:.5E}       {:.5E}'.format(
                it, rnorm, abs(DeltaE)))
            break

        # Report
        print('    {:4d}        {:.5E}       {:.5E}'.format(
            it, rnorm, abs(DeltaE)))
    else:
        raise RuntimeError(
            'Maximum number of iterations ({0:d}) exceeded, but residual norm {1:.5E} still greater than threshold {2:.5E}'.
            format(max_it, rnorm, rtol))
    return x


def jacobi_step(A, b, x):
    D = np.diag(A)
    O = A - np.diagflat(D)
    return (b - np.einsum('ij,j->i', O, x)) / D


def stepper(A, b, iterate: Iterate):
    # Update vector and statistics
    x_new = jacobi_step(A, b, iterate['x'])
    E_new = quadratic_form(A, b, x_new)
    rnorm = compute_residual_norm(A, b, x_new)
    xdiffnorm = LA.norm(x_new - iterate['x'])
    denergy = abs(E_new - iterate['E'])

    # In-place update of dictionary
    iterate['iteration counter'] += 1
    iterate['x'] = x_new
    iterate['E'] = E_new
    iterate['2-norm of residual'] = rnorm
    iterate['2-norm of error'] = xdiffnorm
    iterate['absolute pseudoenergy difference'] = denergy


def checkpointer(iterate: Iterate):
    zarr.save('data/jacobi.zarr', iterate['x'])


def main():
    print('Experiments with linear solvers')
    dim = 50
    M = np.random.randn(dim, dim)
    # Make sure our matrix is SPD
    A = 0.5 * (M + M.transpose())
    A = A * A.transpose()
    A += dim * np.eye(dim)
    b = np.random.rand(dim)
    x_ref = LA.solve(A, b)

    # Jacobi method
    print('Jacobi algorithm')
    x_jacobi = jacobi(A, b, rtol=1.0e-4, etol=1.0e-5, max_it=25)
    print('Jacobi relative error to reference {:.5E}\n'.format(
        relative_error_to_reference(x_jacobi, x_ref)))

    # Jacobi method, with iterator
    it_count = Stat(
        '# it.',
        '{:d}',
        kind='failure',
        criterion=Criterion(
            threshold=25,
            comparison=ge,
            message='Maximum number of iterations ({threshold:d}) exceeded'))

    rnorm = Stat(
        '||r||_2',
        '{:.5E}',
        kind='success',
        criterion=Criterion(
            threshold=1.0e-4,
            comparison=le,
            message='2-norm of residual below threshold {threshold:.1E}'))

    denergy = Stat(
        'abs(Delta E)',
        '{:.5E}',
        kind='success',
        criterion=Criterion(
            threshold=1.0e-5,
            comparison=le,
            message='Pseudoenergy variation below threshold {threshold:.1E}'))

    xdiffnorm = Stat(
        '||x_new - x_old||_2',
        '{:.5E}',
        kind='success',
        criterion=Criterion(
            threshold=1.0e-4,
            comparison=le,
            message='2-norm of error below threshold {threshold:.1E}'))

    energy = Stat('E', '{:.5E}', kind='report')

    stats = {
        '2-norm of residual': rnorm,
        'absolute pseudoenergy difference': denergy,
        '2-norm of error': xdiffnorm,
        'E': energy,
        'iteration counter': it_count
    }

    x_0 = np.zeros_like(b)
    guess = Iterate({
        'x': x_0,
        'E': quadratic_form(A, b, x_0),
        '2-norm of residual': compute_residual_norm(A, b, x_0),
        'absolute pseudoenergy difference': 0.0,
        '2-norm of error': 0.0
    })
    jacobi_loose = IterativeSolver(
        partial(stepper, A, b), guess, stats, RuntimeError, checkpointer)

    for _ in jacobi_loose:
        pass

    print('\njacobi_loose.niterations ', jacobi_loose.niterations)
    print('Jacobi relative error to reference {:.5E}\n'.format(
        relative_error_to_reference(jacobi_loose.iterate['x'], x_ref)))


if __name__ == '__main__':
    main()
