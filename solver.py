import numpy as np
from numpy import linalg as LA


def jacobi(A, b, rtol=1.0e-8, max_it=25, x_0=None):
    if x_0 is None:
        x = np.zeros_like(b)
    else:
        x = x_0

    it = 0
    print('Iteration #    Residual norm')
    while it < max_it:
        # Compute residual
        r = apply_jacobi(A, b, x) - x
        rnorm = LA.norm(r)

        # Report
        print('    {:4d}        {:.5E}'.format(it, rnorm))

        # Check convergence
        if rnorm < rtol:
            break

        # Update solution vector
        x = apply_jacobi(A, b, x)
        it += 1
    else:
        raise RuntimeError(
            'Maximum number of iterations ({0:d}) exceeded, but residual norm {1:.5E} still greater than threshold {2:.5E}'.
            format(max_it, rnorm, rtol))
    return x


def jacobi_diis(A, b, rtol=1.0e-8, max_it=25, max_diis_hist=8, x_0=None):
    if x_0 is None:
        x = np.zeros_like(b)
    else:
        x = x_0

    # Lists of iterates and residuals
    xs = []
    rs = []

    it = 0
    print('Iteration #    Residual norm')
    while it < max_it:
        # Compute residual
        r = apply_jacobi(A, b, x) - x
        rnorm = LA.norm(r)

        # Collect DIIS history
        xs.append(x)
        rs.append(r)

        # Report
        print('    {:4d}        {:.5E}'.format(it, rnorm))

        # Check convergence
        if rnorm < rtol:
            break

        if it >= 2:
            # Prune DIIS history
            diis_hist = len(xs)
            if diis_hist > max_diis_hist:
                xs.pop(0)
                rs.pop(0)
                diis_hist -= 1

            # Build error matrix B
            B = np.empty((diis_hist + 1, diis_hist + 1))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0
            for i, ri in enumerate(rs):
                for j, rj in enumerate(rs):
                    if j > i: continue
                    val = np.einsum('i,i', ri, rj)
                    B[i, j] = val
                    B[j, i] = val

            # Normalize B
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            # Build rhs vector
            rhs = np.zeros(diis_hist + 1)
            rhs[-1] = -1

            # Solve Pulay equations
            cs = LA.solve(B, rhs)

            # Calculate new solution as linear
            # combination of previous solutions
            x = np.zeros_like(x)
            for i, c in enumerate(cs[:-1]):
                x += c * xs[i]

        # Update solution vector
        x = apply_jacobi(A, b, x)
        it += 1
    else:
        raise RuntimeError(
            'Maximum number of iterations ({0:d}) exceeded, but residual norm {1:.5E} still greater than threshold {2:.5E}'.
            format(max_it, rnorm, rtol))
    return x


def jacobi_kain(A, b, rtol=1.0e-8, max_it=25, max_kain_hist=8, x_0=None):
    if x_0 is None:
        x = np.zeros_like(b)
    else:
        x = x_0

    # Lists of iterates and residuals
    xs = []
    rs = []

    it = 0
    print('Iteration #    Residual norm')
    while it < max_it:
        # Compute residual
        r = apply_jacobi(A, b, x) - x
        rnorm = LA.norm(r)

        # Collect DIIS history
        xs.append(x)
        rs.append(r)

        # Report
        print('    {:4d}        {:.5E}'.format(it, rnorm))

        # Check convergence
        if rnorm < rtol:
            break

        if it >= 2:
            # Prune KAIN history
            kain_hist = len(xs)
            if kain_hist > max_kain_hist:
                xs.pop(0)
                rs.pop(0)
                kain_hist -= 1

            # Build Q matrix
            Q = np.empty((kain_hist, kain_hist))
            for i, x in enumerate(xs):
                for j, r in enumerate(rs):
                    Q[i, j] = np.einsum('i,i', x, r)

            # Build subspace matrix B and vector rhs
            B = np.empty((kain_hist - 1, kain_hist - 1))
            rhs = np.zeros(kain_hist - 1)
            for i in range(kain_hist - 1):
                rhs[i] = -Q[i, -1] + Q[-1, -1]
                for j in range(kain_hist - 1):
                    B[i, j] = Q[i, j] - Q[i, -1] - Q[-1, j] + Q[-1, -1]

            # Solve KAIN equations
            cs = LA.solve(B, rhs)
            cs = np.append(cs, 1.0 - np.sum(cs))

            # Calculate new solution as linear
            # combination of previous solutions
            x = np.zeros_like(x)
            for i, c in enumerate(cs):
                x += c * xs[i] - c * rs[i]

        # Update solution vector
        x = apply_jacobi(A, b, x)
        it += 1
    else:
        raise RuntimeError(
            'Maximum number of iterations ({0:d}) exceeded, but residual norm {1:.5E} still greater than threshold {2:.5E}'.
            format(max_it, rnorm, rtol))
    return x


def apply_jacobi(A, b, x):
    D = np.diag(A)
    O = A - np.diagflat(D)
    return (b - np.einsum('ij,j->i', O, x)) / D


def cg(A, b, rtol=1.0e-8, max_it=25, x_0=None):
    if x_0 is None:
        x = np.zeros_like(b)
    else:
        x = x_0

    it = 0
    r = b - np.einsum('ij,j->i', A, x)
    p = r
    print('Iteration #    Residual norm')
    while it < max_it:
        Ap = np.einsum('ij,j->i', A, p)
        rtr = np.einsum('i,i', r, r)
        alpha = rtr / np.einsum('i,i', p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rnorm = LA.norm(r)

        # Report
        print('    {:4d}        {:.5E}'.format(it, rnorm))

        # Check convergence
        if rnorm < rtol:
            break

        beta = np.einsum('i,i', r, r) / rtr
        p = r + beta * p
        it += 1
    else:
        raise RuntimeError(
            'Maximum number of iterations ({0:d}) exceeded, but residual norm {1:.5E} still greater than threshold {2:.5E}'.
            format(max_it, rnorm, rtol))
    return x


def cr(A, b, rtol=1.0e-8, max_it=25, x_0=None):
    if x_0 is None:
        x = np.zeros_like(b)
    else:
        x = x_0

    it = 0
    r = b - np.einsum('ij,j->i', A, x)
    p = r
    print('Iteration #    Residual norm')
    while it < max_it:
        Ap = np.einsum('ij,j->i', A, p)
        rtAr = np.einsum('i,ij,j', r, A, r)
        alpha = rtAr / np.einsum('i,i', Ap, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rnorm = LA.norm(r)

        # Report
        print('    {:4d}        {:.5E}'.format(it, rnorm))

        # Check convergence
        if rnorm < rtol:
            break

        beta = np.einsum('i,ij,j', r, A, r) / rtAr
        p = r + beta * p
        it += 1
    else:
        raise RuntimeError(
            'Maximum number of iterations ({0:d}) exceeded, but residual norm {1:.5E} still greater than threshold {2:.5E}'.
            format(max_it, rnorm, rtol))
    return x


def relative_error_to_reference(x, x_ref):
    return LA.norm(x - x_ref) / LA.norm(x_ref)


def main():
    print('Experiments with linear solvers')
    dim = 1000
    M = np.random.randn(dim, dim)
    # Make sure our matrix is SPD
    A = 0.5 * (M + M.transpose())
    A = A * A.transpose()
    A += dim * np.eye(dim)
    b = np.random.rand(dim)
    x_ref = LA.solve(A, b)

    # Jacobi method
    print('Jacobi algorithm')
    x_jacobi = jacobi(A, b)
    print('Jacobi relative error to reference {:.5E}\n'.format(
        relative_error_to_reference(x_jacobi, x_ref)))

    # Jacobi-DIIS method
    print('Jacobi-DIIS algorithm')
    x_jdiis = jacobi_diis(A, b)
    print('Jacobi-DIIS relative error to reference {:.5E}\n'.format(
        relative_error_to_reference(x_jdiis, x_ref)))

    # Jacobi-KAIN method
    print('Jacobi-KAIN algorithm')
    x_jkain = jacobi_kain(A, b)
    print('Jacobi-KAIN relative error to reference {:.5E}\n'.format(
        relative_error_to_reference(x_jkain, x_ref)))

    # Conjugate gradient method
    print('Conjugate Gradient algorithm')
    x_cg = cg(A, b)
    print('CG relative error to reference {:.5E}\n'.format(
        relative_error_to_reference(x_cg, x_ref)))

    # Preconditioned conjugate gradient method
    #print('Preconditioned conjugate Gradient algorithm')
    #x_pcg = cg(A, b)
    #print('PCG relative error to reference {:.5E}\n'.format(
    #    relative_error_to_reference(x_pcg, x_ref)))

    # Conjugate residual method
    print('Conjugate Residual algorithm')
    x_cr = cr(A, b)
    print('CR relative error to reference {:.5E}\n'.format(
        relative_error_to_reference(x_cr, x_ref)))


if __name__ == '__main__':
    main()
