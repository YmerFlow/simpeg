import unittest
import warnings
from simpeg.utils import sdiag
import numpy as np
import pytest
import scipy.sparse as sp
from simpeg import optimization
from discretize.tests import get_quadratic, rosenbrock

TOL = 1e-2


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.A = sp.identity(2).tocsr()
        self.b = np.array([-5, -5])

    def test_GN_rosenbrock(self):
        GN = optimization.GaussNewton()
        xopt = GN.minimize(rosenbrock, np.array([0, 0]))
        x_true = np.array([1.0, 1.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_GN_quadratic(self):
        GN = optimization.GaussNewton()
        xopt = GN.minimize(get_quadratic(self.A, self.b), np.array([0, 0]))
        x_true = np.array([5.0, 5.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_ProjGradient_quadraticBounded(self):
        PG = optimization.ProjectedGradient(debug=True)
        PG.lower, PG.upper = -2, 2
        xopt = PG.minimize(get_quadratic(self.A, self.b), np.array([0, 0]))
        x_true = np.array([2.0, 2.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_ProjGradient_quadratic1Bound(self):
        myB = np.array([-5, 1])
        PG = optimization.ProjectedGradient()
        PG.lower, PG.upper = -2, 2
        xopt = PG.minimize(get_quadratic(self.A, myB), np.array([0, 0]))
        x_true = np.array([2.0, -1.0])
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)

    def test_NewtonRoot(self):
        def fun(x, return_g=True):
            if return_g:
                return np.sin(x), sdiag(np.cos(x))
            return np.sin(x)

        x = np.array([np.pi - 0.3, np.pi + 0.1, 0])
        xopt = optimization.NewtonRoot(comments=False).root(fun, x)
        x_true = np.array([np.pi, np.pi, 0])
        print("Newton Root Finding")
        print("xopt: ", xopt)
        print("x_true: ", x_true)
        self.assertTrue(np.linalg.norm(xopt - x_true, 2) < TOL, True)


# ── The CG tolerance reaches scipy's cg ──────────────────────────────────────
# SciPy calls it rtol from 1.12 and removed tol in 1.14. Before the fix, InexactGaussNewton's
# tolCG was dropped with a warning (cg ran to its default 1e-5), and ProjectedGradient's CG
# step raised TypeError.

N = 400
# Eigenvalues spread over three decades: CG's residual falls steadily, reaching 10 % in about
# 50 iterations and 1e-5 in about 170.
SPREAD = sp.diags(np.logspace(0, 3, N)).tocsr()
G = np.random.default_rng(0).normal(size=N)


def relative_residual(p, H, g):
    return np.linalg.norm(H @ p + g) / np.linalg.norm(g)


@pytest.mark.parametrize("tolCG", [0.1, 1e-6])
def test_inexact_gauss_newton_solves_to_tolCG(tolCG):
    opt = optimization.InexactGaussNewton()
    opt.tolCG, opt.maxIterCG = tolCG, 1000
    opt.xc, opt.H, opt.g = np.zeros(N), SPREAD, G
    opt.approxHinv = sp.linalg.aslinearoperator(sp.identity(N))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = opt.findSearchDirection()
    res = relative_residual(p, SPREAD, opt.g)
    assert res <= tolCG
    if tolCG == 0.1:
        assert res > 1e-3            # stopped near 10 %, not run on to cg's default 1e-5


def test_projected_gradient_cg_step_solves_to_tolCG():
    opt = optimization.ProjectedGradient()
    opt.tolCG, opt.maxIterCG = 0.1, 1000
    opt.lower, opt.upper = np.full(N, -1e3), np.full(N, 1e3)
    opt.explorePG, opt.exploreCG = False, True
    opt.xc, opt.H, opt.g = np.zeros(N), SPREAD, G
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = opt.findSearchDirection()
    assert opt._itType == ".CG."
    assert 1e-3 < relative_residual(p, SPREAD, opt.g) <= 0.1


def recursive_bfgs(S, Y, n, H0, d):
    """The recursive L-BFGS of earlier versions, with the pairs as the columns of S and Y."""
    nn = min(S.shape[1], n)

    def rec(k, d):
        if k < 0:
            return H0 * d
        khat = 0 if nn == 0 else np.mod(n - nn + k, nn)
        gamma = np.vdot(S[:, khat], d) / np.vdot(Y[:, khat], S[:, khat])
        d = rec(k - 1, d - gamma * Y[:, khat])
        ys = np.vdot(Y[:, khat], S[:, khat])
        return d + (gamma - np.vdot(Y[:, khat], d) / ys) * S[:, khat]

    return rec(nn, d)


def test_two_loop_bfgs_matches_the_recursive_form():
    rng = np.random.default_rng(0)
    opt = optimization.InexactGaussNewton()
    opt._startup_BFGS(np.zeros(N))
    opt.bfgsH0 = optimization.SolverDiag(sp.diags(rng.uniform(1.0, 3.0, N)))
    S, Y = np.zeros((N, opt.nbfgs)), np.zeros((N, opt.nbfgs))
    d = rng.normal(size=N)
    d_in = d.copy()
    # From no pairs, through a part-filled memory, to one that has wrapped around twice.
    for _ in range(2 * opt.nbfgs + 3):
        expected = recursive_bfgs(S, Y, opt._bfgscnt, opt.bfgsH0, d)
        np.testing.assert_allclose(opt.bfgs(d), expected, rtol=1e-12,
                                   atol=1e-12 * np.abs(expected).max())
        assert np.array_equal(d, d_in)
        s = rng.normal(size=N)
        y = s + 0.3 * rng.normal(size=N)                 # y . s > 0, so the pair is kept
        opt.iter, opt.g_last, opt.g, opt.xc = 1, np.zeros(N), y, s
        opt._doEndIteration_BFGS(np.zeros(N))
        assert opt.comment == ""
        S[:, opt._bfgscnt % opt.nbfgs], Y[:, opt._bfgscnt % opt.nbfgs] = s, y


if __name__ == "__main__":
    unittest.main()
