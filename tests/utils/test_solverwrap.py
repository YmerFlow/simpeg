import unittest
import warnings
from simpeg.utils.solver_utils import (
    Solver, SolverLU, SolverCG, SolverBiCG, SolverDiag, SolverWrapI)
import scipy.sparse as sp
import numpy as np
import pytest


class TestSolve(unittest.TestCase):
    def setUp(self):
        # Create a random matrix
        n = 400
        A = sp.random(n, n, density=0.25)

        self.n = n
        self.A = 0.5 * (A + A.T) + n * sp.eye(n)

    def test_Solver(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = Solver(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2)

    def test_SolverLU(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = SolverLU(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2)

    def test_SolverCG(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = SolverCG(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2, decimal=4)

    def test_SolverBiCG(self):
        x = np.random.rand(self.n)
        b = self.A @ x

        with self.assertWarns(Warning):
            Ainv = SolverBiCG(self.A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2, decimal=4)

    def test_SolverDiag(self):
        x = np.random.rand(self.n)
        A = sp.diags(np.random.randn(self.n))
        b = A @ x

        with self.assertWarns(Warning):
            Ainv = SolverDiag(A, bad_kwarg=312)
        x2 = Ainv @ b
        np.testing.assert_almost_equal(x, x2)


# ── The relative tolerance reaches the solver under the name it takes ────────
# SciPy's iterative solvers call it rtol from 1.12 (tol removed in 1.14); simpeg passes tol.


def solver_taking(name, calls):
    """A stand-in iterative solver whose tolerance keyword is ``name``; records its calls."""
    if name == "rtol":
        def fun(A, b, x0=None, *, rtol=1e-5, atol=0.0, maxiter=None, M=None):
            calls.append({"rtol": rtol, "maxiter": maxiter})
            return np.zeros_like(b), 0
    else:
        def fun(A, b, x0=None, tol=1e-5, maxiter=None, M=None, atol=None):
            calls.append({"tol": tol, "maxiter": maxiter})
            return np.zeros_like(b), 0
    return fun


@pytest.mark.parametrize("given, takes", [("tol", "rtol"), ("rtol", "tol"), ("tol", "tol"),
                                          ("rtol", "rtol")])
def test_tolerance_is_passed_under_the_solvers_name(given, takes):
    calls = []
    Ainv = SolverWrapI(solver_taking(takes, calls), checkAccuracy=False)(
        sp.identity(3).tocsr(), **{given: 0.1, "maxiter": 7})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Ainv * np.ones(3)
    assert calls == [{takes: 0.1, "maxiter": 7}]


def test_scipy_cg_stops_at_the_tolerance_given_as_tol():
    n = 400                     # eigenvalues over three decades: CG's residual falls steadily
    A = sp.diags(np.logspace(0, 3, n)).tocsr()
    b = np.random.default_rng(0).normal(size=n)

    def solve(tol):
        iterations = []
        Ainv = SolverCG(A, tol=tol, maxiter=1000, checkAccuracy=False,
                        callback=lambda xk: iterations.append(1))
        x = Ainv * b
        return len(iterations), np.linalg.norm(A @ x - b) / np.linalg.norm(b)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        n_coarse, res_coarse = solve(0.1)
        n_fine, res_fine = solve(1e-5)
    assert res_coarse <= 0.1 and res_fine <= 1e-5
    assert n_coarse < n_fine / 2


if __name__ == "__main__":
    unittest.main()
