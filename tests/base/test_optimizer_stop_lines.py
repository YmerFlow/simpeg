"""Every way out of Minimize.minimize writes one line saying which it was.

An inversion that stopped used to end in the optimizer's STOP block whatever
stopped it, so a stop requested by a directive read as the optimizer's own
criteria (ymerflow-business#132).
"""

import numpy as np
import scipy.sparse as sp
from discretize.tests import get_quadratic, rosenbrock

from simpeg import optimization


def quadratic():
    return get_quadratic(sp.identity(2).tocsr(), np.array([-5.0, -5.0]))


def test_converging_says_its_criteria_are_met(capsys):
    optimization.GaussNewton().minimize(quadratic(), np.array([0.0, 0.0]))
    assert "Optimizer stopping at iteration" in capsys.readouterr().out
    # (the message names the criteria; the STOP block after it has the numbers)


def test_running_out_of_iterations_says_so(capsys):
    optimization.GaussNewton(maxIter=1).minimize(rosenbrock, np.array([0.0, 0.0]))
    assert "Optimizer stopping at iteration 1: maxIter = 1 reached" in capsys.readouterr().out


def test_a_stop_asked_for_by_a_directive_says_so(capsys):
    opt = optimization.GaussNewton()
    opt.callback = lambda x: setattr(opt, "stopNextIteration", True)
    opt.minimize(rosenbrock, np.array([0.0, 0.0]))
    out = capsys.readouterr().out
    assert "Optimizer stopping at iteration 1: a directive asked it to stop" in out
    assert "convergence criteria" not in out


def test_a_failed_line_search_says_so(capsys):
    def uphill(x, return_g=True, return_H=True):
        """x.x with the gradient's sign flipped, so every step goes uphill."""
        out = (float(x @ x),)
        if return_g:
            out += (-2 * x,)
        if return_H:
            out += (2 * sp.identity(len(x)).tocsr(),)
        return out if len(out) > 1 else out[0]

    optimization.GaussNewton(maxIterLS=3).minimize(uphill, np.array([1.0, 1.0]))
    assert "the line search found no step that lowers the objective in 3 tries" in capsys.readouterr().out
