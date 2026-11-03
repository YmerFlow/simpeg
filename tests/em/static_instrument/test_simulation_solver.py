"""simulation__solver picks the solver the simulation (and the BFGS preconditioner) uses.

'pardiso' used to fail with "NameError: name 'PardisoSolver' is not defined": the
module imported the class under another name.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import libaarhusxyz
from pymatsolver import SolverLU

from simpeg.electromagnetics.utils.static_instrument import base, dual


N_SOUNDINGS, N_GATES = 3, 4


class FakeGEX:
    def __init__(self):
        two_point = np.array([[-1e-3, 1.0], [0.0, 0.0]])
        self.General = {"TxLoopArea": 342.0, "RxCoilPosition": np.array([-13.25, 0.0, -2.0]),
                        "WaveformLMPoint": two_point, "WaveformHMPoint": two_point}
        self.gex_dict = {"General": self.General}
        for ch, (moment, dipole) in {1: ("LM", 2736.0), 2: ("HM", 451440.0)}.items():
            self.gex_dict["Channel%d" % ch] = {
                "TransmitterMoment": moment, "ReceiverPolarizationXYZ": "Z",
                "ApproxDipoleMoment": dipole, "GateFactor": 1.0, "NoGates": N_GATES}

    def gate_times(self, channel):
        return np.column_stack([np.logspace(-5, -3, N_GATES)] * 3)


def simulation(solver):
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({
        "x": np.arange(N_SOUNDINGS, dtype=float), "y": np.zeros(N_SOUNDINGS),
        "alt": np.full(N_SOUNDINGS, 30.0), "line_no": np.zeros(N_SOUNDINGS, dtype=int),
        "tilt_x": np.zeros(N_SOUNDINGS), "tilt_y": np.zeros(N_SOUNDINGS)})
    for ch in (1, 2):
        xyz.layer_data["dbdt_ch%dgt" % ch] = pd.DataFrame(np.full((N_SOUNDINGS, N_GATES), 1e-9))
        xyz.layer_data["dbdt_std_ch%dgt" % ch] = pd.DataFrame(np.full((N_SOUNDINGS, N_GATES), 0.05))

    class System(dual.MeasuredTEMXYZSystem):
        pass
    System.gex = FakeGEX()
    s = System(xyz, validate=False, simulation__solver=solver, simulation__parallel=False,
               gate_filter__start_lm=0, gate_filter__end_lm=N_GATES,
               gate_filter__start_hm=0, gate_filter__end_hm=N_GATES)
    return s.make_simulation(s.make_survey(), s.make_thicknesses())


def test_lu_is_the_default():
    assert simulation("LU").solver is SolverLU


@pytest.mark.skipif(not base.PARDISO_AVAILABLE, reason="pymatsolver's Pardiso is not available here")
def test_pardiso_builds_the_simulation_and_solves():
    sim = simulation("pardiso")
    assert sim.solver is base.PardisoSolver
    A = sp.diags([4.0, 5.0, 6.0]).tocsr() + sp.eye(3, k=1) + sp.eye(3, k=-1)
    np.testing.assert_allclose(A @ (sim.solver(A) * np.ones(3)), np.ones(3))


def test_pardiso_where_it_is_not_available_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(base, "PARDISO_AVAILABLE", False)
    with pytest.raises(ValueError, match="'pardiso', but pymatsolver's Pardiso is not available"):
        simulation("pardiso")
