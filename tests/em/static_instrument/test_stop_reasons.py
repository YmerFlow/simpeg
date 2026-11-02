"""The target misfit counts the data in use, and every stop says why.

Wave-1 inversions stopped at a logged RMS of 1.4-1.7 with no reason in the log
(ymerflow-business#132). TargetMisfit had stopped them, silently, against a
target of every gate slot, nD, while phi_d sums only gates with a weight. A
third of the slots on a line can be empty (no datum, or no error), so the runs
stopped at RMS sqrt(nD / in use), about 1.2 on the gates in use.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

import libaarhusxyz

from simpeg.directives import TargetMisfit
from simpeg.electromagnetics.utils.static_instrument import dual
from simpeg.electromagnetics.utils.static_instrument.schedules import (
    MisfitDrivenBetaSchedule, TargetMisfitInUse)


N_SOUNDINGS, N_GATES = 4, 5


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


def system_with_empty_gates():
    """Two moments of 5 gates on 4 soundings: 40 slots, 9 of them empty."""
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({
        "x": np.arange(N_SOUNDINGS, dtype=float), "y": np.zeros(N_SOUNDINGS),
        "alt": np.full(N_SOUNDINGS, 30.0), "line_no": np.zeros(N_SOUNDINGS, dtype=int),
        "tilt_x": np.zeros(N_SOUNDINGS), "tilt_y": np.zeros(N_SOUNDINGS)})
    for ch in (1, 2):
        dbdt = np.full((N_SOUNDINGS, N_GATES), 1e-9)
        std = np.full((N_SOUNDINGS, N_GATES), 0.05)
        xyz.layer_data["dbdt_ch%dgt" % ch] = pd.DataFrame(dbdt)
        xyz.layer_data["dbdt_std_ch%dgt" % ch] = pd.DataFrame(std)
    xyz.layer_data["dbdt_ch1gt"].iloc[:, -2:] = np.nan     # 8 late LM gates culled
    xyz.layer_data["dbdt_std_ch2gt"].iloc[0, 0] = np.nan   # 1 HM gate with no error

    class System(dual.DualMomentTEMXYZSystem):
        pass
    System.gex = FakeGEX()
    s = System(xyz, validate=False)
    s.options.update(gate_filter__start_lm=0, gate_filter__end_lm=N_GATES,
                     gate_filter__start_hm=0, gate_filter__end_hm=N_GATES)
    return s


def test_the_target_counts_only_the_data_in_use(capsys):
    s = system_with_empty_gates()
    target = next(d for d in s.make_directives() if isinstance(d, TargetMisfit))
    assert isinstance(target, TargetMisfitInUse)
    dmisfit = s.make_misfit(s.make_thicknesses())      # the weights phi_d is summed with
    target.inversion = SimpleNamespace(invProb=SimpleNamespace(dmisfit=dmisfit))
    target.initialize()
    assert dmisfit.simulation.survey.nD == 40
    assert target.phi_d_star == 31 and target.target == 31
    assert "target phi_d 31 = chifact 1 x 31 data in use (9 of 40 have weight 0)" in capsys.readouterr().out


def test_target_misfit_says_why_it_stops(capsys):
    target = TargetMisfit(phi_d_star=100.0)
    opt = SimpleNamespace(stopNextIteration=False, print_type=None)
    target.inversion = SimpleNamespace(invProb=SimpleNamespace(phi_d=80.0, opt=opt))
    target.endIter()
    assert opt.stopNextIteration is True
    assert "TargetMisfit stopping: phi_d 80 < target 100 (chifact 1 x phi_d_star 100)" in capsys.readouterr().out


def test_the_schedule_says_when_the_target_is_reached(capsys):
    target = TargetMisfit()
    target.target = 50.0
    schedule = MisfitDrivenBetaSchedule()
    opt = SimpleNamespace(iter=0, stopNextIteration=False, print_type=None)
    schedule.inversion = SimpleNamespace(
        invProb=SimpleNamespace(beta=100.0, phi_d=400.0, phi_d_last=np.nan, opt=opt),
        directiveList=SimpleNamespace(dList=[target]))
    schedule.initialize()
    schedule.invProb.phi_d, opt.iter = 45.0, 1
    schedule.endIter()
    assert "phi_d=45 target=50: target reached, beta held at 100" in capsys.readouterr().out
