"""One beta for a whole survey, with the misfit as QC (docs/plans/fixed-beta.md).

The directives are driven by hand on a fake inversion, as in
``test_misfit_driven_beta.py``; the wiring is checked through
``make_directives``, and one small inversion runs end to end.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import libaarhusxyz

from simpeg.directives import BetaSchedule
from simpeg.electromagnetics.utils.static_instrument import dual
from simpeg.electromagnetics.utils.static_instrument.schedules import (
    BetaFloor,
    MisfitDrivenBetaSchedule,
    RecordBeta,
    RelativeNormChangeStop,
    StartingBeta,
    TargetMisfitInUse,
)

N_SOUNDINGS, N_GATES = 4, 5


def fake_inversion(beta, target=50.0):
    """A fake inversion with a TargetMisfit at ``target`` in its directive list."""
    opt = SimpleNamespace(iter=0, stopNextIteration=False, print_type=None)
    inv_prob = SimpleNamespace(
        beta=beta, phi_d=np.nan, phi_d_last=np.nan, phi_m=np.nan, opt=opt
    )
    tm = TargetMisfitInUse()
    tm.target = target
    return SimpleNamespace(invProb=inv_prob, directiveList=SimpleNamespace(dList=[tm]))


def drive(directives, start_misfit, misfits, beta0=100.0, phi_m=None):
    """Run ``directives`` over a misfit trajectory, one endIter per misfit."""
    inv = fake_inversion(beta0)
    inv.invProb.phi_d = start_misfit
    for d in directives:
        d.inversion = inv
        d.initialize()
    for i, phi_d in enumerate(misfits, start=1):
        if inv.invProb.opt.stopNextIteration:
            break
        inv.invProb.phi_d_last = inv.invProb.phi_d
        inv.invProb.phi_d = phi_d
        inv.invProb.phi_m = 1.0 if phi_m is None else phi_m[i - 1]
        inv.invProb.opt.iter = i
        for d in directives:
            d.endIter()
    return inv


# ── The misfit schedule ──────────────────────────────────────────────────────


def test_a_cooling_factor_of_one_is_a_hold_and_never_stalls():
    d = MisfitDrivenBetaSchedule(
        cooling_factor=1.0, stall_iterations=3, stall_progress=0.05, verbose=False
    )
    # 1% per iteration for 10 iterations: a stall at any factor above 1.
    misfits = list(400.0 * 0.99 ** np.arange(1, 11))
    inv = drive([d], 400.0, misfits)
    assert [h["action"] for h in d.history] == ["hold"] * 10
    assert inv.invProb.beta == 100.0
    assert inv.invProb.opt.stopNextIteration is False


def test_beta_min_is_an_absolute_floor_in_place_of_the_ratio():
    d = MisfitDrivenBetaSchedule(
        cooling_factor=2.0, beta_min_ratio=1e-8, beta_min=30.0, verbose=False
    )
    inv = drive([d], 400.0, [399.0, 398.0, 397.0])
    assert [h["beta_after"] for h in d.history] == [50.0, 30.0, 30.0]
    assert d.history[2]["action"] == "floor"
    assert inv.invProb.beta == 30.0


def test_a_stall_on_the_floor_says_so(capsys):
    d = MisfitDrivenBetaSchedule(
        beta_min=100.0, stall_iterations=2, stall_progress=0.05, verbose=False
    )
    inv = drive([d], 400.0, [399.0, 398.0, 397.0])
    assert inv.invProb.opt.stopNextIteration is True
    assert "while beta sat on its floor of 100" in d.stopped_reason


# ── The floor for the fixed schedule ─────────────────────────────────────────


def test_the_fixed_schedule_cools_to_beta_min_and_holds():
    schedule = BetaSchedule(coolingFactor=2, coolingRate=1)
    inv = drive([schedule, BetaFloor(30.0)], 400.0, [300.0, 200.0, 100.0])
    assert inv.invProb.beta == 30.0


def test_a_starting_beta_below_the_floor_is_raised():
    inv = drive([StartingBeta(10.0), BetaFloor(30.0)], 400.0, [])
    assert inv.invProb.beta == 30.0


# ── The convergence stop ─────────────────────────────────────────────────────


def test_stops_when_the_objective_stops_falling_at_a_held_beta(capsys):
    stop = RelativeNormChangeStop(0.01)
    # objective = phi_d + 100 * 1: 500, 300, 298 (0.7% fall) -> stop after the third
    inv = drive([stop], 600.0, [400.0, 200.0, 198.0, 197.0])
    assert inv.invProb.opt.stopNextIteration is True
    assert inv.invProb.opt.iter == 3
    assert "converged at this beta" in stop.stopped_reason
    assert "RelativeNormChangeStop it=3" in capsys.readouterr().out


def test_no_test_across_a_change_of_beta():
    stop = RelativeNormChangeStop(0.01)
    schedule = BetaSchedule(coolingFactor=2, coolingRate=1)
    # The objective barely moves, but beta halves every iteration.
    inv = drive([stop, schedule], 600.0, [400.0, 399.0, 398.0])
    assert inv.invProb.opt.stopNextIteration is False
    assert stop.stopped_reason is None


# ── The target-misfit stop ───────────────────────────────────────────────────


def target_with_data(n_used, n_empty, chifact=1.0, stop=True):
    """A TargetMisfitInUse initialized on ``n_used`` weighted data."""
    tm = TargetMisfitInUse(chifact=chifact)
    tm.stop = stop
    weights = sp.diags(np.r_[np.ones(n_used), np.zeros(n_empty)])
    opt = SimpleNamespace(iter=0, stopNextIteration=False, print_type=None)
    dmisfit = SimpleNamespace(objfcts=[SimpleNamespace(W=weights)])
    inv_prob = SimpleNamespace(dmisfit=dmisfit, phi_d=np.nan, opt=opt)
    tm.inversion = SimpleNamespace(invProb=inv_prob)
    tm.initialize()
    return tm


def test_chifact_scales_the_target():
    assert target_with_data(40, 10, chifact=2.0).target == 80.0


def test_with_the_target_off_the_run_goes_on(capsys):
    tm = target_with_data(40, 10, stop=False)
    for i, phi_d in enumerate([50.0, 30.0, 20.0], start=1):
        tm.invProb.phi_d, tm.opt.iter = phi_d, i
        tm.endIter()
    assert tm.opt.stopNextIteration is False
    out = capsys.readouterr().out
    assert "reaching it does not stop the run" in out
    assert out.count("not stopping") == 1


def test_with_the_target_on_it_stops():
    tm = target_with_data(40, 10)
    tm.invProb.phi_d = 30.0
    tm.endIter()
    assert tm.opt.stopNextIteration is True


# ── Wiring ───────────────────────────────────────────────────────────────────


class FakeGEX:
    def __init__(self):
        two_point = np.array([[-1e-3, 1.0], [0.0, 0.0]])  # time [s], current [A]
        self.General = {
            "TxLoopArea": 342.0,
            "RxCoilPosition": np.array([-13.25, 0.0, -2.0]),
            "WaveformLMPoint": two_point,
            "WaveformHMPoint": two_point,
        }
        self.gex_dict = {"General": self.General}
        for ch, (moment, dipole) in {1: ("LM", 2736.0), 2: ("HM", 451440.0)}.items():
            self.gex_dict["Channel%d" % ch] = {
                "TransmitterMoment": moment,
                "ReceiverPolarizationXYZ": "Z",
                "ApproxDipoleMoment": dipole,
                "GateFactor": 1.0,
                "NoGates": N_GATES,
            }

    def gate_times(self, channel):
        return np.column_stack([np.logspace(-5, -3, N_GATES)] * 3)


def system(**options):
    """A two-moment system on four soundings of one line, data from a 50 ohm.m halfspace."""
    xyz = libaarhusxyz.XYZ()
    n = N_SOUNDINGS
    xyz.flightlines = pd.DataFrame(
        {
            "x": 10.0 * np.arange(n),
            "y": np.zeros(n),
            "alt": np.full(n, 30.0),
            "line_no": np.zeros(n, dtype=int),
            "tilt_x": np.zeros(n),
            "tilt_y": np.zeros(n),
        }
    )
    for ch in (1, 2):
        xyz.layer_data["dbdt_ch%dgt" % ch] = pd.DataFrame(np.full((n, N_GATES), 1e-9))
        xyz.layer_data["dbdt_std_ch%dgt" % ch] = pd.DataFrame(
            np.full((n, N_GATES), 0.05)
        )

    class System(dual.MeasuredTEMXYZSystem):
        pass

    System.gex = FakeGEX()
    s = System(xyz, validate=False)
    s.options.update(
        gate_filter__start_lm=0,
        gate_filter__end_lm=N_GATES,
        gate_filter__start_hm=0,
        gate_filter__end_hm=N_GATES,
        **options,
    )
    return s


def kinds(directives):
    return [type(d).__name__ for d in directives]


def test_defaults_build_the_same_directives_as_before():
    dirs = system().make_directives()
    assert kinds(dirs)[1:] == ["BetaSchedule", "TargetMisfitInUse", "RecordBeta"]
    assert dirs[2].stop is True and dirs[2].chifact == 1.0


def test_schedule_none_holds_the_given_beta():
    dirs = system(
        directives__beta__schedule="none", directives__beta__beta0=300.0
    ).make_directives()
    assert kinds(dirs) == ["StartingBeta", "TargetMisfitInUse", "RecordBeta"]
    assert dirs[0].beta0 == 300.0


def test_the_new_options_reach_their_directives():
    dirs = system(
        directives__beta__schedule="misfit",
        directives__beta__beta_min=30.0,
        directives__target__enable=False,
        directives__target__chifact=1.5,
        optimizer__rel_norm_change=0.01,
    ).make_directives()
    assert kinds(dirs)[1:] == [
        "RelativeNormChangeStop",
        "MisfitDrivenBetaSchedule",
        "BetaFloor",
        "TargetMisfitInUse",
        "RecordBeta",
    ]
    assert dirs[1].rel_change == 0.01 and dirs[2].beta_min == 30.0
    assert dirs[3].beta_min == 30.0
    assert dirs[4].stop is False and dirs[4].chifact == 1.5


def test_the_convergence_stop_is_refused_with_irls():
    s = system(optimizer__rel_norm_change=0.01, directives__irls__enable=True)
    with pytest.raises(ValueError, match="IRLS"):
        s.make_directives()


def test_a_held_beta_run_end_to_end(capsys):
    s = system(
        directives__beta__schedule="none",
        directives__beta__beta0=10.0,
        directives__target__enable=False,
        optimizer__rel_norm_change=0.05,
        optimizer__max_iter=15,
        simulation__parallel=False,
        simulation__n_cpu=1,
        doi__enable=False,
    )
    s.invert()
    recorder = next(d for d in s.inv.directiveList.dList if isinstance(d, RecordBeta))
    assert recorder.beta == 10.0
    assert s.l2.model_info["final_beta"] == 10.0
    out = capsys.readouterr().out
    assert "##MISFIT## final phi_d" in out
    stop = next(
        d for d in s.inv.directiveList.dList if isinstance(d, RelativeNormChangeStop)
    )
    assert stop.stopped_reason is not None or s.inv.invProb.opt.iter >= 15
