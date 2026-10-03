"""The misfit-driven beta schedule, exercised on a fake inversion.

The directive is deliberately thin: it reads ``invProb.beta`` and
``invProb.phi_d``, the optimizer's iteration counter, and the ``TargetMisfit``
in its directive list, and writes ``invProb.beta`` and
``opt.stopNextIteration``. Those are the only things a test has to provide, so
the whole schedule can be driven by hand, iteration by iteration, with the
misfit trajectory chosen to exercise each branch.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from SimPEG.directives import BetaSchedule, TargetMisfit
from SimPEG.electromagnetics.utils.static_instrument.schedules import MisfitDrivenBetaSchedule


# ── A fake inversion the directive can run against ───────────────────────────

class FakeInversion:
    def __init__(self, target, beta0):
        self.invProb = SimpleNamespace(beta=beta0, phi_d=np.nan, phi_d_last=np.nan,
                                       opt=SimpleNamespace(iter=0, stopNextIteration=False, print_type=None))
        tm = TargetMisfit()
        tm.target = target
        self.directiveList = SimpleNamespace(dList=[tm])


# ── Behaviour ────────────────────────────────────────────────────────────────
# drive() feeds the misfit at the start model as phi_d_last of iteration 1, so
# the first iteration has a real progress number like every later one.

def drive_from(directive, start_misfit, misfits, beta0=100.0, target=50.0):
    inv = FakeInversion(target, beta0)
    inv.invProb.phi_d = start_misfit
    directive.inversion = inv
    directive.initialize()
    for i, phi_d in enumerate(misfits, start=1):
        if inv.invProb.opt.stopNextIteration:
            break
        inv.invProb.phi_d_last = inv.invProb.phi_d
        inv.invProb.phi_d = phi_d
        inv.invProb.opt.iter = i
        directive.endIter()
    return inv


def test_holds_beta_while_the_misfit_is_still_falling():
    d = MisfitDrivenBetaSchedule(progress_threshold=0.10, verbose=False)
    inv = drive_from(d, 1000.0, misfits=[600.0, 400.0, 300.0], beta0=100.0, target=50.0)
    assert [h["action"] for h in d.history] == ["hold", "hold", "hold"]
    assert inv.invProb.beta == 100.0


def test_cools_on_a_plateau_harder_far_from_target_than_near_it():
    d = MisfitDrivenBetaSchedule(cooling_factor=2.0, ratio_cap=4.0, verbose=False)   # ratio scaling switched on
    # Plateau at 395 (ratio 7.9, capped at 4 -> factor 8), progress, then plateau at 57 (ratio 1.14 -> factor 2.28).
    inv = drive_from(d, 400.0, misfits=[395.0, 58.0, 57.0], beta0=100.0, target=50.0)
    acts = [h["action"] for h in d.history]
    assert acts == ["cool", "hold", "cool"]
    assert d.history[0]["beta_after"] == pytest.approx(100.0 / 8)
    assert d.history[2]["beta_after"] == pytest.approx((100.0 / 8) / (2.0 * 57.0 / 50.0))
    assert inv.invProb.beta == pytest.approx(d.history[2]["beta_after"])


def test_a_rising_misfit_counts_as_a_plateau():
    d = MisfitDrivenBetaSchedule(verbose=False)
    drive_from(d, 400.0, misfits=[420.0], beta0=100.0, target=50.0)
    assert d.history[0]["action"] == "cool" and d.history[0]["progress"] < 0


def test_stops_when_a_window_of_cooling_bought_almost_no_misfit():
    d = MisfitDrivenBetaSchedule(progress_threshold=0.10, stall_iterations=4, stall_progress=0.05, verbose=False)
    # 1% per iteration: each iteration is a plateau (cool), and over the 4-iteration window only ~4% < 5%.
    inv = drive_from(d, 400.0, misfits=[396.0, 392.0, 388.0, 384.0, 380.0, 376.0], beta0=100.0, target=50.0)
    assert all(h["action"] == "cool" for h in d.history)
    assert inv.invProb.opt.stopNextIteration is True
    assert len(d.history) == 5                                     # window of 4 cools needs 5 misfits; the 6th was never seen
    assert "Cooling is not buying misfit" in d.stopped_reason
    assert inv.invProb.beta == pytest.approx(100.0 / 2 ** 5)       # it did keep cooling while it tried


def test_slow_but_steady_progress_is_not_a_stall():
    d = MisfitDrivenBetaSchedule(progress_threshold=0.10, stall_iterations=4, stall_progress=0.05, verbose=False)
    # 4% per iteration is below the hold threshold (so it cools every time) but ~15% over a 4-iteration window.
    inv = drive_from(d, 400.0, misfits=[384.0, 369.0, 354.0, 340.0, 326.0, 313.0], beta0=100.0, target=50.0)
    assert all(h["action"] == "cool" for h in d.history)
    assert inv.invProb.opt.stopNextIteration is False
    assert len(d.history) == 6


def test_a_hold_restarts_the_stall_window():
    d = MisfitDrivenBetaSchedule(progress_threshold=0.10, stall_iterations=3, stall_progress=0.05, verbose=False)
    inv = drive_from(d, 400.0, misfits=[399.0, 398.0, 200.0, 199.0, 198.0, 197.0], beta0=100.0, target=50.0)
    assert [h["action"] for h in d.history] == ["cool", "cool", "hold", "cool", "cool", "cool"]
    # window after the hold: 200 -> 197 over 3 cools = 1.5% < 5%  -> stop, but not before
    assert inv.invProb.opt.stopNextIteration is True and len(d.history) == 6


def test_floor_is_honoured_and_a_plateau_on_it_counts_toward_the_stall():
    d = MisfitDrivenBetaSchedule(cooling_factor=10.0, ratio_cap=4.0, beta_min_ratio=1e-3,
                                 stall_iterations=3, stall_progress=0.05, verbose=False)
    # Every iteration a plateau far above target: factor 40 each time; the floor (0.1) is hit on the 2nd cool.
    inv = drive_from(d, 4000.0, misfits=[3990.0, 3980.0, 3970.0, 3960.0], beta0=100.0, target=50.0)
    betas = [h["beta_after"] for h in d.history]
    assert betas[0] == pytest.approx(2.5)
    assert betas[1] == pytest.approx(0.1)                         # clamped, not 2.5/40
    assert d.history[2]["action"] == "floor"
    assert inv.invProb.opt.stopNextIteration is True              # 3-iteration window, <5% progress
    assert min(betas) >= 0.1 - 1e-12


def test_does_not_cool_once_target_is_reached():
    d = MisfitDrivenBetaSchedule(verbose=False)
    inv = drive_from(d, 400.0, misfits=[395.0, 45.0], beta0=100.0, target=50.0)
    assert d.history[1]["action"] == "target"
    assert d.history[1]["beta_after"] == d.history[1]["beta_before"]
    assert inv.invProb.opt.stopNextIteration is False  # that is TargetMisfit's job, not ours


def test_picks_up_beta0_at_first_iteration_when_no_estimator_ran():
    d = MisfitDrivenBetaSchedule(beta_min_ratio=0.5, verbose=False)
    inv = FakeInversion(target=50.0, beta0=np.nan)    # nothing has set beta yet
    d.inversion = inv
    d.initialize()
    assert d._beta0 is None
    inv.invProb.beta = 80.0                            # an estimator sets it before the first endIter
    inv.invProb.phi_d_last = 400.0
    inv.invProb.phi_d = 399.0                          # a plateau, so it cools
    inv.invProb.opt.iter = 1
    d.endIter()
    assert d._beta0 == 80.0 and d.beta_floor == pytest.approx(40.0)
    assert inv.invProb.beta == pytest.approx(40.0)    # 80/8 = 10 would breach the floor of 40


def test_target_falls_back_to_half_the_data_count_without_a_target_directive():
    d = MisfitDrivenBetaSchedule(verbose=False)
    inv = FakeInversion(target=50.0, beta0=100.0)
    inv.directiveList.dList = []                       # no TargetMisfit in the list
    d.inversion = inv
    # The fallback reads survey.nD through the data misfit, so give it one with two surveys.
    objfcts = [SimpleNamespace(simulation=SimpleNamespace(survey=SimpleNamespace(nD=n))) for n in (30, 70)]
    d._dmisfit = SimpleNamespace(objfcts=objfcts)
    assert d.target == 50.0


# ── Wiring: the instrument builds the right directive from its options ───────

def test_make_directives_selects_the_schedule_from_the_option():
    from test_gate_filter import stub, N_GATES_LM, N_GATES_HM   # pytest puts this directory on sys.path
    from SimPEG.electromagnetics.utils.static_instrument.schedules import MisfitDrivenBetaSchedule
    system = stub({"dbdt_ch1gt": N_GATES_LM, "dbdt_ch2gt": N_GATES_HM})
    fixed = system.make_directives()
    assert isinstance(fixed[1], BetaSchedule)                       # default is unchanged
    system.options.update(directives__beta__schedule="misfit",
                          directives__beta__progress_threshold=0.2,
                          directives__beta__stall_iterations=5,
                          directives__beta__stall_progress=0.02)
    chosen = system.make_directives()
    assert isinstance(chosen[1], MisfitDrivenBetaSchedule)
    assert chosen[1].progress_threshold == 0.2 and chosen[1].stall_iterations == 5 and chosen[1].stall_progress == 0.02
    assert isinstance(chosen[2], TargetMisfit)                      # still last of the three


# ── The fixed schedule is untouched ──────────────────────────────────────────

def test_fixed_schedule_still_halves_every_iteration():
    """Regression guard: selecting the stock schedule must behave exactly as before."""
    inv = FakeInversion(target=50.0, beta0=100.0)
    d = BetaSchedule(coolingFactor=2, coolingRate=1)
    d.inversion = inv
    for i in range(1, 4):
        inv.invProb.opt.iter = i
        d.endIter()
    assert inv.invProb.beta == pytest.approx(100.0 / 8)
