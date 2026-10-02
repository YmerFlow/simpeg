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


def drive(directive, misfits, beta0=100.0, target=50.0):
    """Run the directive over a misfit trajectory; returns the fake inversion."""
    inv = FakeInversion(target, beta0)
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


# ── Behaviour ────────────────────────────────────────────────────────────────

def test_cools_faster_far_from_target_and_slower_near_it():
    d = MisfitDrivenBetaSchedule(cooling_factor=2.0, ratio_cap=4.0, verbose=False)
    # Far from target (phi_d/target = 8, capped at 4): factor 8. Near target (ratio 1.2): factor 2.4.
    inv = drive(d, misfits=[400.0, 60.0], beta0=100.0, target=50.0)
    before, after = [(h["beta_before"], h["beta_after"]) for h in d.history]
    assert before[0] == 100.0 and after[0] == pytest.approx(100.0 / 8)
    assert after[1] == pytest.approx(after[0] / (2.0 * 1.2))


def test_holds_beta_when_progress_is_below_threshold_and_stops_after_n_stalls():
    d = MisfitDrivenBetaSchedule(progress_threshold=0.10, stall_iterations=3, verbose=False)
    # 400 -> 395 -> 392 -> 390: each step improves by <10%, so three holds then stop.
    inv = drive(d, misfits=[400.0, 395.0, 392.0, 390.0, 388.0], beta0=100.0, target=50.0)
    actions = [h["action"] for h in d.history]
    assert actions[0] == "cool"                       # first iteration always cools
    assert actions[1:4] == ["hold", "hold", "hold"]
    assert inv.invProb.opt.stopNextIteration is True
    assert "no progress" in d.stopped_reason
    assert len(d.history) == 4                        # the run stopped; the 5th misfit was never seen
    assert inv.invProb.beta == d.history[0]["beta_after"]   # beta untouched during the holds


def test_progress_resets_the_stall_counter():
    d = MisfitDrivenBetaSchedule(progress_threshold=0.10, stall_iterations=3, verbose=False)
    inv = drive(d, misfits=[400.0, 395.0, 392.0, 200.0, 198.0, 196.0], beta0=100.0, target=50.0)
    actions = [h["action"] for h in d.history]
    assert actions == ["cool", "hold", "hold", "cool", "hold", "hold"]
    assert inv.invProb.opt.stopNextIteration is False


def test_floor_is_honoured_and_counts_as_a_stall_once_reached():
    d = MisfitDrivenBetaSchedule(cooling_factor=10.0, ratio_cap=4.0, beta_min_ratio=1e-3,
                                 stall_iterations=2, verbose=False)
    # Huge misfit every iteration: cooling by 40x each time hits the floor (0.1) on the 2nd iteration.
    inv = drive(d, misfits=[4000.0, 3000.0, 2000.0, 1000.0, 500.0], beta0=100.0, target=50.0)
    betas = [h["beta_after"] for h in d.history]
    assert betas[0] == pytest.approx(2.5)
    assert betas[1] == pytest.approx(0.1)             # clamped to the floor, not 2.5/40
    assert min(betas) >= 0.1 - 1e-12
    assert d.history[2]["action"] == "floor" and d.history[3]["action"] == "floor"
    assert inv.invProb.opt.stopNextIteration is True  # two floor-stalls at stall_iterations=2
    assert "cannot be fit" in d.stopped_reason


def test_does_not_cool_once_target_is_reached():
    d = MisfitDrivenBetaSchedule(verbose=False)
    inv = drive(d, misfits=[400.0, 45.0], beta0=100.0, target=50.0)
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
    inv.invProb.phi_d = 400.0
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
                          directives__beta__stall_iterations=5)
    chosen = system.make_directives()
    assert isinstance(chosen[1], MisfitDrivenBetaSchedule)
    assert chosen[1].progress_threshold == 0.2 and chosen[1].stall_iterations == 5
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
