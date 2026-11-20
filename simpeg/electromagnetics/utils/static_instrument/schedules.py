"""Beta schedules for the stitched 1D inversion.

The stock ``BetaSchedule`` divides beta by a fixed factor every iteration whether
or not the optimizer had finished at the previous one. Under it the number of Gauss-Newton iterations is
simply how many divisions it takes to reach the target misfit, and it fails
differently on every survey: a stall on step size with the misfit still above
target, a run to the iteration cap because beta keeps cooling with no floor, or
an over-fit collapse to vanishingly small beta.

:class:`MisfitDrivenBetaSchedule` is a drop-in replacement that reads the data
misfit before deciding. It depends only on the directive hooks (``initialize``,
``endIter``), ``invProb.beta``, ``invProb.phi_d`` and the ``TargetMisfit``
directive in the same list, which are identical across the SimPEG lineages this
code moves between, so the module is portable as-is.
"""

import math

import numpy as np

from ....directives import InversionDirective, TargetMisfit


class MisfitDrivenBetaSchedule(InversionDirective):
    """Cool beta when the Gauss-Newton iterations have stopped improving the
    data misfit at the current beta, by an amount that depends on how far the
    misfit still is from its target.

    This is the cooling rule of Oldenburg and Li's Tikhonov inversions and of
    SimPEG's ``PGI_BetaAlphaSchedule``: beta is the trade-off the optimizer is
    currently solving for, so it is lowered once that problem has converged
    (the misfit plateaus), not on a clock. Per iteration, once the misfit at
    the new model is known:

    * **Still improving.** If the data misfit fell by more than
      ``progress_threshold`` (a fraction of the previous misfit), hold beta and
      let the optimizer finish at this level.
    * **Plateau.** Otherwise cool: divide beta by
      ``cooling_factor * min(ratio_cap, phi_d / target)``. Far from the target
      the schedule cools hard; near it, by the base factor alone.
    * **Floor.** Beta never drops below ``beta_min`` when given, else
      ``beta_min_ratio`` times the starting beta. A plateau on the floor is a
      stall, not silent continuation. A cooling factor of 1, which leaves beta
      where it was, counts as a hold, not a cooling step.
    * **Stall stop.** If the data misfit has improved by less than
      ``stall_progress`` (a fraction) over the last ``stall_iterations``
      iterations while cooling was being applied, the inversion stops with the
      reason in :attr:`stopped_reason` and in the log: cooling is no longer
      buying misfit, so either the data cannot be fit to the target or the
      optimizer is step-limited and a lower beta is not the remedy. A window rather
      than a per-iteration count, because slow, steady progress is still
      progress.

    Reaching the target is still ``TargetMisfit``'s job; place this directive
    before it in the list, where ``BetaSchedule`` used to be. A misfit that
    *rises* counts as a plateau.

    ``history`` holds one record per iteration (iteration, phi_d, progress,
    beta before, beta after, action) so a run can be inspected afterwards.
    """

    progress_threshold = 0.10
    cooling_factor = 2.0
    ratio_cap = 1.0          # >1 multiplies the cooling factor by min(ratio_cap, phi_d/target); 1 = off
    beta_min_ratio = 1e-8
    beta_min = None  # absolute floor; takes the place of beta_min_ratio when given
    stall_iterations = 6
    stall_progress = 0.05
    verbose = True

    def initialize(self):
        # BetaEstimate_ByEig runs before this directive in the list and has set
        # the starting beta by now. If this directive is used without an
        # estimator, the first endIter picks the starting beta up instead.
        beta = getattr(self.invProb, "beta", None)
        self._beta0 = beta if _is_positive(beta) else None
        self._phi_d_prev = None
        self._phi_d_trail = []      # misfits at the last stall_iterations+1 iterations
        self._cooled_in_window = 0  # cooling steps inside that window
        self.stopped_reason = None
        self.history = []

    @property
    def target(self):
        """The target data misfit, read from the ``TargetMisfit`` directive in
        the same list, or the data count when there is none (phi_d is the sum
        of squared weighted residuals, with no factor 1/2, as in TargetMisfit)."""
        for directive in self.inversion.directiveList.dList:
            if isinstance(directive, TargetMisfit):
                return directive.target
        return sum(survey.nD for survey in self.survey)

    @property
    def beta_floor(self):
        if self.beta_min is not None:
            return float(self.beta_min)
        return self._beta0 * self.beta_min_ratio if self._beta0 is not None else 0.0

    def endIter(self):
        beta = self.invProb.beta
        if self._beta0 is None and _is_positive(beta):
            self._beta0 = beta
        phi_d = float(self.invProb.phi_d)
        target = float(self.target)

        if phi_d <= target:
            # TargetMisfit stops the run on this same iteration. Nothing to cool.
            self._record(phi_d, math.nan, beta, beta, "target")
            print("MisfitDrivenBetaSchedule it=%d phi_d=%.4g target=%.4g: target reached, beta held at %.3g"
                  % (self.opt.iter, phi_d, target, beta))
            self._phi_d_prev = phi_d
            return

        prev = self._phi_d_prev
        if prev is None:
            prev = getattr(self.invProb, "phi_d_last", np.nan)   # the misfit at the start model
        if _is_positive(prev) and math.isfinite(prev):
            progress = (prev - phi_d) / prev
        else:
            progress = math.inf   # nothing to compare against: treat as improving

        if progress > self.progress_threshold:
            new_beta = beta
            action = "hold"
        elif beta <= self.beta_floor:
            new_beta = beta
            action = "floor"
        else:
            factor = self.cooling_factor
            if self.ratio_cap > 1.0:
                factor *= min(self.ratio_cap, max(1.0, phi_d / target))
            new_beta = max(beta / factor, self.beta_floor)
            self.invProb.beta = new_beta
            # a factor of 1 leaves beta where it was: that is a hold, not a cooling step
            action = "cool" if new_beta < beta else "hold"

        self._record(phi_d, progress, beta, new_beta, action)
        if self.verbose:
            print(
                "MisfitDrivenBetaSchedule it=%d phi_d=%.4g target=%.4g progress=%s beta %.3g -> %.3g (%s)"
                % (self.opt.iter, phi_d, target,
                   "n/a" if progress == math.inf else "%.3f" % progress,
                   beta, new_beta, action))

        # Stall: over the last stall_iterations iterations the misfit barely moved although beta
        # was being cooled (or sat on its floor). Holds do not count toward the window.
        self._phi_d_trail.append(phi_d)
        if action in ("cool", "floor"):
            self._cooled_in_window += 1
        if len(self._phi_d_trail) > self.stall_iterations + 1:
            dropped = self._phi_d_trail.pop(0)
        if len(self._phi_d_trail) == self.stall_iterations + 1 and self._cooled_in_window >= self.stall_iterations:
            window_progress = (self._phi_d_trail[0] - phi_d) / self._phi_d_trail[0]
            if window_progress < self.stall_progress:
                where = (
                    "sat on its floor of %.3g"
                    if action == "floor"
                    else "was cooled to %.3g"
                ) % new_beta
                self.stopped_reason = (
                    "the data misfit improved only %.1f%% over the last %d iterations while beta %s "
                    "(phi_d=%.4g, target=%.4g). Cooling is not buying misfit: either the data cannot be fit to the "
                    "target at this noise level or discretization, or the optimizer is step-limited (watch the line-search "
                    "count); a better start model or step damping, not a lower beta, is the remedy"
                    % (
                        100 * window_progress,
                        self.stall_iterations,
                        where,
                        phi_d,
                        target,
                    )
                )
                self.opt.stopNextIteration = True
                print("MisfitDrivenBetaSchedule stopping: " + self.stopped_reason)
        if action == "hold":
            # a hold means the optimizer is still working at this beta: restart the window
            self._phi_d_trail = [phi_d]
            self._cooled_in_window = 0

        self._phi_d_prev = phi_d

    def _record(self, phi_d, progress, beta_before, beta_after, action):
        self.history.append({
            "iteration": int(getattr(self.opt, "iter", len(self.history))),
            "phi_d": float(phi_d),
            "progress": None if progress is None or not math.isfinite(progress) else float(progress),
            "beta_before": float(beta_before),
            "beta_after": float(beta_after),
            "action": action,
        })


class StartingBeta(InversionDirective):
    """Start beta at a given value instead of estimating it.

    Takes the place of ``BetaEstimate_ByEig`` at the head of the directive list, so a
    schedule after it reads this value as its starting beta.

    Parameters
    ----------
    beta0 : float
        The starting beta.
    """

    def __init__(self, beta0, **kwargs):
        super().__init__(**kwargs)
        self.beta0 = float(beta0)

    def initialize(self):
        self.invProb.beta = self.beta0


class BetaFloor(InversionDirective):
    """Keep beta at or above an absolute floor, for schedules that have none.

    Place it after the schedule. It raises a starting beta below the floor at
    ``initialize``, and after every iteration puts beta back on the floor if
    the schedule took it lower.

    Parameters
    ----------
    beta_min : float
        The floor on beta [dimensionless].
    """

    def __init__(self, beta_min, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = float(beta_min)

    def initialize(self):
        self._clamp()

    def endIter(self):
        self._clamp()

    def _clamp(self):
        """Put beta back on the floor if it is below it."""
        if self.invProb.beta < self.beta_min:
            self.invProb.beta = self.beta_min


class RelativeNormChangeStop(InversionDirective):
    """Stop when the objective function stops falling at a held beta.

    After each iteration the total objective, phi_d + beta * phi_m, is
    compared with the previous iteration's. If beta was the same for both and
    the objective fell by less than ``rel_change`` of its previous value, the
    run stops, with the reason in :attr:`stopped_reason` and in the log. This
    is AarhusInv's ``RelNormChange`` test on the total norm. Across a change of
    beta the objective changes by itself, so no test is made there.

    Place it before any directive that changes beta, so it reads the beta the
    iteration was solved with.

    Parameters
    ----------
    rel_change : float
        Smallest relative fall of the objective that keeps the run going
        [fraction, e.g. 0.01 for 1%].
    """

    def __init__(self, rel_change, **kwargs):
        super().__init__(**kwargs)
        self.rel_change = float(rel_change)

    def initialize(self):
        self._previous = None  # (beta, objective) at the last iteration
        self.stopped_reason = None

    def endIter(self):
        beta = float(self.invProb.beta)
        objective = float(self.invProb.phi_d) + beta * float(self.invProb.phi_m)
        previous, self._previous = self._previous, (beta, objective)
        if previous is None or previous[0] != beta or not _is_positive(previous[1]):
            return
        change = (previous[1] - objective) / previous[1]
        if change < self.rel_change:
            self.stopped_reason = (
                "the objective fell by %.3g%% at beta %.4g (from %.6g to %.6g), less than "
                "optimizer__rel_norm_change %.3g%%: converged at this beta"
                % (100 * change, beta, previous[1], objective, 100 * self.rel_change)
            )
            self.opt.stopNextIteration = True
            print(
                "RelativeNormChangeStop it=%d: %s"
                % (self.opt.iter, self.stopped_reason)
            )


class RecordBeta(InversionDirective):
    """Keep the beta the most recent Gauss-Newton iteration was solved with, as :attr:`beta`.

    Place it last in the directive list. Schedules lower beta in ``endIter``, after the
    iteration has been solved, so ``invProb.beta`` at the end of a run can already be the
    value for an iteration that never ran. This directive notes beta after every other
    directive has had its say, which is the value the next iteration uses.
    """

    beta = None

    def initialize(self):
        self.beta = None
        self._next = float(self.invProb.beta)

    def endIter(self):
        self.beta = self._next
        self._next = float(self.invProb.beta)


def _is_positive(value):
    try:
        return value is not None and float(value) > 0 and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class TargetMisfitInUse(TargetMisfit):
    """``TargetMisfit`` whose data count is the data in use.

    phi_d sums squared weighted residuals over the data with a nonzero weight;
    a gate with no datum or no error has weight 0. ``TargetMisfit`` counts every
    datum (``survey.nD``), so with empty gates it stopped a run at an RMS of
    sqrt(nD / in use) on the gates in use instead of 1 (ymerflow-business#132).
    The count is taken at ``initialize`` from the weights phi_d uses.

    With ``stop`` False the target is still there for the schedule to read,
    but reaching it doesn't stop the run; the log says so once.
    """

    stop = True
    _reached = False

    def initialize(self):
        dmisfits = getattr(self.invProb.dmisfit, "objfcts", [self.invProb.dmisfit])
        weights = np.concatenate([dmisfit.W.diagonal() for dmisfit in dmisfits])
        n_used = int(np.count_nonzero(weights))
        self.phi_d_star = n_used
        self._reached = False
        print(
            "TargetMisfit: target phi_d %.6g = chifact %g x %d data in use (%d of %d have weight 0)%s"
            % (
                self.target,
                self.chifact,
                n_used,
                weights.size - n_used,
                weights.size,
                "" if self.stop else "; reaching it does not stop the run",
            )
        )

    def endIter(self):
        if self.stop:
            return super().endIter()
        if self.invProb.phi_d < self.target and not self._reached:
            self._reached = True
            print(
                "TargetMisfit: phi_d %.6g < target %.6g at iteration %d; not stopping "
                "(directives__target__enable is False)"
                % (self.invProb.phi_d, self.target, self.opt.iter)
            )
