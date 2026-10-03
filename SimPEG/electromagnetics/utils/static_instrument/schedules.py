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
    * **Floor.** Beta never drops below ``beta_min_ratio`` times the starting
      beta. A plateau on the floor is a stall, not silent continuation.
    * **Stall stop.** If the data misfit has improved by less than
      ``stall_progress`` (a fraction) over the last ``stall_iterations``
      iterations while cooling was being applied, the inversion stops with the
      reason in :attr:`stopped_reason` and in the log: cooling is no longer
      buying misfit, so the data cannot be fit to the target. A window rather
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
        the same list, or half the data count when there is none."""
        for directive in self.inversion.directiveList.dList:
            if isinstance(directive, TargetMisfit):
                return directive.target
        return 0.5 * sum(survey.nD for survey in self.survey)

    @property
    def beta_floor(self):
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
            action = "cool"

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
                self.stopped_reason = (
                    "the data misfit improved only %.1f%% over the last %d iterations while beta was cooled to %.3g "
                    "(phi_d=%.4g, target=%.4g); the data cannot be fit to the target at this noise level or discretization"
                    % (100 * window_progress, self.stall_iterations, new_beta, phi_d, target))
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


def _is_positive(value):
    try:
        return value is not None and float(value) > 0 and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
