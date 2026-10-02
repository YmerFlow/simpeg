"""Beta schedules for the stitched 1D inversion.

The stock ``BetaSchedule`` divides beta by a fixed factor every iteration whether
or not the last step helped. Under it the number of Gauss-Newton iterations is
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
    """Cool beta only when the last iteration made progress, by an amount that
    depends on how far the data misfit still is from its target.

    Per Gauss-Newton iteration, once the misfit at the new model is known:

    * **Progress gate.** If the data misfit fell by less than
      ``progress_threshold`` (a fraction of the previous misfit), hold beta and
      count a stall. Otherwise cool.
    * **Misfit-ratio cooling.** Divide beta by
      ``cooling_factor * min(ratio_cap, phi_d / target)``. Far from the target
      the schedule cools fast; near it, by the base factor alone.
    * **Floor.** Beta never drops below ``beta_min_ratio`` times the starting
      beta. Reaching the floor while the misfit is still above target is a
      stall, not silent continuation.
    * **Stall stop.** ``stall_iterations`` consecutive holds stop the inversion
      with the reason recorded in :attr:`stopped_reason` and printed.

    Reaching the target is still ``TargetMisfit``'s job; place this directive
    before it in the list, where ``BetaSchedule`` used to be.

    ``history`` holds one record per iteration (iteration, phi_d, beta before,
    beta after, action) so a run can be inspected afterwards.
    """

    progress_threshold = 0.10
    cooling_factor = 2.0
    ratio_cap = 4.0
    beta_min_ratio = 1e-8
    stall_iterations = 3
    verbose = True

    def initialize(self):
        # BetaEstimate_ByEig runs before this directive in the list and has set
        # the starting beta by now. If this directive is used without an
        # estimator, the first endIter picks the starting beta up instead.
        beta = getattr(self.invProb, "beta", None)
        self._beta0 = beta if _is_positive(beta) else None
        self._phi_d_prev = None
        self._stalls = 0
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
            self._record(phi_d, beta, beta, "target")
            self._phi_d_prev = phi_d
            return

        prev = self._phi_d_prev
        if prev is None:
            prev = getattr(self.invProb, "phi_d_last", np.nan)
        if _is_positive(prev) and math.isfinite(prev):
            progress = (prev - phi_d) / prev
        else:
            progress = math.inf  # first iteration: nothing to compare against

        if progress > self.progress_threshold:
            factor = self.cooling_factor * min(self.ratio_cap, phi_d / target)
            new_beta = max(beta / factor, self.beta_floor)
            if new_beta >= beta and beta <= self.beta_floor:
                # Already on the floor and still above target: cooling cannot
                # help any more, so this counts as a stall.
                self._stalls += 1
                action = "floor"
            else:
                self._stalls = 0
                action = "cool"
            self.invProb.beta = new_beta
        else:
            new_beta = beta
            self._stalls += 1
            action = "hold"

        self._record(phi_d, beta, new_beta, action)
        if self.verbose:
            print(
                "MisfitDrivenBetaSchedule it=%d phi_d=%.4g target=%.4g progress=%s beta %.3g -> %.3g (%s)"
                % (self.opt.iter, phi_d, target,
                   "n/a" if progress == math.inf else "%.3f" % progress,
                   beta, new_beta, action))

        if self._stalls >= self.stall_iterations:
            self.stopped_reason = (
                "no progress on the data misfit for %d iterations (phi_d=%.4g, target=%.4g, beta=%.3g); "
                "the data cannot be fit to the target at this noise level or discretization"
                % (self._stalls, phi_d, target, new_beta))
            self.opt.stopNextIteration = True
            print("MisfitDrivenBetaSchedule stopping: " + self.stopped_reason)

        self._phi_d_prev = phi_d

    def _record(self, phi_d, beta_before, beta_after, action):
        self.history.append({
            "iteration": int(getattr(self.opt, "iter", len(self.history))),
            "phi_d": float(phi_d),
            "beta_before": float(beta_before),
            "beta_after": float(beta_after),
            "action": action,
        })


def _is_positive(value):
    try:
        return value is not None and float(value) > 0 and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
