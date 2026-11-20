# One beta for a whole survey, with the misfit as QC

**State:** decided 2026-11-19 (the suggested answers), built in the same branch
**Ticket:** YmerFlow/ymerflow-business#182 (option A of three; B is a Workbench-style constraint mode, C is Marquardt damping separate from beta)
**Builds on:** #35 (every stop says why), the misfit-driven beta schedule

## Motivation

The stitched inversion lowers beta on every run until the data misfit reaches its target. Each line of a survey then ends at its own beta, and the strength of the smoothing has no meaning set beforehand. Aarhus Workbench works the other way round: the regularization strength is fixed before the run, and the data residual it ends with is read as QC.

On 64 line-by-line inversions of one 16-line helicopter TEM dataset (misfit schedule, `beta0_ratio` 10, `cooling_factor` 2, target misfit at chifact 1):

- final beta ranged from 0.12 to 1,523 between the lines of one run, and one line ended overfit at 0.45 of its target with beta 0.17;
- every run spent about 14 of its median 25 iterations above 50 times the target while beta halved down from about 1.2e6;
- two vertical length scales, 50 m and 75 m, ended at median betas about four times apart, so comparing them also compared two smoothing strengths. Where both happened to end at the same beta, the 75 m runs were 17-34% smoother vertically, as they should be.

Holding one beta across a survey is possible today only in roundabout ways, and three things get in the way:

1. The target-misfit stop always runs, at chifact 1, since neither chifact nor an off switch is an option. A line that reaches its target before beta gets down to the chosen value stops there.
2. There is no stop for "converged at this beta". With the target off, a run goes to `optimizer__max_iter`.
3. Under the misfit schedule with `cooling_factor` 1, a slow iteration is recorded as a cooling step although beta doesn't change, so after `stall_iterations` of them the stall rule stops the run with a message saying beta was being cooled.

## Design

### Holding beta

A new schedule value, `directives__beta__schedule = 'none'`: beta stays at its starting value for the whole run. The starting value is `directives__beta__beta0` when given, else the usual estimate (or the warm start's). Given as a number, `beta0` is the survey-wide beta.

An absolute floor, `directives__beta__beta_min`, for the `fixed` and `misfit` schedules: beta is cooled as now but never below it, and held there once reached. It takes the place of `beta_min_ratio` when given. This is "cool to one survey-wide beta and hold it", which keeps cooling as a continuation from a poor start model.

### The target-misfit stop

Two options: `directives__target__enable` (default True, as today) and `directives__target__chifact` (default 1, as today). With the target off, the misfit schedule still uses the target as today: it holds beta once phi_d is below it, and `ratio_cap` scales cooling by the distance to it. Nothing stops the run when phi_d reaches it.

### A stop on convergence at a held beta

`optimizer__rel_norm_change`, named after AarhusInv's `RelNormChange`: stop when the total objective, phi_d + beta phi_m, fell by less than this fraction from one iteration to the next. It is only checked between two iterations at the same beta, since a change of beta changes the objective by itself. Default None, off. The log line says which iteration, the two objective values and the fraction. With IRLS on, the run refuses it: IRLS changes the regularization weights at a held beta and has its own stop rules.

### The stall count at a cooling factor of 1

A step that leaves beta where it was counts as a hold, not a cooling step, so the stall rule doesn't fire on a run whose beta never changes. A step that sits on the floor still counts, as now: there beta can't go lower, so a stall is a real stall. The message then says beta sat on its floor.

### Reporting

At the end of every run, one more log line (`##MISFIT##`) gives the final phi_d against the target and the RMS misfit over the data in use, so the misfit reads as QC. `final_beta`, `resdata`, `restotal` and `numdata` are written as now.

## Decisions

| # | Question | Options | Suggested |
|---|---|---|---|
| 1 | How to hold one beta | A: a new schedule `none`. B: `fixed` with `cooling_factor` 1, which works today. | A: it says what it does, and `fixed` with factor 1 reads as a cooling schedule. |
| 2 | Cool to a beta, then hold it | A: an absolute floor `beta_min` in both schedules. B: keep `beta_min_ratio` only, so the floor is `beta_min_ratio` times the estimate. | A: an absolute value carries over between lines and runs; a ratio of an estimate doesn't. |
| 3 | The target-misfit stop | A: `target__enable` and `target__chifact`. B: `target__chifact` only. | A: an off switch is clearer than a chifact of 0, and chifact is useful on its own. |
| 4 | The convergence stop | A: relative change of the total objective at a held beta. B: relative change of phi_d only. | A: it is what AarhusInv tests, and phi_d alone can keep falling while the model only roughens. |
| 5 | Its default | A: off (None), with 0.01 suggested in the description. B: on at 0.01 whenever the target is off. | A: no change to any run that doesn't ask for it. |
| 6 | Stall at a factor of 1 | A: a step that leaves beta unchanged counts as a hold. B: keep the count, change the message. | A: a run at a held beta should end on the convergence stop or `max_iter`. |

## Tests

- Schedule `none`: beta is the same at every iteration, with `beta0` given and with it estimated.
- `beta_min`: both schedules cool to it and hold; it overrides `beta_min_ratio`.
- `target__enable` False: a run goes past the target; `target__chifact` scales the target and the log line.
- `optimizer__rel_norm_change`: stops when the objective change at a held beta falls below it, never across a change of beta, and names the reason.
- Misfit schedule at `cooling_factor` 1: no stall stop, and the iteration is recorded as a hold.
- Defaults: a run that sets none of the new options takes the same directives and stops as before.

## Work

- simpeg: the options and their wiring in `static_instrument/base.py` (`make_directives`, `make_optimizer`), the `none` schedule, `beta_min` and the stall count in `schedules.py`, a small directive for the convergence stop, and tests.
- aem-processes: nothing beyond the regenerated option schema, since invert_tem's options come from the system class.
- Platform: an environment build, once approved.

Rough size: half a day to a day in simpeg.
