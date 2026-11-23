# Regularization set as Workbench constraints

**State:** decisions answered 2026-11-23; the form of `ConstraintByThickness` is being tested
**Ticket:** YmerFlow/ymerflow-business#182 (option B of three: A holds one beta for a survey, #45; C is Marquardt damping separate from beta)
**Builds on:** #45 (schedule `none`, the target switch, `optimizer__rel_norm_change`), #38 (always distance-weighted, links split along and between lines), #31

## Motivation

With option A a survey runs at one beta, but that beta still multiplies alphas, length scales and simpeg's cell-volume weighting, so a Workbench user can't carry a constraint setup over, and the weight each constraint ends up with depends on depth and line spacing in ways nobody chose. AarhusInv, which Workbench runs, has no beta: each vertical and lateral constraint is one equation with its own uncertainty, set beforehand, and the data residual it ends with is QC.

The weight one unit difference in ln(conductivity) gets per constraint, at beta 300 with alpha_s 1e-4, a lateral length of 200 m, L_z 75 m and 40 layers from 5 m to 750 m, against Workbench's Medium preset (both data misfits are sum((r / std)^2), so the weights compare directly):

| Constraint | Top layer (5 m) | Deepest layer (48 m) |
|---|---|---|
| along the line, soundings 30 m apart | 0.5 x Workbench | 4 x |
| vertical | 15 x | 1.7 x |
| between lines 2 km apart | 0.007 x | 0.06 x |
| pull toward the reference model | 3 x | 30 x |

The overall strength is in Workbench's range, but the balance differs: lines are 15 to 150 times more independent than Workbench would make them, because our weight falls with distance squared and Workbench's with a power of 0.75; and the cell-volume weighting makes deep layers stiffer along the line, looser vertically and much more strongly pulled to the reference. This mode gives each constraint Workbench's weight instead.

## How Workbench sets the constraints

From the saved inversion settings and CON files of a finished Workbench project, and the Workbench SkyTEM guide:

- Constraint factors are multiplicative: a factor C lets the parameter vary between its value divided or multiplied by C, so in the model's own log units the STD is log(C). The settings files store C - 1. A factor of 99 is described as disabling a constraint.
- Presets on resistivity: Medium is vertical 2.0 and lateral 1.3; Tight is vertical 1.5 and lateral 1.1.
- Lateral constraints scale with the distance d between two soundings: C(d) = 1 + (C_ref - 1) (d / d_ref)^n, never tighter than C_ref below d_ref. The project used d_ref 30 m and n 0.75. A smaller n gives relatively stronger constraints across lines.
- The resistivity prior is 98 (a factor of 99) toward the start model. AarhusInv treats it as a weak soft prior (about 0.05 per cell), enough to bring the model back to the reference where the data say little.
- `ConstraintByThickness = 1` is set in every file, and nothing on hand says what it does.
- The CON file stops on a relative change of the total norm of 0.007 (`RelNormChange`), or 50 iterations, and inverts log data.

## Design

### The mode and its parameters

`regularization__mode`, `'alphas'` (today, the default) or `'constraints'`. In `'constraints'`:

| Parameter | Default | Meaning |
|---|---|---|
| `regularization__vertical_factor` | 2.0 | Factor between adjacent layers of a sounding. |
| `regularization__lateral_factor` | 1.3 | Factor between neighboring soundings at the reference distance. |
| `regularization__otherline_factor` | None | Factor for the links between lines; None takes the lateral factor (SCI-style), 99 turns them off (LCI-style). |
| `regularization__reference_distance` | 30 | d_ref, in the CRS linear unit. With a CRS in feet, give it. |
| `regularization__distance_power` | 0.75 | n in C(d). |
| `regularization__prior_factor` | 99 | Factor of the prior toward the reference model, each cell. |

The alphas and length scales are ignored in this mode, and the run log says so if any were given.

### The weights

Each constraint is one row with weight 1 / sigma^2, sigma = ln(C), in the model's ln units:

- Lateral link of length d, each layer: sigma = ln(1 + (C - 1) max(d / d_ref, 1)^n), with C the lateral or otherline factor. The regularization mesh divides each lateral difference by d (#38), so the link weight passed to it is d^2 / sigma^2.
- Vertical, between layers k and k + 1: sigma = ln(C_v). The mesh divides by the distance between layer centers, dz, so the face weight is dz^2 / sigma^2.
- Prior, each cell: sigma = ln(C_p), toward the reference model.
- simpeg's cell-volume weights are removed from all three terms (`remove_weights("volume")`), and every alpha multiplier is 1.

### Beta and the stop

The mode's defaults, each overridable: schedule `none` with `beta0` 1, the target stop off, and `optimizer__rel_norm_change` 0.007, Workbench's value. A run then minimizes the Workbench-form objective and reports its misfit. Setting a schedule on top stays possible.

### What stays as it is

The data misfit (linear data, weighted by 1 / STD), the DOI, the outputs and the start model. IRLS is refused in this mode: AarhusInv's reweighting is its own scheme and out of scope here. Inverting the transmitter altitude, which Workbench does, is a separate plan.

### Reporting

The run log prints the factors and the resulting sigma at d_ref, at the median inline spacing and at the median line spacing, so the balance between directions is visible. Next to each lateral factor it prints the approximate length scale L, the distance at which one link may vary by a factor of 2: C(d) = 1 + (d / L)^n, so L = d_ref / (C - 1)^(1/n), about 150 m for 1.3 at 30 m with n 0.75, and about 650 m for 1.1. `model_info` records the mode, the factors and the lengths.

## Decisions

Answered 2026-11-23.

| # | Question | Options | Decided |
|---|---|---|---|
| 1 | How the user gives the constraints | A: factors as Workbench's dialog shows them (2.0, 1.3). B: STDs as its files store them (1.0, 0.3). C: sigma in ln units. D: a lateral length in m. | A, with the approximate lateral length scale shown next to each factor (see Reporting). |
| 2 | Links between lines | A: a separate otherline factor, defaulting to the lateral one (SCI-style), 99 for none (LCI-style). B: one lateral factor for all links. | A. |
| 3 | `ConstraintByThickness` | A: uniform vertical factor per interface now; add the thickness scaling once its form is known. B: guess a scaling now. | A. Workbench writes its MOD files to a temporary folder and deletes them, so the form is tested instead from a run's iteration file: the vertical, lateral and prior terms it reports, recomputed from the run's exported final model under each candidate rule. |
| 4 | Beta, schedule and stop in this mode | A: defaults (beta 1, `none`, target off, 0.007) that can be overridden. B: fixed, not overridable. | A. |
| 5 | Prior | A: per cell toward the reference model at `prior_factor` 99, no volume weighting. B: keep `alpha_s`. | A. |
| 6 | Data space | A: keep linear data with 1 / STD. B: invert log data, as AarhusInv does by default. | A. Log data space becomes an option for every run in its own change, YmerFlow/ymerflow-business#185. |
| 7 | IRLS | A: refused in this mode. B: allowed. | A. A blocky option (L1 on the constraints, as Workbench's blocky) on top of this mode is planned in YmerFlow/ymerflow-business#186. Sharp is out of scope. |

## Tests

- sigma(d) from the factors: at, below and above d_ref, for both link kinds, and 99 turning the otherline links off.
- The weights the regularization ends up with, read from its matrices as in the comparison above, equal 1 / sigma^2 per constraint, with no dependence on layer thickness.
- On two soundings of three layers, the objective at beta 1 equals the Workbench-form objective computed by hand.
- The mode's defaults, each override, and the IRLS refusal.
- A run in `'alphas'` mode builds the same regularization as before.

## Validation

Chosen: one HeliTEM line from a Workbench project whose settings, CON and iteration files are on hand, with its data, final model and forward response exported. It runs locally only, and its files and numbers stay out of every repository and ticket.

1. The forward first. #44 matched Workbench for SkyTEM; HeliTEM hasn't been checked. Our forward of Workbench's final model against Workbench's exported forward response, gate by gate. The front gate is still open.
2. The inversion: the same data, layers, start model, prior and constraints, inverted in this mode, against Workbench's model. Compare per cell within the DOI (median and 90th percentile of |log10(ours / Workbench)|), the data residual distributions, and the iteration counts against the run's iteration file.

## Work

- simpeg: the mode and its parameters in `static_instrument/base.py`, the weights in `make_regularization` (link and face weights, volume weights removed), the mode's defaults in `make_directives`, the log line, and tests.
- aem-processes: the regenerated option schema; grouping the constraint options in the form so the two modes don't show together is worth doing with YmerFlow/Ymerflow#138's group widget.
- Platform: an environment build once approved.

Rough size: 2-3 days in simpeg with tests, then 1-2 days of validation, plus about half a day for the HeliTEM forward check.
