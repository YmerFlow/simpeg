# Warm start for the spatially constrained inversion

**State:** decided 2026-10-31, ready to build
**Labels:** enhancement, inversion

## Motivation

A spatially constrained inversion (SCI) of a whole survey is the most expensive run we make: every sounding in one Gauss-Newton system, started from a flat halfspace at `startmodel__res` or from the clustered start model (`docs/plans/clustered-start-model.md`). By the time a survey gets an SCI it usually already has a laterally constrained inversion (LCI) of each line, and those models are close to the SCI's answer. Starting the SCI from them should cut the number of Gauss-Newton iterations, which is where the time goes. The test at the end measures how much.

## What exists

- `make_startmodel(thicknesses)` returns one log-conductivity block per sounding. The clustered start already uses it to give each sounding its own start.
- `make_mref(thicknesses)` follows `regularization__mref`, whose default `'startmodel'` makes the reference model the start model. The clustered-start plan chose that on purpose, to regularize toward the cluster solution.
- β starts at `directives__beta__beta0_ratio` (default 10) times the eigenvalue estimate. `directives__beta__schedule = 'misfit'` cools only when the misfit stops improving.
- `make_regularization` links each sounding to its Delaunay neighbors: its two along-line neighbors and the nearest few on each neighboring line, about six in all, as in the original SCI formulation. The coupling is local, and the work per iteration grows roughly linearly with the number of soundings.

## Design

### 1. Start from earlier models

A new option, `startmodel__dataset`: one or more XYZ models, the outputs of earlier inversions, for example one LCI per line. Each sounding being inverted takes its start from them:

1. the model sounding with the same `Line` and `sounding_id`, when both carry ids (Ymerflow plan "Sounding links and downline distance");
2. otherwise, the nearest model sounding by position, within `startmodel__fill_distance` (CRS linear unit; default 1.5 times the median line spacing, so a line whose LCI failed is filled from its neighbors);
3. otherwise, the existing start: the clustered start if `clustering__enabled`, else the halfspace at `startmodel__res` [ohm.m].

There is no interpolation between model soundings. It is a start model, and the SCI moves it; where a gap is filled from two lines, the step halfway between them is smoothed out in the first iterations. The run report counts how many soundings took each path.

The models' layers must match the inversion's. A mismatch stops the run with an error that names both layerings, unless `startmodel__resample_layers = True`, which resamples log10 resistivity onto the inversion's layers for a deliberate change of layering.

### 2. Keep the reference model separate

With `startmodel__dataset` set, `regularization__mref` defaults to `'halfspace'`. With `'startmodel'` the smallness term would pull the SCI back toward the line-by-line models it started from, including the line-to-line differences the SCI is there to remove. Setting `regularization__mref = 'startmodel'` explicitly still works. The clustered start keeps its current default.

### 3. Starting β from the recorded β

Every inversion writes the β of its last iteration into its output's `model_info` as `final_beta`. A warm start reads it from its start models (the median, when there are several) and starts at `directives__beta__warm_factor` times that value; the `misfit` schedule takes over from there.

The factor is there to leave the model room to change. At the LCI's own final β the data term dominates, the new links between lines carry little weight, and the SCI would keep the line-by-line differences; started higher, the regularization reshapes the model before β cools back down. Too high, and the first iterations smooth away the fit the warm start brought. The test below sets the factor's default.

An explicit `directives__beta__beta0` overrides all of this. If no start model records a β, the run falls back to `beta0_ratio` times the eigenvalue estimate and says so in the log.

### 4. Optional cross-line smoothing of the start

`startmodel__lateral_median` (CRS linear unit; default 0, off): before the run, each sounding's start is replaced, layer by layer, by the median log10 resistivity of the start soundings within that radius. It takes the line-to-line differences out of the start, so the SCI doesn't begin inside them.

### 5. On the platform

`invert_tem` in aem-processes exposes `startmodel__dataset` as a dataset field, so the process editor shows the dataset picker. Step inputs are fetched before the run (aem-processes#6).

## Test: cold against warm on one block

An eight-line block whose SCI has already run cold, on 8 cores, from its minimum-gates output.

1. An LCI of each of the eight lines, on the same input.
2. Three SCIs warm-started from those LCIs, with the halfspace reference and `warm_factor` 1, 3 and 10, every other setting as in the cold run.
3. Optionally, the best of step 2 again with cross-line smoothing.

Compared against the cold run: Gauss-Newton iterations; wall time of the LCIs plus each warm SCI; final misfit; the model difference in log10 resistivity above the DOI; line-to-line differences between neighboring lines. The best factor becomes the default for `warm_factor`. All runs go on the default cluster, and each job's parameters are agreed before it is submitted. The result decides whether the full survey's SCI is warm-started.

## Not in this plan

- A reference model from its own dataset, independent of the start: a preconceived structure, for example from boreholes or a geological model, as the target of the smallness term. Planned separately.
- A maximum link length, dropping Delaunay links across survey gaps, around block edges and between widely spaced lines (on by default), together with splitting a survey into overlapping tiles run as separate SCIs in parallel. That gets its own plan, building on the distance weighting in #28.
- Changes to the clustered start, apart from the reference default when a dataset start is given.

## Decisions (2026-10-31)

| # | Question | Decision |
|---|---|---|
| 1 | The models' layers differ from the inversion's | Stop with an error, unless `startmodel__resample_layers` asks for resampling. |
| 2 | Reference model when starting from earlier models | The halfspace. A reference model from its own dataset is planned separately. |
| 3 | Starting β | Every inversion records `final_beta`; a warm start begins at `warm_factor` times the recorded value, with the default factor set by the test (1, 3 or 10). `beta0` overrides. |
| 4 | Soundings the models don't cover | The nearest model sounding within the fill distance, then the clustered or halfspace start. No interpolation. |
| 5 | Maximum link length | Its own plan, with tiling, on by default there. |
| 6 | Matching before soundings carry ids | The nearest model sounding by position, which on a matched line is the same sounding. Sounding ids are used when present. |

## Work

- `simpeg/electromagnetics/utils/static_instrument/base.py`: the options above; `make_startmodel` reads the start models; `make_mref` defaults to the halfspace with a dataset start; the starting-β rule in the directives; `final_beta` written into the output's `model_info`.
- `simpeg/electromagnetics/utils/static_instrument/utils.py`: the matching helper, the layer check and resampling, the lateral median.
- Tests: each matching path, the layer-mismatch error and the resampling switch, the reference default, `final_beta` written and read, the median over several start models, the `beta0` override and the fallback, the lateral median.
- aem-processes: the dataset field for `invert_tem`, if the option is not picked up from the class as it stands.

Rough size: two days, plus the test runs.
