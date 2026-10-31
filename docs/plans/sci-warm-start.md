# Warm start for the spatially constrained inversion

**State:** proposed 2026-10-31, decisions open
**Labels:** enhancement, inversion

## Motivation

A spatially constrained inversion (SCI) of a whole survey is the most expensive run we make: every sounding in one Gauss-Newton system, started from a flat halfspace at `startmodel__res` or from the clustered start model (`docs/plans/clustered-start-model.md`). By the time a survey gets an SCI it usually already has a laterally constrained inversion (LCI) of each line, and those models are close to the SCI's answer. Starting the SCI from them should cut the number of Gauss-Newton iterations, which is where the time goes. The test at the end measures how much.

## What exists

- `make_startmodel(thicknesses)` returns one log-conductivity block per sounding. The clustered start already uses it to give each sounding its own start.
- `make_mref(thicknesses)` follows `regularization__mref`, whose default `'startmodel'` makes the reference model the start model. The clustered-start plan chose that on purpose, to regularize toward the cluster solution.
- β starts at `directives__beta__beta0_ratio` (default 10) times the eigenvalue estimate. `directives__beta__schedule = 'misfit'` cools only when the misfit stops improving.
- `make_regularization` links each sounding to its Delaunay neighbors, about six per sounding, so the lateral coupling is local and the work per iteration grows roughly linearly with the number of soundings. The lateral regularization mesh already accepts `active_edges` to leave links out.

## Design

### 1. Start from a model dataset

A new option, `startmodel__dataset`: an XYZ model, the output of an earlier inversion. Each sounding being inverted takes its start from that model, matched in this order:

1. the same `Line` and `sounding_id`, once soundings carry ids (Ymerflow plan "Sounding links and downline distance");
2. the same `Line` and the nearest position within `startmodel__match_distance` (CRS linear unit; default half the median sounding spacing);
3. inverse-distance interpolation of log10 resistivity, layer by layer, from the nearest model soundings within `startmodel__fill_distance` (CRS linear unit), for soundings on lines the model doesn't cover, such as ties or lines whose LCI failed;
4. the existing start: the clustered start if `clustering__enabled`, otherwise the halfspace at `startmodel__res` [ohm.m].

The run report counts how many soundings took each path.

The model's layers must match the inversion's. Layering is decided per survey, so a mismatch stops the run with an error that names both layerings (decision 1).

### 2. Keep the reference model separate

With `startmodel__dataset` set, `regularization__mref` defaults to `'halfspace'`. With `'startmodel'` the smallness term would pull the SCI back toward the line-by-line models it started from, including the line-to-line striping the SCI is there to remove. Setting `regularization__mref = 'startmodel'` explicitly still works. The clustered start keeps its current default.

### 3. Start β low

A warm start already fits the data, so a cold start's β, ten times the estimate, would first smooth away what the start model got right. A new option, `directives__beta__beta0`, sets an absolute starting β and overrides the estimate times `beta0_ratio` when given. The LCI logs show the β each line ended at, and the SCI starts near that, with the `misfit` schedule taking over (decision 3).

### 4. Optional cross-line smoothing of the start

`startmodel__lateral_median` (CRS linear unit; default 0, off): before the run, each sounding's start is replaced, layer by layer, by the median log10 resistivity of the start soundings within that radius. It takes the line-to-line striping out of the start, so the SCI doesn't begin inside it.

### 5. A maximum link length

`regularization__max_link_length` (CRS linear unit; default off): Delaunay links longer than this are left out through `active_edges`. Delaunay also joins soundings across survey gaps, around block edges and between widely spaced lines, and those links tie together ground that shouldn't be tied. With `regularization__distance_weighted` (#28) long links already count for less; this removes them. The speed effect is small, since the coupling is already local.

### 6. On the platform

`invert_tem` in aem-processes exposes `startmodel__dataset` as a dataset field, so the process editor shows the dataset picker. Step inputs are fetched before the run (aem-processes#6).

## Test: cold against warm on one block

An eight-line block whose SCI has already run cold, on 8 cores, from its minimum-gates output.

1. An LCI of each of the eight lines, on the same input.
2. An SCI warm-started from those LCIs, with the halfspace reference, β started from the LCI logs, and every other setting as in the cold run.
3. Optionally, step 2 again with cross-line smoothing.

Compared against the cold run: Gauss-Newton iterations; wall time of steps 1 and 2 together; final misfit; the model difference in log10 resistivity above the DOI; line-to-line differences between neighboring lines. All runs go on the default cluster, and each job's parameters are agreed before it is submitted. The result decides whether the full survey's SCI is warm-started.

## Not in this plan

- Splitting a survey into overlapping tiles run as separate SCIs in parallel, each keeping its central part. That is the change that would let a full-survey SCI run on smaller machines, and it gets its own plan once this test shows how quickly a warm-started block converges.
- Changes to the clustered start, apart from the reference default when a dataset start is given.

## Decisions

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | The model's layers differ from the inversion's | A: stop with an error. B: interpolate in depth, with a warning. | A. Layering is a per-survey choice, and a silent interpolation would hide a mismatch. |
| 2 | Reference model when starting from a dataset | A: halfspace. B: the start model, as for the clustered start. | A, for the reason in section 2. |
| 3 | Starting β | A: a new absolute `directives__beta__beta0`, set from the LCI logs. B: `beta0_ratio` near 1. | A, with B as the fallback when no LCI β is known. |
| 4 | Soundings the model doesn't cover | A: interpolate from neighbors, then fall back. B: fall back directly. | A. Ties and failed lines are the common case. |
| 5 | Maximum link length | A: in this plan, off by default. B: its own plan. | A. It is small and uses `active_edges`, which already exists. |
| 6 | Matching before soundings carry ids | A: same line, nearest position. B: wait for ids. | A. The id match replaces it when ids arrive. |

## Work

- `simpeg/electromagnetics/utils/static_instrument/base.py`: the options above; `make_startmodel` reads the dataset; `make_mref` defaults to the halfspace with a dataset start; the β override in the directives; `active_edges` from the maximum link length in `make_regularization`.
- `simpeg/electromagnetics/utils/static_instrument/utils.py`: the matching and interpolation helper, the lateral median.
- Tests: each matching path, the layer-mismatch error, the reference default, the β override, the lateral median, and links dropped by length.
- aem-processes: the dataset field for `invert_tem`, if the option is not picked up from the class as it stands.

Rough size: two days, plus the test runs.
