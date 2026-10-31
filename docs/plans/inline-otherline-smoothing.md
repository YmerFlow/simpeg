# Lateral smoothing along a line and between lines

**State:** draft 2026-10-31, decisions open
**Labels:** enhancement, inversion
**Builds on:** #28 (distance-weighted lateral and vertical smoothness)

## Motivation

With #28's distance weighting, `alpha_r` becomes a length scale, L_r = sqrt(alpha_r / alpha_s), and one L_r applies to every Delaunay link. Airborne surveys are sampled very unevenly. On an eight-line block of a real survey:

- soundings are 26 m apart along a line;
- the nearest sounding on another line is a median 1.9 km away (10th percentile 1.66 km, 90th 2.17 km);
- Delaunay gives each sounding six links, two along its own line and about four to the neighboring lines, with a median length of 1.94 km; hull links reach 47 km.

For parallel lines a distance l apart, the pull a sounding feels from the neighboring lines, relative to the smallness term, is about 4 (L_r / l)^2: each interior sounding has about four links to other lines, each with stiffness alpha_r / l^2 against alpha_s. At l = 1.9 km that is about 1 % at L_r = 100 m and 4 % at 200 m, so the lines are effectively independent. Coupling them needs L_r of the order of l, which would also smooth along the line over kilometers. One length scale can't serve both directions.

Delaunay also makes a poor chain along a single line. On a curving line most of its links are chords rather than links between neighbors (#28's own measurement found only about a third of links along-line on a single line), and at a tie crossing it can route a line's links through the tie's soundings.

## Design

### 1. Two kinds of lateral link

- Inline links join consecutive soundings of the same line, in flight order (by along-line distance, falling back to time). They are built explicitly, not taken from the triangulation.
- Otherline links are the Delaunay edges whose two soundings have different `Line` values.
- Delaunay edges between two soundings of the same line that are not consecutive, the chords, are left out; the inline chain replaces them.

A link is classified only by whether its two soundings share a `Line` value. Whether the other line is a tie or a parallel line is never needed: a tie crossing is an otherline link that happens to be short, and distance weighting makes it strong. Crossing soundings are therefore held close to equal, which is right physically since they measured the same ground, and a level shift between two flights shows up as misfit concentrated at their crossings rather than being smoothed away.

### 2. Two weights and their length scales

The lateral term becomes the sum over links of alpha_c (dm / length)^2, distance-weighted as in #28, with alpha_c depending on the link's kind:

| Option | Length scale [CRS linear unit] |
|---|---|
| `regularization__alpha_r_inline` | L_inline = sqrt(alpha_r_inline / alpha_s) |
| `regularization__alpha_r_otherline` | L_otherline = sqrt(alpha_r_otherline / alpha_s) |

`alpha_s`, `alpha_z` and L_z are unchanged. Every docstring states the length scale, and the run report prints L_inline, L_otherline and L_z in meters (or feet) beside the alphas.

Alphas for a given length scale, at `alpha_s` = 1e-4, and the resulting pull from neighboring lines 1.9 km away:

| L [m] | alpha | Neighbor-line pull, otherline at l = 1.9 km |
|---|---|---|
| 25 | 0.0625 | |
| 50 | 0.25 | |
| 100 | 1 | 1 % |
| 150 | 2.25 | 2.5 % |
| 200 | 4 | 4 % |
| 500 | 25 | 28 % |
| 1000 | 100 | 1.1 times smallness |
| 2000 | 400 | 4.4 times smallness |

### 3. When the new options apply

- The split applies only when either new alpha is set. With both unset, #28's behavior is unchanged: Delaunay links and one `alpha_r`.
- With one set, the other defaults to `alpha_r`.
- Both need `regularization__distance_weighted = True`, since without it they are per-link weights with no length; setting either with distance weighting off raises an error that says so.
- The `Line` column is required; without it the run stops before inverting.

### 4. `alpha_r_otherline = 0`: lines inverted independently in one job

With no otherline weight, each line is regularized on its own, so one job inverts every line as an LCI. The lines still share one β schedule and one stopping rule, which separate per-line jobs do not: on the eight-line block, per-line LCIs ended at β from 0.9 to 7,000, while the joint run took every line to β of about 1. It is a cheap first pass over a survey, not a replacement for per-line LCIs.

### 5. β and earlier runs

The alphas set the size of the regularization term, so a β recorded under one set of alphas is not on the same scale under another. A warm start from LCIs (#30) uses the same alphas as the LCIs, or accepts that the recorded β is only an order of magnitude.

## Test on the eight-line block

Same input and every other setting as the cold SCI already run on the block, with distance weighting on. Four runs on one 127-core node, 30 cores each, submitted together:

| Run | L_inline | L_otherline | Compared with |
|---|---|---|---|
| 1 | 100 m | 0 | the eight per-line LCIs: the effect of the shared β alone |
| 2 | 100 m | 1 km | runs 1 and 4 |
| 3 | 100 m | 2 km | runs 1 and 4 |
| 4 | 200 m | 2 km | run 3: the effect of a longer along-line scale |

Measured for each: misfit per line, iterations, wall time, and the difference in log10 resistivity between neighboring lines at matching along-line positions, per depth, against the per-line LCIs. Each job's parameters are agreed before it is submitted. The block has no tie lines, so crossings are not tested here; that needs a dataset with flight lines and ties merged.

## Not in this plan

- A depth-dependent L_z. The shallow layers are 2 m thick, and a single L_z of 50 m spans more than a dozen of them.
- A maximum link length and tiling (planned separately). With otherline links weighted by length, the 47 km hull links are already very weak.
- Separate β schedules per line in one job.

## Decisions

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | Parameter form | A: alphas (`alpha_r_inline`, `alpha_r_otherline`), matching `alpha_s`, `alpha_r` and `alpha_z`, with the length scales printed. B: lengths in meters (`L_inline`, `L_otherline`). C: both, lengths overriding alphas. | Open. A keeps one convention; B is what people think in. |
| 2 | Same-line Delaunay chords | A: leave them out; the inline chain replaces them. B: keep them as inline links. | A. Chords tie a curving line's soundings across the bend. |
| 3 | Behavior with both new options unset | A: #28 unchanged. B: always use the inline chain and otherline links. | A, so existing runs reproduce. |
| 4 | Distance weighting off | A: error if either new option is set. B: allow as per-link weights. | A. Without distance the two weights have no length meaning. |
| 5 | Test runs | The four above, or another set. | The four above. |

## Work

- `simpeg/regularization/regularization_mesh_lateral.py`: build the lateral operator from an explicit edge list with a weight per edge (the inline chain plus otherline Delaunay edges), alongside #28's simplex-mesh path.
- `simpeg/electromagnetics/utils/static_instrument/base.py`: the two options, the checks in section 3, the link classification from the `Line` column and along-line distance, and the lengths in the run report.
- Tests: link classification on two parallel lines and on a line with a tie crossing (the crossing must not break either line's chain), chords left out on a curving line, the two weights applied to the right links, `alpha_r_otherline = 0` giving the same model as inverting the lines separately with the same β, and #28 behavior when both options are unset.

Rough size: a day and a half on top of #28.
