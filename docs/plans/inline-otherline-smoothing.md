# Lateral smoothing along a line and between lines

**State:** decided 2026-10-31; schema shape revised 2026-11-02 (group-object `x-format`, see section 2); decision 1 settled 2026-11-03; decision 3 changed 2026-11-03, with a fallback 2026-11-05; decision 8 settled 2026-11-10
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

The lateral term becomes the sum over links of alpha_c (dm / length)^2, distance-weighted as in #28, with alpha_c depending on the link's kind. Each weight has an alpha and a length scale, and either can be set:

| Alpha | Length scale [CRS linear unit] | Relation |
|---|---|---|
| `regularization__alpha_r_inline` | `regularization__L_inline` | alpha = alpha_s L^2 |
| `regularization__alpha_r_otherline` | `regularization__L_otherline` | alpha = alpha_s L^2 |

The same pairing is added for the existing weights: `regularization__L_z` with `alpha_z`, and `regularization__L_r` with `alpha_r` for #28's single lateral scale (decision 6).

How a run resolves a pair:

- a length given sets its alpha;
- an alpha given alone sets its length;
- both given and differing by more than one part in a million stops the run, naming both values and the alpha they disagree on;
- defaults never count as given (decision 1): with neither set, the default length holds and the alpha follows `alpha_s`, and of a pair submitted together, one left at its default gives way to the other, so a form that sends every field does not trip the disagreement check.

When `alpha_s` changes, the lengths are held and the alphas recomputed, since the length scales are the geological choice (decision 7). The run report prints every alpha with its length, in meters or feet.

In the platform's process editor the alphas and lengths are linked: editing a length rewrites its alpha, editing an alpha rewrites its length, and changing `alpha_s` rewrites the alphas, all using the form's current `alpha_s`. The relation is always alpha = alpha_s L^2.

The schema carries this as a single `x-format` annotation on the enclosing **group object**, the way every other custom widget in Ymerflow is declared (a leaf's `x-format` picks its widget in `CustomStringField` - `dataset`, `upload`, `datasetPath` - and an `anyOf`'s picks `ExpressionField`). The smoothness parameters already form a group object: swaggerspect turns the flattened `regularization__*` attributes into a `regularization` object (`deps/swaggerspect/swaggerspect/__init__.py`, `_group_parameters`). That object carries the format and, beside it, which of its own members pair up (decision 8):

```json
"regularization": {
  "type": "object",
  "x-format": "length-scale",
  "x-length-scales": {
    "alpha_s": "alpha_s",
    "pairs": {"L_r": "alpha_r", "L_z": "alpha_z",
              "L_inline": "alpha_r_inline", "L_otherline": "alpha_r_otherline"}
  },
  "properties": {
    "mref":              {"type": "string", "enum": ["auto", "startmodel", "halfspace"]},
    "alpha_s":           {"type": "number"},
    "alpha_r":           {"type": "number"},
    "L_r":               {"type": "number"},
    "alpha_z":           {"type": "number"},
    "L_z":               {"type": "number"},
    "alpha_r_inline":    {"type": "number"},
    "L_inline":          {"type": "number"},
    "alpha_r_otherline": {"type": "number"},
    "L_otherline":       {"type": "number"}
  }
}
```

The widget owns the whole group: it renders every member, the pairs together and the rest (`mref`) as the form would anyway, and keeps the pairs in step. The pairs are named on the group, by the node the widget owns, so no field refers to another and nothing outside the group is touched. On the frontend this is one object-level dispatch (`if (schema['x-format'] === 'length-scale')`) returning a self-contained widget, and any other object goes to the default object field unchanged, mirroring `CustomStringField`. It does **not** wrap the generic `ObjectField` or inspect sibling keys.

Enabling work: `_group_parameters` synthesizes the `regularization` object itself and today preserves annotations only on leaf fields, not on the group it builds, which is why a per-leaf annotation was the only shape that survived. A class gives the group its keys by declaring the group under its own name, as an annotation with no value (decision 8):

```python
regularization: typing.Annotated[typing.Any, {"json_schema": {
    "x-format": "length-scale",
    "x-length-scales": {"alpha_s": "alpha_s", "pairs": {...}}}}]
"Smallness and smoothness weights, each smoothness weight with its length scale."
```

swaggerspect already reads such an annotation (`swaggerspect.JsonSchema` is the same thing); what it lacks is the merge. Today a declared `regularization` stops the grouping with a `KeyError` when it comes before its members and replaces the group, members and all, when it comes after. The extension merges the declared keys into the group it builds, in either order, and keeps the group's own `type`, `properties` and `x-python-type: null`, the marker by which `GroupMergingValidator` flattens the submitted group back to `regularization__*`. The attribute's docstring becomes the group's description, which today is just "regularization". The declaration has no value, so the class gains no attribute and a run receives exactly the parameters it does now; a class without one gets today's schema. The length scales (`L_*`) are members of this group and submitted, beside the alphas (decision 1).

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

### 3. When the split applies

- Always, by default (decision 3, changed 2026-11-03): `L_inline` and `L_otherline` both default to 500 m, so every run splits the links.
- Data without a `Line` column, or without the flight order within each line (an along-line distance, `xdist`, or a time, `timestamp`), fall back to #28's Delaunay links with one `alpha_r` (`L_r` 500 m). The run log says which column was missing, and the outputs' `model_info` records `lateral_links` as `inline_otherline` or `triangulation`. Data normalized by libaarhusxyz always have both columns.
- Both weights need distance weighting, since without it they are per-link weights with no length. Since 2026-11-03 distance weighting is always on (ymerflow-business#119), so `regularization__distance_weighted` and this check are gone.

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

| # | Question | Options | Decision |
|---|---|---|---|
| 1 | Parameter form (source of truth) | A: alphas only in the schema, lengths derived in the widget and only printed in the report. B: lengths only. C: both `alpha_*` and `L_*` are schema members and submitted, each updating the other, backend resolves pairs. | **C, settled 2026-11-03** (ymerflow-business#119). Was C (2026-10-31), reopened 2026-11-02 because the group-widget convention makes A clean; C keeps lengths usable by non-form callers (MCP/CLI). Every alpha and its length carry defaults that give the same numbers at the default `alpha_s` 1e-4: `L_r`, `L_inline`, `L_otherline` 500 m with alpha 25, `L_z` 50 m with `alpha_z` 0.25. A default never counts as given: the member set wins and the other follows from alpha = alpha_s L^2 (section 2). |
| 2 | Same-line Delaunay chords | A: leave them out; the inline chain replaces them. B: keep them as inline links. | A. Chords tie a curving line's soundings across the bend. |
| 3 | Behavior with both new options unset | A: #28 unchanged. B: always use the inline chain and otherline links. | **B, changed 2026-11-03** (ymerflow-business#119): the split is on by default, both kinds of link at 500 m, both pairs exposed with the same defaults. Was A. Data without a line column or flight order fall back to #28's links with `alpha_r` (settled 2026-11-05). Reproducing an earlier run is the environment's job: it pins the simpeg version. |
| 4 | Distance weighting off | A: error if either new option is set. B: allow as per-link weights. | A. Without distance the two weights have no length meaning. Moot since 2026-11-03: distance weighting is always on (ymerflow-business#119). |
| 5 | Test runs | The four above, or another set. | The four above. |
| 6 | Length fields for the existing weights | A: add `L_z` and `L_r` beside `alpha_z` and `alpha_r`, linked the same way. B: only the two new weights get lengths. | A, so every smoothness setting reads the same way. |
| 7 | What `alpha_s` changes | A: hold the lengths, recompute the alphas. B: hold the alphas, recompute the lengths. | A. The lengths are the geological choice; `alpha_s` sets how strongly the model is held to the reference. |
| 8 | How a flattened group declares a group-level `x-format`, and how the widget knows the pairs | Declaring: A: the class declares the group under its own name, an annotation with no value, and `_group_parameters` merges its keys into the group it builds (the review's marker, without a value, so nothing changes at run time). B: a class-level table, e.g. `__swaggerspect_groups__ = {"regularization": {...}}`, read by swaggerspect's class API and passed through to the grouping. C: one member carries keys for its parent, lifted by `_group_parameters`. Pairing: P: the group names its pairs (`x-length-scales`). N: the widget pairs by name (`L_x` with `alpha_x` or `alpha_r_x`). | **A with P, settled 2026-11-10** (proposed 2026-11-07). A reuses the annotation swaggerspect already reads and gives the group a real description; B is a second mechanism beside annotations; C has a field speak for its group, which is what the review on Ymerflow #138 asked to avoid. P keeps simpeg's parameter names out of the frontend; N would tie the widget to them. Was open since 2026-11-02. |

## Work

- `simpeg/regularization/regularization_mesh_lateral.py`: build the lateral operator from an explicit edge list with a weight per edge (the inline chain plus otherline Delaunay edges), alongside #28's simplex-mesh path.
- `simpeg/electromagnetics/utils/static_instrument/base.py`: the new alphas (and, if decision 1 lands on C, the lengths, resolving each pair as in section 2 with the section 3 checks), the link classification from the `Line` column and along-line distance, every alpha with its length in the run report, and declaring the `regularization` group with `x-format: "length-scale"` and its pairs (decision 8). The per-length `x-length-scale` annotation (`length_scale_of` in `smoothness.py`) goes.
- `deps/swaggerspect`: extend `_group_parameters` to merge a declared group's keys into the group it builds, in either order, keeping its `properties` and `x-python-type: null` (decision 8); today a declared group either stops the grouping or replaces its members.
- Ymerflow, process editor forms: a `CustomObjectField` whose only job is the dispatch: `x-format === 'length-scale'` returns a self-contained length-scale widget, anything else goes to the default object field unchanged, matching how `CustomStringField` dispatches `dataset`/`upload`. The widget renders the group's members, the pairs named in `x-length-scales` together, and keeps them in sync internally. It must **not** wrap the generic `ObjectField`'s `onChange` or inspect sibling keys (this is the PR #138 correction).
- Tests: link classification on two parallel lines and on a line with a tie crossing (the crossing must not break either line's chain), chords left out on a curving line, the two weights applied to the right links, `alpha_r_otherline = 0` giving the same model as inverting the lines separately with the same β, #28 behavior when both options are unset, and each way of resolving a pair, including the error when both are given and disagree. In Ymerflow, a form test for both directions of the link and for an `alpha_s` change.

Rough size: two days in simpeg on top of #28, and half a day in Ymerflow.
