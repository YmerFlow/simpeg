# Check the DOI against the model base

**State:** proposed 2026-11-08, waiting for decisions
**Ticket:** YmerFlow/ymerflow-business#161
**Builds on:** #29 (the Christiansen-Auken DOI written with every inversion)

## Motivation

The model base, the top of the half-space, is a modelling choice. When it is too shallow for the system and the ground, the lower DOI piles up at the base and the deepest part of the section is cut off. Nothing in a run says so today; it is found by plotting the DOI against depth by hand.

On one helicopter TEM survey inverted with 40 layers and a 500 m base, the lower DOI sat at the base on 44% of soundings and the upper DOI on 21%, and the 90th percentile of the lower DOI was the base itself. With a 700 m base the same survey gave 1% and 0%. A survey flown with a system that reaches less deep, inverted with the 500 m layering, gave 0.1%. So a fixed layering can be right for one system and wrong for another, and the run is the place to say which.

## Design

### What is computed

For each model an inversion writes (both models with IRLS), per line and for the run as a whole:

- `lower_at_base` and `upper_at_base`: the share of soundings with a DOI whose DOI lies within half a bottom-layer thickness of the half-space top, `doi >= z_base - t_bottom / 2`, with `z_base` the half-space top and `t_bottom` the thickness of the layer above it. A DOI equal to `z_base` is the case where the half-space alone is above the threshold, so the true DOI is somewhere below the model.
- `lower_p90_of_base`: the 90th percentile of the lower DOI as a fraction of `z_base`, a headroom figure. Near 1 the base is about to clip, even if few soundings sit on it yet.
- `upper_in_first_layer` (optional, decision 5): the share of soundings whose upper DOI lies inside the first layer, which shows a first layer coarser than the earliest gates resolve.
- `n_soundings` and `n_with_doi`.

Soundings without a DOI (NaN: filtered out before the inversion, a DOI that failed, or a model read in without one) are left out of every share and only counted. They are never treated as depth 0 or as the base.

Lines come from the line column, as for the inline / otherline split. Without one, only the run figures are written.

### Where it lives

In simpeg, where the DOI is computed and the layering is known:

- `doi_utils.doi_base_check(doi_upper, doi_lower, depth_top, line=None, threshold=0.05)` returns the figures above as a plain dict. It needs numpy only, so aem-processes, or a notebook, can run it on any finished model that has DOI columns and layer tops, including models from elsewhere.
- `XYZSystem._add_doi` calls it right after the DOI, puts the dict on the model as `model_info["doi_check"]`, and writes two kinds of log line (below).

The alternative is `invert_tem` in aem-processes, computing it from the model it has just written. That would need the layering again from the written model, and local runs, which don't go through `invert_tem`, would not get the check. `invert_tem` can still call `doi_base_check` on any model, and the run summary proposed in ymerflow-business#153 (item 3) can copy `model_info["doi_check"]` when it is built.

### Output

On the model, `model_info["doi_check"]`:

```json
{"threshold": 0.05, "base_depth": 500.0, "bottom_layer": 23.4,
 "run": {"n_soundings": 1200, "n_with_doi": 1180, "lower_at_base": 0.44, "upper_at_base": 0.21,
         "lower_p90_of_base": 1.0, "upper_in_first_layer": 0.02},
 "lines": {"101": {"...": "the same keys"}, "...": {}},
 "lines_over_threshold": ["101", "102"]}
```

It travels with the model: it survives the unfilter to all soundings, the `alc` normalize `invert_tem` runs before writing, and the msgpack round trip (checked). In an Aarhus XYZ export it becomes a header comment (`/doi_check` and the dict on the next line), which XYZ readers skip (decision 4).

In the run log, one summary line per model:

```
##DOI_CHECK## lower DOI at the base on 44.0% of soundings (upper 21.0%), lower DOI p90 at 100% of the 500 m base; 3 of 8 lines over 5%
```

and, when any line is over the threshold:

```
WARNING: the model base (500 m) cuts off the lower DOI on more than 5% of soundings on these lines: 101, 102, ...; consider a deeper base
```

### Parameter

`doi__base_warning: float = 0.05`: the share of a line's soundings (with a DOI) whose lower DOI may sit at the model base before the log warns; 1 never warns. The warning uses the lower DOI only, since an upper DOI at the base means the lower one is there too.

## Tests

- Synthetic DOI arrays and layer tops with known shares: at the base within half a bottom-layer thickness and not beyond, the 90th percentile, the first-layer share, NaN DOIs left out and counted, per-line grouping, no line column.
- The warning names exactly the lines over the threshold, and none when all are under.
- An inversion stand-in, as in the DOI tests: `model_info["doi_check"]` on the smooth model, on both models with IRLS, and through msgpack.
- Locally, not committed: the 500 m and 700 m models of the survey above should give about 44% / 21% and 1% / 0%.

## Decisions

| # | Question | Options | Suggested |
|---|---|---|---|
| 1 | Where the check runs | A: simpeg, right after the DOI. B: aem-processes `invert_tem`, from the written model. | A, with the function usable from B. |
| 2 | "At the base" | A: within half a bottom-layer thickness of the half-space top. B: only a DOI exactly at the half-space top. | A, as in the ticket. |
| 3 | Warning threshold | Share of a line's soundings with the lower DOI at the base. | 5%. |
| 4 | Where the numbers go | A: `model_info` on the model (also a header comment in XYZ exports). B: a JSON file next to the outputs, which needs `invert_tem`. C: both. | A now; B arrives with the #153 run summary. |
| 5 | First-layer share | Include, or leave out. | Include; it costs one line. |
| 6 | Headroom | Informational only, or also warn when the lower DOI p90 reaches some fraction of the base. | Informational only. |

## Work

- simpeg: `doi_utils.doi_base_check` with tests; in `base.py`, `doi__base_warning`, the call in `_add_doi`, `model_info["doi_check"]` and the two log lines.
- aem-processes, later: copy `model_info["doi_check"]` into the run summary when #153 item 3 is built.
- Platform: an environment build to reach jobs, once approved.

Rough size: half a day in simpeg.
