# Clustered Start Model for AEM Inversion

## Motivation

AEM surveys typically contain tens of thousands of soundings. Inverting all of them
from a flat halfspace start model is expensive: each Gauss-Newton iteration runs a
forward model for every sounding. Many soundings are geophysically similar (same
decay shape, same flight altitude) and will converge to nearly identical models.

The idea is to cluster soundings by their raw data and flight altitude, invert only
one representative per cluster, and use the resulting models as the start model for
the full inversion. This should both reduce the number of GN iterations needed
(better starting point) and optionally allow a cheap first-stage inversion over far
fewer soundings.

## Design Overview

Two-stage approach:

1. **Cluster stage**: group all soundings into K clusters, invert one representative
   per cluster.
2. **Full stage**: run the normal inversion, but initialise every sounding from its
   cluster's inverted model instead of a flat halfspace.

Enabled by setting `clustering__n_clusters` to a positive integer. Default is
`None`, which disables clustering entirely and preserves current behaviour.

## Implementation Location

Everything goes in `base.py` (`XYZSystem`). No changes needed in `dual.py` or
`single.py`: the existing abstractions (`data_array_nan`, `data_uncert_array`,
`flightlines[alt_column]`) already expose all the instrument-specific data the
clustering logic needs.

## Parameters

All new parameters follow the existing `namespace__name` convention and can be
overridden at instantiation time.

| Parameter | Default | Description |
|---|---|---|
| `clustering__n_clusters` | `None` | Number of clusters K. `None` disables clustering. |
| `clustering__random_seed` | `None` | Random seed for K-means reproducibility. |
| `clustering__valid_gate_threshold` | `0.5` | Gates present in fewer than this fraction of soundings are excluded from the feature vector. |
| `regularization__mref` | `'startmodel'` | Reference model for regularization: `'startmodel'` (cluster-derived) or `'halfspace'` (flat). |

All other inversion parameters for the cluster sub-inversion can be overridden with
the `cluster_inversion__` prefix, which strips the prefix before passing them to the
sub-inversion. Sensible defaults:

| Parameter | Default | Rationale |
|---|---|---|
| `cluster_inversion__regularization__alpha_r` | `0` | Cluster representative positions are artificial; lateral regularization is meaningless. |
| `cluster_inversion__validate` | `False` | Cluster XYZ is constructed, not measured; unit check would be misleading. |

## Feature Vector Construction

Per sounding, the feature vector is:

1. `log(|data_array_nan|)` for each gate, restricted to gates that are non-NaN in at
   least `clustering__valid_gate_threshold` of all soundings.
2. Flight altitude from `flightlines[alt_column]`.

The full feature matrix is then z-scored (subtract mean, divide by std per feature)
before K-means, so no single feature dominates by scale.

Soundings excluded by `sounding_filter` are not clustered and do not appear in the
inversion; they remain NaN in all outputs. No start model is needed for them.

## Cluster Representative XYZ

For each cluster, a single representative sounding is constructed:

**`flightlines`**: column-wise `nanmean` of all member soundings (x, y, altitude,
etc.).

**Data channels** (`dbdt_ch1gt`, `dbdt_ch2gt`, …): `nanmean` per gate across member
soundings. A gate is NaN in the representative only if all members are NaN for that
gate.

**Uncertainty channels** (`dbdt_std_ch1gt`, …): combined from two independent
sources, added in quadrature:

```
std_representative = sqrt(
    (nanmean(std_individual) / sqrt(count_non_nan))**2   # uncertainty on the mean
    + nanstd(data_values)**2                              # within-cluster spread
)
```

The within-cluster spread term is small when the cluster is tight (good clustering)
and dominates when it is not, giving the inversion appropriate scepticism about a
poorly-representative cluster average.

## Cluster Sub-inversion

The K representative soundings are assembled into a new `libaarhusxyz.XYZ` and
inverted using `type(self)` — the same instrument class, inheriting all waveform,
gate-time, and model-discretisation parameters. Only inversion-control parameters
differ, via the `cluster_inversion__` override namespace.

The same `make_thicknesses()` is used for both stages, ensuring model
parameterisation is identical.

## Start Model Copy-back

After the cluster sub-inversion, each sounding receives the inverted model of its
cluster as its start model. Concretely, `make_startmodel(thicknesses)` is overridden
to return `log(1/resistivity)` shaped `(N * n_layers,)`, where each sounding's
block is filled from its cluster representative's inverted resistivity.

## Reference Model (`mref`)

`make_regularization` currently calls `make_startmodel` for `reg.mref`. A new
`make_mref(thicknesses)` method is introduced that `make_regularization` delegates
to. Controlled by `regularization__mref`:

- `'startmodel'` (default): `mref` = cluster-derived start model, regularising the
  full inversion toward the cluster solution.
- `'halfspace'`: `mref` = flat halfspace at `startmodel__res`, matching current
  behaviour.

## Cluster ID Output

After the full inversion, the cluster index for each sounding is written as a
`cluster_id` column in `flightlines` of all output XYZ objects. This allows the
user to visualise cluster assignments on a map and assess geographic coherence.

## Choosing K

The right K trades off coverage of dataset variance against the cost of the cluster
sub-inversion. Plotting within-cluster inertia (sum of squared distances to cluster
centre) vs K produces an L-shaped curve; the knee of the curve is a principled
choice.

The `kneed` library (Kneedle algorithm) can locate this knee automatically. A helper
method `clustering_knee_plot(k_range)` will run K-means for each K in `k_range`,
plot the inertia curve, and mark the detected knee — useful for choosing K before
committing to a full run.

## Sequence of Method Calls (full inversion with clustering)

```
invert()
  └─ make_startmodel(thicknesses)          # overridden: triggers cluster stage
       ├─ _build_feature_matrix()
       ├─ _run_kmeans()                    # returns cluster_ids, centroids
       ├─ _build_cluster_xyz()             # representative XYZ for K soundings
       ├─ type(self)(cluster_xyz, **cluster_inversion_opts).invert()
       └─ _cluster_models_to_startmodel()  # copy-back → (N*n_layers,) array
  └─ make_inversion()
       └─ make_regularization(thicknesses)
            └─ make_mref(thicknesses)      # flat or cluster-derived per option
  └─ inv.run(startmodel)
  └─ make_inversion_outputs()              # writes cluster_id to flightlines
```

## Files Changed

- `SimPEG/electromagnetics/utils/static_instrument/base.py` — all new logic
- No changes to `dual.py`, `single.py`, `utils.py`, `xyzfilter.py`
