import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
import libaarhusxyz
import pandas as pd

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import os, tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh, SimplexMesh
#from pymatsolver import PardisoSolver

from simpeg.utils import mkvc
from simpeg import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from simpeg.utils import mkvc
import simpeg.electromagnetics.time_domain as tdem
import simpeg.electromagnetics.utils.em1d_utils
from simpeg.electromagnetics.utils.em1d_utils import get_vertical_discretization_time
from simpeg.regularization import LaterallyConstrained

import scipy.stats


def detect_cpu_availability():
    """Return the number of CPUs this process may actually use, as an int >= 1.

    Precedence: CPU_LIMIT env var (decimal cores, e.g. "3.5") -> cgroup CFS quota
    (v2 cpu.max, then v1 cfs_quota_us/cfs_period_us) -> os.cpu_count(). Fractional
    core counts are floored; the result is clamped to a minimum of 1 so Pool() never
    raises. Set by the Kubernetes job launcher (limits == requests) so parallel
    forward simulations size their process pool to the pod's real quota, not the
    whole node's core count.
    """
    # 1. Explicit env var from the pod launcher (authoritative).
    #    Absent OR empty/whitespace-only -> fall through (not an error).
    raw = os.environ.get("CPU_LIMIT")
    if raw and raw.strip():
        cores = float(raw)          # non-empty but non-numeric should surface, not be swallowed
        if cores >= 1:
            return int(cores)       # floor; >=1 guaranteed
        return 1

    # 2. cgroup CFS quota
    #    v2: /sys/fs/cgroup/cpu.max  -> "<quota> <period>" or "max <period>"
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            cores = int(quota) / int(period)
            return max(1, int(cores))
    except FileNotFoundError:
        #    v1: cpu.cfs_quota_us / cpu.cfs_period_us  (quota == -1 means unlimited)
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            if quota > 0 and period > 0:
                return max(1, int(quota / period))
        except FileNotFoundError:
            pass

    # 3. Fallback: node core count (last resort — over-counts under a CFS quota)
    return os.cpu_count() or 1


def make_2layer(xdist, dtb, layers, res_upper=30, res_lower=300, x=None, y=None):
    """Make a synthetic model with two layers of differing resistivity (in addition to the air above).
    
    xdist is an array of shape (N,) of positions along the cross section (given as distances from the start of the line)
    dtb is an array of shape (N,) of depth to the second layer at each position.
    layers is an array of shape (M,) of depths to layer boundaries. Should include the top of the topmost layer (0), and bottom of the bottom most layer (inf).
    x, y are arrays of shape (N,) of x and y positions for each cross section point. Will default to xdist and 0 respectively.
    """
    gxdist, gz = np.meshgrid(xdist, layers)
    gdtb, dummy = np.meshgrid(dtb, layers)
    
    gdtb = gdtb[:-1,:]
    gzt = gz[:-1,:]
    gzb = gz[1:,:]
        
    bedrockoverlap = np.clip((gdtb - gzb) / (gzt - np.fmin(gzb, gzb[np.isfinite(gzb)].max())), 0, 1)
    res = bedrockoverlap * res_lower + (1 - bedrockoverlap) * res_upper
    
    if x is None:
        x = xdist
    if y is None:
        y = 0 * xdist
    
    flightlines = pd.DataFrame({"xdist": xdist, "x": x, "y": y, "interface_depth": dtb}).assign(line_no=0)
    resistivity = pd.DataFrame(res.T)
    
    xyz = libaarhusxyz.XYZ()

    xyz.flightlines = flightlines
    xyz.layer_data["resistivity"] = resistivity
    xyz.layer_data["dep_top"] = pd.DataFrame(gzt.T)
    xyz.layer_data["dep_bot"] = pd.DataFrame(gzb.T)
    
    return xyz

def add_noise(xyz, rel_uncertainty=0.01):
    """Add artificial random noise to a (synthetic) model.
    """
    dpred = xyz.dbdt_ch1gt.values.flatten()
    noise = rel_uncertainty*np.abs(dpred)*np.random.rand(len(dpred))
    xyz.layer_data["dbdt_ch1gt"] += noise.reshape(xyz.dbdt_ch1gt.shape)

def add_uncertainty_normal(xyz, rel_uncertainty):
    """Add artificial random uncertainty to a (synthetic) model, as if a set of measurements
    had been stacked for each sounding, and the stddev measured.
    """
    xyz.layer_data["dbdt_std_ch1gt"] = np.abs(rel_uncertainty
                                              * xyz.layer_data["dbdt_ch1gt"]
                                              * np.random.randn(*xyz.layer_data["dbdt_ch1gt"].shape))

def add_uncertainty(xyz, rel_uncertainty):
    """Add artificial static uncertainty to a (synthetic) model, as if a set of measurements
    had been stacked for each sounding, and the stddev measured and found to all be the same as a fraction of the measurement.
    """
    xyz.layer_data["dbdt_std_ch1gt"] = rel_uncertainty*np.abs(xyz.dbdt_ch1gt)


# ── Warm start from earlier inversion models ──────────────────────────────────

def load_start_models(dataset):
    """Return the start models in ``dataset`` as a list of XYZ objects.

    Parameters
    ----------
    dataset : libaarhusxyz.XYZ, str, os.PathLike, or a list of these
        Earlier inversion models. A path ending in ``.msgpack`` is read with
        ``libaarhusxyz.export.msgpack``, any other path as an XYZ file, and both are
        normalized to libaarhusxyz naming. An XYZ object is used as given, so it must
        already carry libaarhusxyz names (``resistivity``, ``dep_top``).

    Returns
    -------
    list of libaarhusxyz.XYZ
    """
    if isinstance(dataset, (str, os.PathLike, libaarhusxyz.XYZ)):
        dataset = [dataset]
    models = []
    for item in dataset:
        if not isinstance(item, libaarhusxyz.XYZ):
            path = os.fspath(item)
            if path.endswith(".msgpack"):
                from libaarhusxyz.export import msgpack as xyz_msgpack
                item = xyz_msgpack.load(path)
            else:
                item = libaarhusxyz.XYZ(path)
            item.normalize(naming_standard="libaarhusxyz")
        for key in ("resistivity", "dep_top"):
            if key not in item.layer_data:
                raise ValueError("Start model has no '%s' layer data (layer data: %s)"
                                 % (key, ", ".join(item.layer_data) or "none"))
        models.append(item)
    return models


def layer_tops(thicknesses):
    """Top depth of every layer, the halfspace included [m], from the layer ``thicknesses`` [m]."""
    return np.r_[0.0, np.cumsum(thicknesses)]


def describe_layers(tops):
    """Name a layering in a log line, from its layer top depths [m]."""
    if len(tops) < 2:
        return "1 layer (a halfspace)"
    return "%d layers, boundaries %.4g to %.4g m" % (len(tops), tops[1], tops[-1])


def resample_resistivity(resistivity, tops, target_tops):
    """Resample resistivity onto another layering, linearly in log10 resistivity over depth.

    Every layer is represented by its mid-depth; a halfspace's mid-depth is its top plus
    half the thickness of the layer above it. Above the first and below the last source
    mid-depth the end values are held.

    Parameters
    ----------
    resistivity : ndarray, shape (n, m)
        Resistivity of ``n`` soundings on the source layering [ohm.m].
    tops : ndarray, shape (m,)
        Source layer top depths [m], shared by the ``n`` soundings.
    target_tops : ndarray, shape (k,)
        Target layer top depths [m].

    Returns
    -------
    ndarray, shape (n, k)
        Resistivity on the target layering [ohm.m].
    """
    def mid_depths(t):
        thick = np.diff(t)
        return t + np.r_[thick, thick[-1:] if len(thick) else [0.0]] / 2

    source_mid, target_mid = mid_depths(np.asarray(tops, float)), mid_depths(np.asarray(target_tops, float))
    # np.interp is linear in the values it is given, so interpolating the unit vectors
    # gives the (k, m) matrix that maps source layers to target layers.
    weights = np.array([np.interp(target_mid, source_mid, unit) for unit in np.eye(len(source_mid))]).T
    return 10 ** (np.log10(resistivity) @ weights.T)


def start_model_resistivity(model, thicknesses, resample=True):
    """Resistivity of a start model's soundings on the inversion's layers.

    Parameters
    ----------
    model : libaarhusxyz.XYZ
        Start model with ``resistivity`` [ohm.m] and ``dep_top`` [m] layer data.
    thicknesses : ndarray
        The inversion's layer thicknesses [m]; the halfspace below them is implied.
    resample : bool
        Resample a model whose layers differ from the inversion's
        (:func:`resample_resistivity`). If False, a mismatch is an error.

    Returns
    -------
    resistivity : ndarray, shape (n_soundings, n_layers)
        Resistivity on the inversion's layers [ohm.m]; NaN rows for soundings with no
        usable model (a NaN or non-positive value, or no layer depths).
    layerings : list of str
        Each layering found that differs from the inversion's, described for the log.

    Raises
    ------
    ValueError
        If the layers differ and ``resample`` is False.
    """
    res = model.layer_data["resistivity"].values.astype(float)
    tops = model.layer_data["dep_top"].values.astype(float)
    target = layer_tops(thicknesses)
    out = np.full((len(res), len(target)), np.nan)
    usable = np.isfinite(tops).all(axis=1) & np.isfinite(res).all(axis=1) & (res > 0).all(axis=1)
    layerings = []
    for layering in np.unique(tops[usable], axis=0):
        rows = usable & (tops == layering).all(axis=1)
        if len(layering) == len(target) and np.allclose(layering, target, rtol=1e-3, atol=1e-2):
            out[rows] = res[rows]
            continue
        if not resample:
            raise ValueError(
                "Start model layers (%s) differ from the inversion's (%s); set "
                "startmodel__resample_layers = True to resample them"
                % (describe_layers(layering), describe_layers(target)))
        layerings.append(describe_layers(layering))
        out[rows] = resample_resistivity(res[rows], layering, target)
    return out, layerings


def median_line_spacing(xy, line):
    """Median distance from a sounding to the nearest sounding on another line.

    Parameters
    ----------
    xy : ndarray, shape (n, 2)
        Sounding positions [CRS linear unit].
    line : array-like, shape (n,)
        Line id of each sounding.

    Returns
    -------
    float
        Median line spacing [CRS linear unit]; NaN when there is only one line.
    """
    line = np.asarray(line)
    distances = []
    for line_id in np.unique(line):
        on = line == line_id
        if on.all():
            return np.nan
        distances.append(cKDTree(xy[~on]).query(xy[on])[0])
    return float(np.median(np.concatenate(distances)))


def sounding_keys(flightlines, line_column):
    """Return ``(line, sounding_id)`` per sounding as a two-column DataFrame, or None.

    Parameters
    ----------
    flightlines : pandas.DataFrame
    line_column : str or None
        Name of the line column (``XYZ.line_id_column``).

    Returns
    -------
    pandas.DataFrame or None
        Columns ``line`` and ``sounding_id``, numeric ones as float and others as str so
        two datasets compare equal whatever integer or float type they were read with;
        None when either column is missing.
    """
    if line_column is None or "sounding_id" not in flightlines.columns:
        return None
    keys = flightlines[[line_column, "sounding_id"]].set_axis(["line", "sounding_id"], axis=1)
    return keys.apply(lambda col: col.astype(float) if pd.api.types.is_numeric_dtype(col) else col.astype(str))


def match_soundings(xy, xy_model, fill_distance, keys=None, keys_model=None):
    """Find each sounding's start-model sounding: by id, else the nearest within ``fill_distance``.

    Parameters
    ----------
    xy : ndarray, shape (n, 2)
        Positions of the soundings being inverted [CRS linear unit].
    xy_model : ndarray, shape (m, 2)
        Positions of the usable start-model soundings [CRS linear unit].
    fill_distance : float
        Largest distance for a match by position [CRS linear unit].
    keys, keys_model : pandas.MultiIndex, optional
        ``(line, sounding_id)`` of each sounding and each model sounding. Ids are used
        only when both are given.

    Returns
    -------
    index : ndarray of int, shape (n,)
        Row of the matched model sounding, -1 where there is none.
    by_id : ndarray of bool, shape (n,)
        True where the match came from the ids.
    """
    index = np.full(len(xy), -1)
    if keys is not None and keys_model is not None:
        lookup = pd.Series(np.arange(len(keys_model)), index=keys_model)
        lookup = lookup[~lookup.index.duplicated()]
        index = lookup.reindex(keys).fillna(-1).values.astype(int)
    by_id = index >= 0
    rest = ~by_id
    if rest.any() and len(xy_model):
        dist, nearest = cKDTree(xy_model).query(xy[rest], distance_upper_bound=fill_distance)
        index[rest] = np.where(np.isfinite(dist), nearest, -1)
    return index, by_id


def lateral_median(model, xy, radius):
    """Replace each sounding's model, layer by layer, by the median over the soundings within ``radius``.

    Parameters
    ----------
    model : ndarray, shape (n, n_layers)
        One model per sounding, as log conductivity [ln(S/m)] or log10 resistivity
        [log10(ohm.m)]: the two differ by a linear change of scale, which the median
        commutes with.
    xy : ndarray, shape (n, 2)
        Sounding positions [CRS linear unit].
    radius : float
        Search radius [CRS linear unit].

    Returns
    -------
    ndarray, shape (n, n_layers)
    """
    neighbors = cKDTree(xy).query_ball_point(xy, radius)
    return np.array([np.median(model[rows], axis=0) for rows in neighbors])
