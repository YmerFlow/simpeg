"""Lateral smoothing along a line and between lines (YmerFlow/simpeg#31).

Setting ``regularization__alpha_r_inline`` / ``alpha_r_otherline`` (or their lengths
``L_inline`` / ``L_otherline``) splits the lateral links in two: a chain along each line in
flight order, and the Delaunay edges between lines. Every alpha can also be given as a length,
alpha = alpha_s * L**2. These tests build the regularization of a bare XYZSystem from sounding
positions alone; no data or simulation is needed.
"""

import typing

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial import Delaunay

import libaarhusxyz

from simpeg.electromagnetics.utils.static_instrument.base import XYZSystem

LAYERS = np.full(5, 10.0)                       # thicknesses [m], so 6 layers
NZ = len(LAYERS) + 1
DW = dict(regularization__distance_weighted=True, regularization__alpha_s=1e-4)


def make_xyz(x, y, line, xdist=None, timestamp=None, epsg=32632):
    xyz = libaarhusxyz.XYZ()
    fl = {"x": np.asarray(x, float), "y": np.asarray(y, float), "line_no": line}
    if xdist is not None:
        fl["xdist"] = xdist
    if timestamp is not None:
        fl["timestamp"] = timestamp
    xyz.flightlines = pd.DataFrame(fl)
    xyz.model_info["projection"] = epsg
    return xyz


def parallel_lines(n=6, spacing=25.0, gap=1900.0, shuffle=True):
    """Two parallel lines, rows shuffled so that flight order is not file order."""
    x = np.tile(np.arange(n) * spacing, 2)
    y = np.repeat([0.0, gap], n)
    line = np.repeat([101, 102], n)
    order = np.random.default_rng(0).permutation(2 * n) if shuffle else np.arange(2 * n)
    return make_xyz(x[order], y[order], line[order], xdist=x[order])


def links(reg):
    """The lateral links as {sorted sounding pair: weight}."""
    mesh = reg.regularization_mesh
    w = np.ones(mesh.n_edges) if mesh.edge_weights is None else mesh.edge_weights
    return {tuple(sorted(e)): wi for e, wi in zip(mesh.lateral_edges.tolist(), w)}


def chain(xyz, by="xdist"):
    """Expected inline links: consecutive soundings of each line, by flight order."""
    pairs = set()
    for _, g in xyz.flightlines.groupby("line_no"):
        idx = g.sort_values(by, kind="stable").index.to_numpy()
        pairs |= {tuple(sorted(p)) for p in zip(idx[:-1].tolist(), idx[1:].tolist())}
    return pairs


def delaunay_between_lines(xyz):
    fl = xyz.flightlines
    tri = Delaunay(fl[["x", "y"]].to_numpy())
    line = fl["line_no"].to_numpy()
    pairs = {tuple(sorted((int(a), int(b)))) for s in tri.simplices
             for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2]))}
    return {p for p in pairs if line[p[0]] != line[p[1]]}


def reg_of(xyz, **opts):
    return XYZSystem(xyz, **opts).make_regularization(LAYERS)


# ── Which links ──────────────────────────────────────────────────────────────

def test_parallel_lines_get_a_chain_each_and_the_delaunay_edges_between():
    xyz = parallel_lines()
    reg = reg_of(xyz, **DW, regularization__L_inline=100.0, regularization__L_otherline=1000.0)
    line = xyz.flightlines["line_no"].to_numpy()
    got = links(reg)
    inline = {p: w for p, w in got.items() if line[p[0]] == line[p[1]]}
    other = {p: w for p, w in got.items() if line[p[0]] != line[p[1]]}
    assert set(inline) == chain(xyz)
    assert set(other) == delaunay_between_lines(xyz)
    np.testing.assert_allclose(list(inline.values()), 1.0)       # 1e-4 * 100**2
    np.testing.assert_allclose(list(other.values()), 100.0)      # 1e-4 * 1000**2
    assert reg.alpha_r == 1.0                                    # the alphas are on the links


def test_tie_crossing_does_not_break_either_lines_chain():
    xa = np.arange(9) * 25.0                                     # line along x, y = 0
    yt = np.arange(9) * 25.0 - 100.0                             # tie along y, crossing at x = 112.5
    xyz = make_xyz(np.r_[xa, np.full(9, 112.5)], np.r_[np.zeros(9), yt],
                   np.repeat([1, 2], 9), xdist=np.r_[xa, yt + 100.0])
    got = links(reg_of(xyz, **DW, regularization__L_inline=100.0, regularization__L_otherline=1000.0))
    line = xyz.flightlines["line_no"].to_numpy()
    assert {p for p in got if line[p[0]] == line[p[1]]} == chain(xyz)
    assert (4, 5) in got                                         # 100 m -> 125 m on the line, past the tie
    crossing = 9 + 4                                             # the tie's sounding at (112.5, 0)
    assert (4, crossing) in got and (5, crossing) in got         # short otherline links at the crossing
    assert got[(4, crossing)] == 100.0


def test_chords_of_a_curving_line_are_left_out():
    angle = np.linspace(0, np.radians(150), 9)
    x, y = 300 * np.cos(angle), 300 * np.sin(angle)
    xdist = 300 * angle
    xyz = make_xyz(x, y, np.ones(9, int), xdist=xdist)
    assert len(links(reg_of(xyz, **DW))) > 8                     # Delaunay alone has chords
    got = links(reg_of(xyz, **DW, regularization__L_inline=100.0))
    assert set(got) == chain(xyz)


def test_time_orders_a_line_without_along_line_distance():
    x = np.array([50.0, 0.0, 75.0, 25.0])
    t = pd.Timestamp("2026-01-01") + pd.to_timedelta(x / 25.0, unit="s")
    xyz = make_xyz(np.r_[x, x], np.r_[np.zeros(4), np.full(4, 2000.0)], np.repeat([1, 2], 4),
                   timestamp=np.r_[t, t])
    got = links(reg_of(xyz, **DW, regularization__L_inline=100.0, regularization__alpha_r_otherline=0))
    assert set(got) == chain(xyz, by="timestamp") == {(1, 3), (0, 3), (0, 2), (5, 7), (4, 7), (4, 6)}


def test_unset_keeps_the_triangulation_and_one_alpha_r():
    xyz = parallel_lines()
    for opts, alpha_r in ((DW, 1.0), (dict(DW, regularization__alpha_r=4.0), 4.0), ({}, 1.0)):
        reg = reg_of(xyz, **opts)
        mesh = reg.regularization_mesh
        assert mesh.edges is None and mesh.edge_weights is None
        assert reg.alpha_r == alpha_r


# ── alpha_r_otherline = 0 ────────────────────────────────────────────────────

def test_otherline_zero_is_each_line_regularized_on_its_own():
    """The joint objective is the sum of the per-line ones, so with the same beta (and the
    misfit separable by sounding anyway) the joint inversion has the per-line minimizers."""
    xyz = parallel_lines(shuffle=False)
    opts = dict(DW, regularization__L_inline=100.0, regularization__alpha_r_otherline=0.0)
    fl = xyz.flightlines
    a, b = (make_xyz(g.x, g.y, g.line_no.to_numpy(), xdist=g.xdist.to_numpy())
            for _, g in fl.groupby("line_no"))
    joint, reg_a, reg_b = reg_of(xyz, **opts), reg_of(a, **opts), reg_of(b, **opts)
    line = fl["line_no"].to_numpy()
    assert all(line[p[0]] == line[p[1]] for p in links(joint))
    m = np.random.default_rng(1).normal(np.log(0.01), 1.0, len(fl) * NZ)
    ma, mb = m[:len(a.flightlines) * NZ], m[len(a.flightlines) * NZ:]
    np.testing.assert_allclose(joint(m), reg_a(ma) + reg_b(mb), rtol=1e-12)
    np.testing.assert_allclose(joint.deriv(m), np.r_[reg_a.deriv(ma), reg_b.deriv(mb)], rtol=1e-12)
    blocks = sp.block_diag([reg_a.deriv2(ma), reg_b.deriv2(mb)])
    assert abs(joint.deriv2(m) - blocks).max() < 1e-12 * abs(blocks).max()


# ── Alphas and lengths ───────────────────────────────────────────────────────

def weights(**opts):
    return XYZSystem(parallel_lines(), **dict(DW, **opts)).smoothness_weights()


@pytest.mark.parametrize("opts, key, alpha, length", [
    (dict(regularization__L_inline=100.0), "inline", 1.0, 100.0),                 # a length sets its alpha
    (dict(regularization__alpha_r_inline=4.0), "inline", 4.0, 200.0),             # an alpha sets its length
    (dict(regularization__alpha_r_inline=1.0000005, regularization__L_inline=100.0), "inline", 1.0, 100.0),
    (dict(regularization__alpha_s=4e-4, regularization__L_inline=100.0), "inline", 4.0, 100.0),  # alpha_s moves the alpha
    (dict(regularization__L_r=200.0), "r", 4.0, 200.0),                           # alpha_r left at its default
    (dict(regularization__L_z=25.0), "z", 0.0625, 25.0),
])
def test_a_pair_resolves_from_either_side(opts, key, alpha, length):
    w = weights(**opts)
    names = {"inline": ("alpha_r_inline", "L_inline"), "r": ("alpha_r", "L_r"), "z": ("alpha_z", "L_z")}[key]
    assert w[names[0]] == pytest.approx(alpha, rel=1e-6)
    assert w[names[1]] == pytest.approx(length)


def test_a_pair_that_disagrees_stops_the_run_naming_both():
    with pytest.raises(ValueError, match=r"regularization__alpha_r_inline = 2 and "
                                         r"regularization__L_inline = 100 disagree"):
        weights(regularization__alpha_r_inline=2.0, regularization__L_inline=100.0)


def test_parameters_set_on_a_subclass_count_as_given():
    """The platform sets parameters as attributes of a subclass, not as options."""
    class System(XYZSystem):
        regularization__alpha_r = 1.0
        regularization__L_r = 200.0
    with pytest.raises(ValueError, match="regularization__alpha_r = 1 and regularization__L_r = 200"):
        System(parallel_lines(), **DW).smoothness_weights()


def test_options_override_a_subclass_length_as_the_cluster_start_model_does():
    class System(XYZSystem):
        regularization__L_r = 200.0
    w = System(parallel_lines(), **DW, regularization__alpha_r=0,
               regularization__L_r=None).smoothness_weights()
    assert w["alpha_r"] == 0 and not w["split"]


def test_the_weight_not_set_defaults_to_alpha_r():
    w = weights(regularization__L_inline=100.0, regularization__alpha_r=9.0)
    assert w["split"]
    assert (w["alpha_r_otherline"], w["L_otherline"]) == pytest.approx((9.0, 300.0))


@pytest.mark.parametrize("opts, match", [
    (dict(regularization__alpha_r_inline=1.0), "need regularization__distance_weighted"),
    (dict(regularization__L_otherline=1000.0), "needs regularization__distance_weighted"),
    (dict(regularization__L_z=25.0), "needs regularization__distance_weighted"),
])
def test_lengths_and_the_split_need_distance_weighting(opts, match):
    with pytest.raises(ValueError, match=match):
        XYZSystem(parallel_lines(), **opts).smoothness_weights()


def test_the_split_needs_a_line_column_and_flight_order():
    xyz = parallel_lines()
    no_line = make_xyz(xyz.flightlines.x, xyz.flightlines.y, None, xdist=xyz.flightlines.xdist)
    del no_line.flightlines["line_no"]
    with pytest.raises(ValueError, match="needs a line column"):
        reg_of(no_line, **DW, regularization__L_inline=100.0)
    no_order = make_xyz(xyz.flightlines.x, xyz.flightlines.y, xyz.flightlines.line_no)
    with pytest.raises(ValueError, match="flight order"):
        reg_of(no_order, **DW, regularization__L_inline=100.0)


# ── The run log and the schema ───────────────────────────────────────────────

def test_the_run_log_prints_every_alpha_with_its_length(capsys):
    reg_of(parallel_lines(), **DW, regularization__L_inline=100.0, regularization__L_otherline=1000.0)
    assert ("##SMOOTHNESS## alpha_s 0.0001, alpha_r_inline 1 (L_inline 100 m), "
            "alpha_r_otherline 100 (L_otherline 1000 m), alpha_z 1 (L_z 100 m)") in capsys.readouterr().out
    xyz = parallel_lines()
    xyz.model_info["projection"] = 2258                          # a State Plane CRS in US survey feet
    reg_of(xyz, **DW, regularization__L_r=300.0)
    assert "alpha_r 9 (L_r 300 ft)" in capsys.readouterr().out
    reg_of(parallel_lines())                                     # distance weighting off: no lengths
    assert "##SMOOTHNESS## alpha_s 0.0001, alpha_r 1, alpha_z 1\n" in capsys.readouterr().out


LENGTHS = {"L_r": "alpha_r", "L_z": "alpha_z", "L_inline": "alpha_r_inline",
           "L_otherline": "alpha_r_otherline"}


def test_each_length_is_annotated_with_its_alpha():
    hints = typing.get_type_hints(XYZSystem, include_extras=True)
    for length, alpha in LENGTHS.items():
        assert hints["regularization__" + length].__metadata__ == (
            {"swaggerspect_x-length-scale": {"alpha": alpha, "alpha_s": "alpha_s"}},)


def test_the_schema_links_each_length_to_its_sibling_alpha():
    swaggerspect = pytest.importorskip("swaggerspect")
    from swaggerspect import class_api
    api = swaggerspect.group_apis_parameters({"anyOf": [class_api.get_class_api(XYZSystem, name="S")]})
    group = api["anyOf"][0]["properties"]["S"]["properties"]["regularization"]["properties"]
    for length, alpha in LENGTHS.items():
        assert group[length]["type"] == "number"
        assert group[length]["x-length-scale"] == {"alpha": alpha, "alpha_s": "alpha_s"}
        assert alpha in group and "alpha_s" in group
