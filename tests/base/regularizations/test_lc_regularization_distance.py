"""Distance weighting in the laterally constrained regularization (ymerflow-business#119).

With ``distance_weighted=False`` (the default) every triangulation link is a plain
difference whatever its length, so on a survey with sounding spacing much smaller than
line spacing, soundings are tied to the next line as strongly as to their neighbours.
With ``distance_weighted=True`` each difference is divided by the distance it spans.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from discretize import SimplexMesh
from scipy.spatial import Delaunay

from simpeg import utils
from simpeg.electromagnetics.utils.em1d_utils import set_mesh_1d
from simpeg.regularization import LaterallyConstrained, LCRegularizationMesh

SPACING = 25.0  # m between soundings along a line
LINE_GAP = 2000.0  # m between the two lines
N_PER_LINE = 21
LAYERS = np.full(10, 10.0)  # 10 layers of 10 m


def two_lines():
    """Sounding positions on two parallel lines, line A at y=0, line B at y=LINE_GAP."""
    x = np.arange(N_PER_LINE) * SPACING
    xy = np.vstack([np.c_[x, np.zeros_like(x)], np.c_[x, np.full_like(x, LINE_GAP)]])
    line = np.r_[np.zeros(N_PER_LINE, int), np.ones(N_PER_LINE, int)]
    return xy, line


def make_mesh(xy, distance_weighted, locations=None, active_edges=None):
    tri = Delaunay(xy)
    radial = SimplexMesh(tri.points, tri.simplices)
    vertical = set_mesh_1d(LAYERS)
    return LCRegularizationMesh(
        [radial, vertical],
        distance_weighted=distance_weighted,
        locations=locations,
        active_edges=active_edges,
    )


def model_from(fn, xy):
    """Model in the regularization's node-major order (nz layers per sounding)."""
    zc = set_mesh_1d(LAYERS).cell_centers
    return np.concatenate([fn(p, zc) for p in xy])


class TestOperators:
    def test_default_is_unweighted_stencil(self):
        xy, _ = two_lines()
        mesh = make_mesh(xy, distance_weighted=False)
        nz = mesh.nz
        expected_r = sp.kron(mesh.nodal_gradient_stencil, utils.speye(nz))
        expected_z = sp.kron(
            utils.speye(mesh.n_nodes), mesh.mesh_vertical.stencil_cell_gradient
        )
        assert abs(mesh.cell_gradient_r - expected_r).max() == 0
        assert abs(mesh.cell_gradient_z - expected_z).max() == 0

    def test_lateral_gradient_divides_by_edge_length(self):
        xy, _ = two_lines()
        mesh = make_mesh(xy, distance_weighted=True)
        rng = np.random.default_rng(0)
        m = rng.standard_normal(mesh.nC)
        got = (mesh.cell_gradient_r @ m).reshape(mesh.n_edges, mesh.nz)
        edges = mesh.mesh_radial._edges
        lengths = np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)
        mm = m.reshape(mesh.n_nodes, mesh.nz)
        expected = (mm[edges[:, 1]] - mm[edges[:, 0]]) / lengths[:, None]
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        np.testing.assert_allclose(mesh.edge_lengths, lengths)
        # One distance per (edge, layer) row, matching the gradient rows.
        np.testing.assert_allclose(mesh.cell_distances_r, np.repeat(lengths, mesh.nz))

    def test_vertical_gradient_divides_by_layer_spacing(self):
        xy, _ = two_lines()
        mesh = make_mesh(xy, distance_weighted=True)
        slope = 0.03  # per metre
        m = model_from(lambda p, zc: slope * zc, xy)
        g = (mesh.cell_gradient_z @ m).reshape(mesh.n_nodes, -1)
        # Interior faces see the true slope; the two boundary faces are zero rows.
        np.testing.assert_allclose(g[:, 1:-1], slope, rtol=1e-12)
        np.testing.assert_allclose(g[:, [0, -1]], 0.0)

    def test_locations_override_mesh_nodes(self):
        """Edge lengths come from the true positions when the mesh used jittered ones."""
        xy, _ = two_lines()
        jittered = xy + np.random.default_rng(1).normal(scale=3.0, size=xy.shape)
        tri = Delaunay(jittered)
        radial = SimplexMesh(tri.points, tri.simplices)
        mesh = LCRegularizationMesh(
            [radial, set_mesh_1d(LAYERS)], distance_weighted=True, locations=xy
        )
        e = radial._edges
        np.testing.assert_allclose(
            mesh.edge_lengths, np.linalg.norm(xy[e[:, 1]] - xy[e[:, 0]], axis=1)
        )

    def test_cell_distances_z_has_one_value_per_z_face(self):
        xy, _ = two_lines()
        for weighted in (False, True):
            mesh = make_mesh(xy, distance_weighted=weighted)
            assert mesh.cell_distances_z.shape[0] == mesh.cell_gradient_z.shape[0]

    def test_switching_the_flag_rebuilds_operators(self):
        xy, _ = two_lines()
        mesh = make_mesh(xy, distance_weighted=False)
        unweighted = mesh.cell_gradient_r.copy()
        mesh.distance_weighted = True
        assert abs(mesh.cell_gradient_r - unweighted).max() > 0


def invert_linear(mesh, m_true, alpha_s, alpha_r, alpha_z, beta=1.0, decay=20.0):
    """Exact solution of a linear problem with depth-decaying sensitivity.

    Data observe every cell with a sensitivity exp(-z / decay), so shallow layers are
    well constrained and deep ones are not: there the regularization decides, which is
    where distance-blind links matter (as at late gates in AEM).
    """
    zc = mesh.mesh_vertical.cell_centers
    sens = np.tile(np.exp(-zc / decay), mesh.n_nodes)
    G = utils.sdiag(sens)
    d = G @ m_true
    m_ref = np.full_like(m_true, m_true.mean())
    reg = LaterallyConstrained(mesh, alpha_s=alpha_s, alpha_r=alpha_r, alpha_z=alpha_z)
    reg.reference_model = m_ref
    # The objective is quadratic, so one Newton step from m_ref is exact.
    H = (G.T @ G) + beta * reg.deriv2(m_ref)
    rhs = G.T @ (G @ m_ref - d) + beta * reg.deriv(m_ref)
    return m_ref - sp.linalg.spsolve(sp.csr_matrix(H), rhs)


def cross_line_edges(mesh):
    """Active-edge mask (per edge and layer) with only along-line links switched on."""
    e = mesh.mesh_radial._edges
    xy = mesh.mesh_radial.nodes
    along = np.abs(xy[e[:, 1], 1] - xy[e[:, 0], 1]) < LINE_GAP / 2
    return np.repeat(along, mesh.nz)


def cross_line_smear(gap, weighted, alpha_r, alpha_z, alpha_s=1e-4, beta=1.0, decay=40.0):
    """Largest deep-model change caused by the cross-line links, for lines `gap` apart.

    Two lines that differ only below 40 m. The result with all triangulation links is
    compared with the same regularization with the cross-line links switched off (the
    lines inverted independently); the difference is what the cross-line links did.
    """
    global LINE_GAP
    saved, LINE_GAP = LINE_GAP, gap
    try:
        xy, _ = two_lines()
        m_true = model_from(lambda p, zc: np.where(zc > 40, 2.0 * (p[1] > 0), 0.0), xy)
        mesh_all = make_mesh(xy, distance_weighted=weighted)
        mesh_sep = make_mesh(
            xy, distance_weighted=weighted, active_edges=cross_line_edges(mesh_all)
        )
        args = dict(alpha_s=alpha_s, alpha_r=alpha_r, alpha_z=alpha_z, beta=beta, decay=decay)
        m_all = invert_linear(mesh_all, m_true, **args)
        m_sep = invert_linear(mesh_sep, m_true, **args)
        deep = np.tile(set_mesh_1d(LAYERS).cell_centers > 40, len(xy))
        return np.abs(m_all - m_sep)[deep].max()
    finally:
        LINE_GAP = saved


def test_unweighted_coupling_ignores_distance():
    """The bug: lines 8 km apart are tied exactly as hard as lines 2 km apart."""
    near = cross_line_smear(2000.0, weighted=False, alpha_r=1.0, alpha_z=1.0)
    far = cross_line_smear(8000.0, weighted=False, alpha_r=1.0, alpha_z=1.0)
    assert near > 0.1, f"expected visible smearing between the lines: {near:.3g}"
    assert abs(far - near) / near < 0.01, f"unweighted coupling changed with distance: {near:.3g} vs {far:.3g}"


def test_weighted_coupling_falls_with_distance():
    """With the fix, at matched along-line strength (alpha_r = spacing**2, i.e. the same
    per-link weight as alpha_r = 1 unweighted), coupling falls roughly as 1/distance**2."""
    ar, az = SPACING**2, LAYERS[0] ** 2
    near = cross_line_smear(2000.0, weighted=True, alpha_r=ar, alpha_z=az)
    far = cross_line_smear(8000.0, weighted=True, alpha_r=ar, alpha_z=az)
    unweighted = cross_line_smear(2000.0, weighted=False, alpha_r=1.0, alpha_z=1.0)
    assert near < unweighted / 5
    assert far < near / 8, f"4x the distance should cut coupling ~16x: {near:.3g} -> {far:.3g}"


def test_weighted_length_scales_keep_lines_independent():
    """At a 100 m lateral / 25 m vertical length scale (alpha_s 1e-4: alpha_r 1,
    alpha_z 0.0625), lines 2 km apart no longer influence each other."""
    smear = cross_line_smear(2000.0, weighted=True, alpha_r=1.0, alpha_z=0.0625)
    unweighted = cross_line_smear(2000.0, weighted=False, alpha_r=1.0, alpha_z=1.0)
    # 0.01 in log10 resistivity is about 2%; measured 1.4e-3 against 0.16 unweighted.
    assert smear < 0.01, f"lines 2 km apart still influence each other: {smear:.3g}"
    assert smear < unweighted / 50
