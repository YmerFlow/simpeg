"""Explicit lateral edges with a weight per edge in LCRegularizationMesh (YmerFlow/simpeg#31).

The lateral mesh normally differences along the edges of its triangulation. Given ``edges``,
it differences along those instead, and ``edge_weights`` scales each edge's term, so a
smoothness term is the sum over edges of weight * (dm / length)**2 with distance weighting.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from discretize import SimplexMesh
from scipy.spatial import Delaunay

from simpeg import utils
from simpeg.electromagnetics.utils.em1d_utils import set_mesh_1d
from simpeg.regularization import LCRegularizationMesh

LAYERS = np.full(4, 10.0)
XY = np.array([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0], [0.0, 500.0], [30.0, 500.0]])
EDGES = np.array([[0, 1], [1, 2], [3, 4], [1, 4]])       # two chains and one link between them
WEIGHTS = np.array([1.0, 1.0, 1.0, 400.0])


def make_mesh(edges=None, edge_weights=None, distance_weighted=True):
    tri = Delaunay(XY)
    return LCRegularizationMesh(
        [SimplexMesh(tri.points, tri.simplices), set_mesh_1d(LAYERS)],
        distance_weighted=distance_weighted, locations=XY,
        edges=edges, edge_weights=edge_weights)


def test_gradient_differences_the_given_edges_scaled_by_root_weight_over_length():
    mesh = make_mesh(EDGES, WEIGHTS)
    lengths = np.linalg.norm(XY[EDGES[:, 1]] - XY[EDGES[:, 0]], axis=1)
    stencil = sp.csr_matrix(
        (np.tile([-1.0, 1.0], len(EDGES)), EDGES.ravel(), 2 * np.arange(len(EDGES) + 1)),
        shape=(len(EDGES), len(XY)))
    expected = sp.kron(utils.sdiag(np.sqrt(WEIGHTS) / lengths) @ stencil, utils.speye(mesh.nz))
    assert mesh.n_edges == len(EDGES)
    np.testing.assert_allclose(mesh.cell_gradient_r.toarray(), expected.toarray())
    np.testing.assert_allclose(mesh.edge_lengths, lengths)


def test_averaging_follows_the_given_edges():
    mesh = make_mesh(EDGES, WEIGHTS)
    ave = mesh.aveCC2Fr.toarray()
    assert ave.shape == (len(EDGES) * mesh.nz, len(XY) * mesh.nz)
    np.testing.assert_allclose(ave.sum(axis=1), 1.0)
    np.testing.assert_allclose(mesh.aveFr2CC.toarray(), ave.T)
    m = np.repeat(np.arange(len(XY), dtype=float), mesh.nz)       # node index in every layer
    np.testing.assert_allclose(ave @ m, np.repeat(EDGES.mean(axis=1), mesh.nz))


def test_without_edges_the_triangulation_is_used_unchanged():
    mesh = make_mesh()
    assert mesh.edges is None and mesh.edge_weights is None
    np.testing.assert_array_equal(mesh.lateral_edges, mesh.mesh_radial._edges)
    assert mesh.n_edges == mesh.mesh_radial.n_edges
    expected = sp.kron(utils.sdiag(1 / mesh.edge_lengths) @ mesh.nodal_gradient_stencil,
                       utils.speye(mesh.nz))
    assert abs(mesh.cell_gradient_r - expected).max() == 0


@pytest.mark.parametrize("edges, weights, match", [
    (np.array([[0, 5]]), None, "index the 5 nodes"),
    (EDGES, np.ones(3), "edge_weights has shape"),
    (EDGES, np.array([1.0, 1.0, 1.0, -1.0]), "non-negative"),
])
def test_bad_edges_or_weights_raise(edges, weights, match):
    with pytest.raises(ValueError, match=match):
        make_mesh(edges, weights)
