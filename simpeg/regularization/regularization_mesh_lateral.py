import numpy as np
import scipy.sparse as sp
from simpeg.utils.code_utils import validate_active_indices

from .. import utils
from .regularization_mesh import RegularizationMesh


class LCRegularizationMesh(RegularizationMesh):
    """
    **LCRegularization Mesh**

    :param list mesh: lit including two discretize meshes
    :param numpy.ndarray active_cells: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells
    :param numpy.ndarray active_edges: bool array, size nE, that is True where we have active edges. Used to reduce the operators so we regularize only on active edges
    :param bool distance_weighted: if True, the lateral gradient divides each model
        difference by the length of its triangulation edge, and the vertical gradient
        by the distance between layer centres, so smoothness is a true gradient in
        metres and ``alpha_r`` / ``alpha_z`` relate to length scales
        (``L = sqrt(alpha / alpha_s)``). If False (the default, and the historical
        behaviour), every lateral link and every layer interface is a plain
        difference: ``alpha_r`` and ``alpha_z`` are then per-link and per-layer
        weights, not length scales, and a link between lines kilometres apart weighs
        the same as one between neighbouring soundings (ymerflow-business#119).
    :param numpy.ndarray locations: optional (n_nodes, 2) sounding coordinates used for
        the edge lengths when ``distance_weighted`` is True. Defaults to the lateral
        mesh's own nodes. Pass the true positions when the mesh was built from
        jittered ones (as for a single straight flight line).
    :param numpy.ndarray edges: optional (n_edges, 2) node index pairs the lateral
        gradient differences, in place of the triangulation's edges (e.g. a chain
        along each flight line plus the triangulation edges between lines). The
        lateral mesh then only supplies the nodes.
    :param numpy.ndarray edge_weights: optional (n_edges,) non-negative weight of each
        lateral edge; its gradient row is scaled by the square root, so the edge
        enters the smoothness as ``weight * (dm / length) ** 2`` (times ``alpha_r``).

    """

    _active_edges = None

    def __init__(
        self,
        mesh,
        active_cells=None,
        active_edges=None,
        distance_weighted=False,
        locations=None,
        edges=None,
        edge_weights=None,
        **kwargs,
    ):
        self.mesh_radial = mesh[0]
        self.mesh_vertical = mesh[1]
        self.active_edges = active_edges
        self.locations = locations
        self.edges = None if edges is None else np.asarray(edges, dtype=int).reshape(-1, 2)
        if self.edges is not None and (
            (self.edges < 0).any() or (self.edges >= self.n_nodes).any()
        ):
            raise ValueError(f"edges must index the {self.n_nodes} nodes of the lateral mesh")
        self.edge_weights = (
            None if edge_weights is None else np.asarray(edge_weights, dtype=float)
        )
        if self.edge_weights is not None:
            if self.edge_weights.shape != (self.n_edges,):
                raise ValueError(
                    f"edge_weights has shape {self.edge_weights.shape}; there are "
                    f"{self.n_edges} lateral edges"
                )
            if (self.edge_weights < 0).any():
                raise ValueError("edge_weights must be non-negative")
        self.distance_weighted = distance_weighted
        utils.set_kwargs(self, **kwargs)

    @property
    def lateral_edges(self) -> np.ndarray:
        """Node pairs the lateral gradient differences: ``edges`` if given, else the triangulation's.

        Returns
        -------
        (n_edges, 2) numpy.ndarray of int
        """
        return self.mesh_radial._edges if self.edges is None else self.edges

    @property
    def average_node_to_lateral_edge(self) -> sp.csr_matrix:
        """Average from the lateral nodes to the lateral edges, (n_edges, n_nodes)."""
        if self.edges is None:
            return self.mesh_radial.average_node_to_edge
        rows = np.repeat(np.arange(self.n_edges), 2)
        return sp.csr_matrix(
            (np.full(2 * self.n_edges, 0.5), (rows, self.edges.reshape(-1))),
            shape=(self.n_edges, self.n_nodes),
        )

    @property
    def distance_weighted(self) -> bool:
        """Whether gradients are divided by real distances (see class docstring)."""
        return self._distance_weighted

    @distance_weighted.setter
    def distance_weighted(self, value: bool):
        self._distance_weighted = bool(value)
        # Operators depend on the setting; drop any cached ones.
        self._cell_gradient_r = None
        self._cell_gradient_z = None
        self._cell_distances_r = None
        self._cell_distances_z = None

    @property
    def edge_lengths(self) -> np.ndarray:
        """Length of each lateral (triangulation) edge, in the units of the locations.

        Returns
        -------
        (n_edges, ) numpy.ndarray
        """
        if getattr(self, "_edge_lengths", None) is None:
            xy = self.mesh_radial.nodes if self.locations is None else self.locations
            xy = np.asarray(xy, dtype=float)
            if xy.shape[0] != self.n_nodes:
                raise ValueError(
                    f"locations has {xy.shape[0]} rows; the lateral mesh has "
                    f"{self.n_nodes} nodes"
                )
            edges = self.lateral_edges
            lengths = np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)
            if np.any(lengths <= 0):
                raise ValueError("lateral mesh has a zero-length edge (duplicate sounding positions)")
            self._edge_lengths = lengths
        return self._edge_lengths

    @property
    def active_cells(self) -> np.ndarray:
        """Active cells on the regularization mesh.

        A boolean array defining the cells in the regularization mesh that are active
        (i.e. updated) throughout the inversion. The values of inactive cells
        remain equal to their starting model values.

        Returns
        -------
        (n_cells, ) array of bool

        Notes
        -----
        If the property is set using a ``numpy.ndarray`` of ``int``, the setter interprets the
        array as representing the indices of the active cells. When called however, the quantity
        will have been internally converted to a boolean array.
        """
        return self._active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray):
        if getattr(self, "_active_cells", None) is not None and not all(
            self._active_cells == values
        ):
            raise AttributeError(
                "The RegulatizationMesh already has an 'active_cells' property set."
            )
        if values is not None:
            values = validate_active_indices("values", values, self.nC)
            # Ensure any cached operators created when
            # active_cells was None are deleted
            self._vol = None
            self._Pac = None
            self._Paer = None
            self._Pafz = None
            self._h_gridded_r = None
            self._h_gridded_z = None
            self._cell_gradient_z = None
            self._aveCC2Fz = None
            self._aveFz2CC = None
        self._active_cells = values

    @property
    def active_edges(self) -> np.ndarray:
        return self._active_edges

    @active_edges.setter
    def active_edges(self, values: np.ndarray):
        if getattr(self, "_active_edges", None) is not None and not all(
            self._active_edges == values
        ):
            raise AttributeError(
                "The RegulatizationMesh already has an 'active_edges' property set."
            )
        if values is not None:
            self._aveCC2Fr = None
            self._cell_gradient_r = None
            self._aveFr2CC = None

        self._active_edges = values

    @property
    def vol(self) -> np.ndarray:
        # Assume a unit area for the radial points)
        # We could use the average of cells to nodes
        self._vol = (
            np.ones(self.n_nodes, dtype=float)[:, None] * self.mesh_vertical.h[0]
        ).flatten()
        return self._vol[self.active_cells].flatten()

    @property
    def h_gridded_r(self) -> np.ndarray:
        """
        Length of cells in the raidal direction

        """
        if getattr(self, "_h_gridded_r", None) is None:
            # assume a unit length scale in radial direction
            n = self.nz * self.n_nodes
            self._h_gridded_r = np.ones(n)
        return self._h_gridded_r

    @property
    def h_gridded_z(self) -> np.ndarray:
        """
        Length of cells in the vertical direction

        """
        if getattr(self, "_h_gridded_z", None) is None:
            self._h_gridded_z = np.tile(self.mesh_vertical.h[0], self.n_nodes).flatten()
        return self._h_gridded_z

    @property
    def base_length(self) -> float:
        """Smallest dimension (i.e. edge length) for smallest cell in the mesh.

        Returns
        -------
        float
            Smallest dimension (i.e. edge length) for smallest cell in the mesh.
        """
        if getattr(self, "_base_length", None) is None:
            self._base_length = self.mesh_vertical.h[0].min()
        return self._base_length

    @property
    def dim(self) -> int:
        """Dimension of regularization mesh.

        Returns
        -------
        {2}
            Dimension of the regularization mesh.
        """
        return 2

    @property
    def cell_gradient(self) -> sp.csr_matrix:
        """Cell gradient operator (cell centers to faces).

        Built from :py:property:`~discretize.operators.differential_operators.DiffOperators.cell_gradient`.

        Returns
        -------
        (n_faces, n_cells) scipy.sparse.csr_matrix
            Cell gradient operator (cell centers to faces).
        """
        return sp.vstack([self.cell_gradient_r, self.cell_gradient_z])

    @property
    def nodal_gradient_stencil(self) -> sp.csr_matrix:
        ind_ptr = 2 * np.arange(self.n_edges + 1)
        col_inds = self.lateral_edges.reshape(-1)
        Aijs = (np.ones(self.n_edges, dtype=float)[:, None] * [-1, 1]).reshape(-1)

        return sp.csr_matrix(
            (Aijs, col_inds, ind_ptr), shape=(self.n_edges, self.n_nodes)
        )

    @property
    def cell_gradient_r(self) -> sp.csr_matrix:
        """
        Nodal gradient in radial direction

        """
        if getattr(self, "_cell_gradient_r", None) is None:
            grad = self.nodal_gradient_stencil
            if self.distance_weighted:
                grad = utils.sdiag(1.0 / self.edge_lengths) @ grad
            if self.edge_weights is not None:
                grad = utils.sdiag(np.sqrt(self.edge_weights)) @ grad
            self._cell_gradient_r = (
                self.Paer.T * sp.kron(grad, utils.speye(self.nz)) * self.Pac
            )
        return self._cell_gradient_r

    @property
    def aveCC2Fr(self) -> sp.csr_matrix:
        """
        Average of cells in the radial direction

        """
        if getattr(self, "_aveCC2Fr", None) is None:
            ave = self.average_node_to_lateral_edge
            self._aveCC2Fr = self.Paer.T * sp.kron(ave, utils.speye(self.nz)) * self.Pac
        return self._aveCC2Fr

    @property
    def cell_distances_r(self) -> np.ndarray:
        """Cell center distance array along the r-direction.

        Returns
        -------
        (n_active_faces_r, ) numpy.ndarray
            Cell center distance array along the r-direction.
        """
        if getattr(self, "_cell_distances_r", None) is None:
            if self.distance_weighted:
                # One distance per (edge, layer) row, in the cell_gradient_r ordering.
                self._cell_distances_r = self.Paer.T @ np.repeat(
                    self.edge_lengths, self.nz
                )
            else:
                Ave = self.aveCC2Fr
                self._cell_distances_r = Ave * (self.Pac.T * self.h_gridded_r)
        return self._cell_distances_r

    @property
    def cell_gradient_z(self) -> sp.csr_matrix:
        """
        Cell gradeint in vertical direction

        """
        if getattr(self, "_cell_gradient_z", None) is None:
            if self.distance_weighted:
                # discretize's cell_gradient divides by the distance between layer
                # centres; same sparsity and zero boundary rows as the stencil.
                grad = self.mesh_vertical.cell_gradient
            else:
                grad = self.mesh_vertical.stencil_cell_gradient
            self._cell_gradient_z = (
                self.Pafz.T * sp.kron(utils.speye(self.n_nodes), grad) * self.Pac
            )
        return self._cell_gradient_z

    @property
    def aveCC2Fz(self) -> sp.csr_matrix:
        """
        Average of cells in the vertical direction

        """
        if getattr(self, "_aveCC2Fz", None) is None:
            ave = self.mesh_vertical.average_cell_to_face
            self._aveCC2Fz = (
                self.Pafz.T * sp.kron(utils.speye(self.n_nodes), ave) * self.Pac
            )
        return self._aveCC2Fz

    @property
    def cell_distances_z(self) -> np.ndarray:
        """Cell center distance array along the r-direction.

        Returns
        -------
        (n_active_faces_z, ) numpy.ndarray
            Cell center distance array along the r-direction.
        """
        if getattr(self, "_cell_distances_z", None) is None:
            # Was self.aveCC2Fr (the radial operator), which has the wrong shape for
            # z faces; the vertical average is aveCC2Fz.
            Ave = self.aveCC2Fz
            self._cell_distances_z = Ave * (self.Pac.T * self.h_gridded_z)
        return self._cell_distances_z

    @property
    def nz(self) -> int:
        """
        Number of cells of the 1D vertical mesh
        """
        if getattr(self, "_nz", None) is None:
            self._nz = self.mesh_vertical.n_cells
        return self._nz

    @property
    def nFz(self) -> int:
        """
        Number of faces in the vertical direction
        """
        if getattr(self, "_nFz", None) is None:
            self._nFz = self.mesh_vertical.n_faces * self.n_nodes
        return self._nFz

    @property
    def nE(self) -> int:
        """
        Number of edges in the radial direction
        """
        if getattr(self, "_nE", None) is None:
            self._nE = self.nz * self.n_edges
        return self._nE

    @property
    def nC(self) -> int:
        """
        reduced number of cells

        :rtype: int
        :return: number of cells being regularized
        """
        if self.active_cells is not None:
            return int(self.active_cells.sum())
        return self.nz * self.n_nodes

    @property
    def n_nodes(self) -> int:
        """
        Number of nodes of the 2D simplex mesh
        """
        if getattr(self, "_n_nodes", None) is None:
            self._n_nodes = self.mesh_radial.n_nodes
        return self._n_nodes

    @property
    def n_edges(self) -> int:
        """
        Number of edges of the 2D simplex mesh
        """
        if getattr(self, "_n_edges", None) is None:
            self._n_edges = len(self.lateral_edges)
        return self._n_edges

    @property
    def Pafz(self):
        """
        projection matrix that takes from the reduced space of active z-faces
        to full modelling space (ie. nFz x nactive_cells_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x projection matrix
        """
        if getattr(self, "_Pafz", None) is None:
            if self.active_cells is None:
                self._Pafz = utils.speye(self.nFz)
            else:
                ave = self.mesh_vertical.average_face_to_cell
                aveFz2CC = sp.kron(utils.speye(self.n_nodes), ave)
                active_cells_Fz = aveFz2CC.T * self.active_cells >= 1
                self._Pafz = utils.speye(self.nFz)[:, active_cells_Fz]
        return self._Pafz

    @property
    def Pac(self):
        """
        projection matrix that takes from the reduced space of active cells to
        full modelling space (ie. nC x nactive_cells)

        :rtype: scipy.sparse.csr_matrix
        :return: active cell projection matrix
        """
        if getattr(self, "_Pac", None) is None:
            if self.active_cells is None:
                self._Pac = utils.speye(self.nz * self.n_nodes)
            else:
                self._Pac = utils.speye(self.nz * self.n_nodes)[:, self.active_cells]
        return self._Pac

    @property
    def Paer(self):
        """
        projection matrix that takes from the reduced space of active edges
        to full modelling space (ie. nE x nactive_cells_E )

        :rtype: scipy.sparse.csr_matrix
        :return: active edge projection matrix
        """
        if getattr(self, "_Paer", None) is None:
            if self.active_edges is None:
                self._Paer = utils.speye(self.nE)
            else:
                # ave = self.mesh_vertical.average_face_to_cell
                self._Paer = utils.speye(self.nE)[:, self.active_edges]
        return self._Paer

    @property
    def aveFz2CC(self):
        """
        averaging from active cell centers to active x-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active x-faces
        """
        if getattr(self, "_aveFz2CC", None) is None:
            ave = self.mesh_vertical.average_face_to_cell
            self._aveFz2CC = (
                self.Pac.T * sp.kron(utils.speye(self.n_nodes), ave) * self.Pafz
            )
        return self._aveFz2CC

    @property
    def aveFr2CC(self):
        """
        averaging from active nodes to active edges

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active edges
        """

        if getattr(self, "_aveFr2CC", None) is None:
            ave = self.average_node_to_lateral_edge.T
            self._aveFr2CC = self.Pac.T * sp.kron(ave, utils.speye(self.nz)) * self.Paer
        return self._aveFr2CC


LCRegularizationMesh.__module__ = "SimPEG.regularization"
