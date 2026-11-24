import numpy as np
import scipy.sparse as sp
from scipy import signal

from ...utils import (
    mkvc,
    validate_type,
    validate_direction,
    validate_float,
    validate_integer,
    validate_ndarray_with_shape,
)
from discretize.utils import Zero
from ...survey import BaseTimeRx


def _validate_filter_order(name, var):
    """Check a low-pass filter order: 0, between 0 and 1, or an integer of 1 or more."""
    var = validate_float(name, var, min_val=0.0)
    if var > 1 and var != int(var):
        raise ValueError(
            f"'{name}' must be 0 (no filter), between 0 and 1, or an integer "
            f"of 1 or more, got {var}"
        )
    return var


def gex_low_pass_response(frequencies, order, cutoff_frequency):
    """Frequency response of one receiver low-pass filter, with the order read
    as AarhusInv reads the filter order in a GEX or TEM file.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Frequencies [Hz].
    order : float
        Filter order as the GEX writes it [dimensionless]. 0 is no filter. An
        integer n of 1 or more is a Butterworth low-pass of order n. A value
        z between 0 and 1 is a Gaussian-RLC filter with damping z, read as
        the second-order low-pass ``w0**2 / (s**2 + 2 * z * w0 * s + w0**2)``
        with ``w0 = 2 * pi * cutoff_frequency``.
    cutoff_frequency : float
        Cutoff frequency [Hz].

    Returns
    -------
    numpy.ndarray of complex
        The response H at ``s = 2j * pi * frequencies``, dimensionless; all
        ones for order 0.

    Raises
    ------
    ValueError
        If the order is negative, or above 1 and not an integer.

    Notes
    -----
    The AarhusInv manual (version 8) names the Gaussian-RLC filter but gives
    no formula for it. With the second-order form above for a SkyTEM coil
    filter of 0.99 at 210 kHz, the forward of a published Workbench model
    comes within 1-2% of Workbench's own forward at the early gates, where
    reading 0.99 as a first-order stage left it 5-10% low.

    AarhusInv applies the Gaussian-RLC filter before the front gate and the
    other filters after it. Without a front gate in the simulation, all the
    filters multiply the whole response, and their order does not matter.
    """
    order = _validate_filter_order("order", order)
    frequencies = np.asarray(frequencies, dtype=float)
    if order == 0:
        return np.ones(frequencies.shape, dtype=complex)
    if order < 1:
        s = 1j * frequencies / cutoff_frequency
        return 1 / (s**2 + 2 * order * s + 1)
    numer, denom = signal.butter(
        int(order), 2 * np.pi * cutoff_frequency, "low", analog=True
    )
    return signal.freqs(numer, denom, worN=2 * np.pi * frequencies)[1]


class BaseRx(BaseTimeRx):
    """Base TDEM receiver class

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'z', 'x', 'y'} or numpy.ndarray
        Receiver orientation.
    times : (n_times) numpy.ndarray
        Time channels
    gate_windows : (n_times, 2) numpy.ndarray, optional
        Open and close time [s] of each gate. The 1D layered simulation then
        returns each gate as the average over its window; see
        ``gate_windows``.
    gate_window_nodes : int, default: 5
        Gauss-Legendre nodes per gate window.
    """

    def __init__(
        self,
        locations,
        times,
        orientation="z",
        use_source_receiver_offset=False,
        bw_cutoff_frequency=3e5,
        bw_power=0.0,
        lp_cutoff_frequency=2.1e5,
        lp_power=0.0,
        lp_cutoff_frequency_1=3e5,
        lp_power_1=0.0,
        lp_cutoff_frequency_2=3e5,
        lp_power_2=0.0,
        gate_windows=None,
        gate_window_nodes=5,
        **kwargs,
    ):
        if (key := "projComp") in kwargs.keys():
            raise TypeError(f"'{key}' property has been removed.")

        if locations is None:
            raise AttributeError("'locations' are required. Cannot be 'None'")

        if times is None:
            raise AttributeError("'times' are required. Cannot be 'None'")

        self.orientation = orientation
        self.use_source_receiver_offset = use_source_receiver_offset
        self.bw_cutoff_frequency = bw_cutoff_frequency
        self.bw_power = bw_power
        self.lp_cutoff_frequency = lp_cutoff_frequency
        self.lp_power = lp_power
        self.lp_cutoff_frequency_1 = lp_cutoff_frequency_1
        self.lp_power_1 = lp_power_1
        self.lp_cutoff_frequency_2 = lp_cutoff_frequency_2
        self.lp_power_2 = lp_power_2
        super().__init__(locations=locations, times=times, **kwargs)
        self.gate_window_nodes = gate_window_nodes
        self.gate_windows = gate_windows

    @property
    def orientation(self):
        """Orientation of the receiver.

        Returns
        -------
        numpy.ndarray
            Orientation of the receiver.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_direction("orientation", var, dim=3)

    @property
    def use_source_receiver_offset(self):
        """Use source-receiver offset.

        Whether to interpret the location as a source-receiver offset.

        Returns
        -------
        bool

        Notes
        -----
        This is currently only implemented for the 1D layered code.
        """
        return self._use_source_receiver_offset

    @use_source_receiver_offset.setter
    def use_source_receiver_offset(self, val):
        self._use_source_receiver_offset = validate_type(
            "use_source_receiver_offset", val, bool
        )

    @property
    def bw_cutoff_frequency(self):
        """Butter worth low pass filter

        Returns
        -------
        numpy.ndarray
            Butter worth low pass filter
        """
        return self._bw_cutoff_frequency

    @bw_cutoff_frequency.setter
    def bw_cutoff_frequency(self, var):
        self._bw_cutoff_frequency = validate_float(
            "bw_cutoff_frequency", var, min_val=0.0
        )

    @property
    def lp_cutoff_frequency(self):
        """Low pass filter

        Returns
        -------
        numpy.ndarray
            Low pass filter
        """
        return self._lp_cutoff_frequency

    @lp_cutoff_frequency.setter
    def lp_cutoff_frequency(self, var):
        self._lp_cutoff_frequency = validate_float(
            "lp_cutoff_frequency", var, min_val=0.0
        )

    @property
    def bw_power(self):
        """Butter worth low pass filter

        Returns
        -------
        numpy.ndarray
            Butter worth low pass filter
        """
        return self._bw_power

    @bw_power.setter
    def bw_power(self, var):
        self._bw_power = validate_float("bw_power", var, min_val=0.0, max_val=2)

    @property
    def lp_power(self):
        """Low pass filter

        Returns
        -------
        numpy.ndarray
            Low pass filter
        """
        return self._lp_power

    @lp_power.setter
    def lp_power(self, var):
        self._lp_power = validate_float("lp_power", var, min_val=0.0, max_val=0.99999)

    @property
    def lp_cutoff_frequency_1(self):
        """First receiver low-pass filter cutoff frequency.

        Returns
        -------
        float
            Cutoff frequency [Hz] of the first receiver low-pass filter.
        """
        return self._lp_cutoff_frequency_1

    @lp_cutoff_frequency_1.setter
    def lp_cutoff_frequency_1(self, var):
        self._lp_cutoff_frequency_1 = validate_float(
            "lp_cutoff_frequency_1", var, min_val=0.0
        )

    @property
    def lp_power_1(self):
        """First receiver low-pass filter order, read as a GEX filter order.

        ``0`` (the default) is no filter, so a receiver that never sets it is
        unaffected. An integer n of 1 or more is a Butterworth filter of order
        n, and a value between 0 and 1 a Gaussian-RLC filter with that
        damping; see :func:`gex_low_pass_response`.

        Returns
        -------
        float
            Order of the first receiver low-pass filter [dimensionless].
        """
        return self._lp_power_1

    @lp_power_1.setter
    def lp_power_1(self, var):
        self._lp_power_1 = _validate_filter_order("lp_power_1", var)

    @property
    def lp_cutoff_frequency_2(self):
        """Second receiver low-pass filter cutoff frequency.

        Returns
        -------
        float
            Cutoff frequency [Hz] of the second receiver low-pass filter.
        """
        return self._lp_cutoff_frequency_2

    @lp_cutoff_frequency_2.setter
    def lp_cutoff_frequency_2(self, var):
        self._lp_cutoff_frequency_2 = validate_float(
            "lp_cutoff_frequency_2", var, min_val=0.0
        )

    @property
    def lp_power_2(self):
        """Second receiver low-pass filter order, read as a GEX filter order.

        ``0`` (the default) is no filter. Otherwise as ``lp_power_1``; see
        :func:`gex_low_pass_response`.

        Returns
        -------
        float
            Order of the second receiver low-pass filter [dimensionless].
        """
        return self._lp_power_2

    @lp_power_2.setter
    def lp_power_2(self, var):
        self._lp_power_2 = _validate_filter_order("lp_power_2", var)

    @property
    def gate_windows(self):
        """Open and close time of each gate, or None to sample at ``times``.

        With windows, the 1D layered simulation returns each gate as the
        average of the response over [open, close], by Gauss-Legendre
        quadrature with ``gate_window_nodes`` nodes. AarhusInv does the same
        when a gate's open and close times are given (AarhusInv manual,
        version 8, p. 56). A window of zero width is the same as sampling at
        its time. Other simulations ignore the windows.

        Returns
        -------
        (n_times, 2) numpy.ndarray or None
            Open and close time [s] of each gate, on the same time axis as
            ``times``, one row per entry of ``times``.
        """
        return self._gate_windows

    @gate_windows.setter
    def gate_windows(self, var):
        if var is None:
            self._gate_windows = None
            return
        var = validate_ndarray_with_shape(
            "gate_windows", var, shape=(len(self.times), 2), dtype=float
        )
        if np.any(var[:, 1] < var[:, 0]):
            raise ValueError(
                "'gate_windows' must close at or after they open, got "
                f"{var[var[:, 1] < var[:, 0]].tolist()}"
            )
        self._gate_windows = var

    @property
    def gate_window_nodes(self):
        """Number of Gauss-Legendre nodes per gate window.

        Five nodes average a SkyTEM gate window to within 1e-6 of a fine
        Simpson rule at the earliest kept gates.

        Returns
        -------
        int
            Nodes per gate window [dimensionless], 1 or more; 1 samples the
            window's midpoint.
        """
        return self._gate_window_nodes

    @gate_window_nodes.setter
    def gate_window_nodes(self, var):
        self._gate_window_nodes = validate_integer("gate_window_nodes", var, min_val=1)

    def getSpatialP(self, mesh, f):
        """Get spatial projection matrix from mesh to receivers.

        Only constructed when called.

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh
        f : simpeg.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        P = Zero()
        field = f._GLoc(self.projField)
        for strength, comp in zip(self.orientation, ["x", "y", "z"]):
            if strength != 0.0:
                P = P + strength * mesh.get_interpolation_matrix(
                    self.locations, field + comp
                )
        return P

    def getTimeP(self, time_mesh, f):
        """Get time projection matrix from mesh to receivers.

        Only constructed when called.

        Parameters
        ----------
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : simpeg.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        projected_time_grid = f._TLoc(self.projField)
        return time_mesh.get_interpolation_matrix(self.times, projected_time_grid)

    def getP(self, mesh, time_mesh, f):
        """Returns projection matrices as a list for all components collected by the receivers.

        Parameters
        ----------
        mesh : discretize.BaseMesh
            A discretize mesh defining spatial discretization
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : simpeg.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            Returns full projection matrix from fields to receivers.

        Notes
        -----
        Projection matrices are stored as a dictionary (mesh, time_mesh) if storeProjections is True
        """
        if (mesh, time_mesh) in self._Ps:
            return self._Ps[(mesh, time_mesh)]

        Ps = self.getSpatialP(mesh, f)
        Pt = self.getTimeP(time_mesh, f)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, time_mesh)] = P

        return P

    def eval(self, src, mesh, time_mesh, f):  # noqa: A003
        """Project fields to receivers to get data.

        Parameters
        ----------
        src : simpeg.electromagnetics.frequency_domain.sources.BaseTDEMSrc
            A time-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : simpeg.electromagnetic.time_domain.fields.FieldsTDEM
            The solution for the fields defined on the mesh

        Returns
        -------
        numpy.ndarray
            Fields projected to the receiver(s)
        """
        P = self.getP(mesh, time_mesh, f)
        f_part = mkvc(f[src, self.projField, :])
        return P * f_part

    def evalDeriv(self, src, mesh, time_mesh, f, v, adjoint=False):
        """Derivative of projected fields with respect to the inversion model times a vector.

        Parameters
        ----------
        src : simpeg.electromagnetics.frequency_domain.sources.BaseTDEMSrc
            A time-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : simpeg.electromagnetic.time_domain.fields.FieldsTDEM
            The solution for the fields defined on the mesh
        v : numpy.ndarray
            A vector
        adjoint : bool, default = ``False``
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            derivative of fields times a vector projected to the receiver(s)
        """
        P = self.getP(mesh, time_mesh, f)

        if not adjoint:
            return P * v
        elif adjoint:
            # dP_dF_T = P.T * v #[src, self]
            # newshape = (len(dP_dF_T)/time_mesh.nN, time_mesh.nN )
            return P.T * v  # np.reshape(dP_dF_T, newshape, order='F')


class PointElectricField(BaseRx):
    """Measure TDEM electric field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="z", **kwargs):
        self.projField = "e"
        super(PointElectricField, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFluxDensity(BaseRx):
    """Measure TDEM magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="z", **kwargs):
        self.projField = "b"
        super(PointMagneticFluxDensity, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFluxTimeDerivative(BaseRx):
    """Measure time-derivative of magnetic flux density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="z", **kwargs):
        self.projField = "dbdt"
        super(PointMagneticFluxTimeDerivative, self).__init__(
            locations, times, orientation, **kwargs
        )

    def eval(self, src, mesh, time_mesh, f):  # noqa: A003
        """Project solution of fields to receivers to get data.

        Parameters
        ----------
        src : simpeg.electromagnetics.frequency_domain.sources.BaseTDEMSrc
            A time-domain EM source
        mesh : discretize.base.BaseMesh
            The mesh on which the discrete set of equations is solved
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : simpeg.electromagnetic.time_domain.fields.FieldsTDEM
            The solution for the fields defined on the mesh

        Returns
        -------
        numpy.ndarray
            Fields projected to the receiver(s)
        """

        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).eval(
                src, mesh, time_mesh, f
            )

        P = self.getP(mesh, time_mesh, f)
        f_part = mkvc(f[src, "b", :])
        return P * f_part

    def getTimeP(self, time_mesh, f):
        """Get time projection matrix from mesh to receivers.

        Only constructed when called.

        Parameters
        ----------
        time_mesh : discretize.TensorMesh
            A 1D ``TensorMesh`` defining the time discretization
        f : simpeg.electromagnetics.time_domain.fields.FieldsTDEM

        Returns
        -------
        scipy.sparse.csr_matrix
            P, the interpolation matrix
        """
        if self.projField in f.aliasFields:
            return super(PointMagneticFluxTimeDerivative, self).getTimeP(time_mesh, f)

        return (
            time_mesh.get_interpolation_matrix(self.times, "CC")
            * time_mesh.face_divergence
        )


class PointMagneticField(BaseRx):
    """Measure TDEM magnetic field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "h"
        super(PointMagneticField, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointCurrentDensity(BaseRx):
    """Measure TDEM current density at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "j"
        super(PointCurrentDensity, self).__init__(
            locations, times, orientation, **kwargs
        )


class PointMagneticFieldTimeDerivative(BaseRx):
    """Measure time-derivative of magnet field at a point.

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    times : (n_times) numpy.ndarray
        Time channels
    orientation : {'z', 'x', 'y'}
        Receiver orientation.
    """

    def __init__(self, locations=None, times=None, orientation="x", **kwargs):
        self.projField = "dhdt"
        super(PointMagneticFieldTimeDerivative, self).__init__(
            locations, times, orientation, **kwargs
        )
