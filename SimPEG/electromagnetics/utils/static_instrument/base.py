import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer
import libaarhusxyz
import pandas as pd

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import os, tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh, SimplexMesh

from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
import SimPEG.electromagnetics.utils.em1d_utils
from SimPEG.electromagnetics.utils.em1d_utils import get_2d_mesh,plot_layer, get_vertical_discretization_time
from SimPEG.regularization import LaterallyConstrained, RegularizationMesh

from .thickness import build_log_spaced_layer_thick
from .utils import detect_cpu_availability

import warnings

try:
    from pymatsolver import PardisoSolver as Solver
    print("pymatsolver.PardisoSolver available for fwd modelling")
except:
    print("Could not import PardisoSolver, only default (spLU) available")
    
import scipy.stats
import copy
import re
import typing

from . import xyzfilter

class XYZSystem(object):
    """This is a base class for system descriptions for moving EM
    acquisition platforms such as AEM (aerial EM), TTEM (towed time
    domain EM). The base assumption and simplification provided by
    this class is that the setup of receiver(s) and transmitter(s) is
    independent of the data, save for their absolute positions (but
    relative positions are still independent from data).

    Each subclass of this class, describes a particular setup of
    transmitters, receivers including dipole moments, waveforms,
    positions etc, as well as inversion parameters.

    A subclass can then be instantiated together with an XYZ file
    structure with raw data read using libaarhusxyz.XYZ(), to form an
    invertible object, or with a model read using the same library to
    do forward modelling.

    Basic usage:

    ```
    class MySystem(XYZSystem):
        def make_system(self, idx, location, times):
            # Your code here

    inv = MySystem(libaarhusxyz.XYZ("measured.xyz"))
    sparse, l2 = inv.invert()
    sparse.dump("sparse.xyz")
    l2.dump("l2.xyz")
    ```

    Not that any class level attribute, such as `startmodel__n_layer`, can be
    overridden by a parameter when instantiating the class, e.g. 

    ```
    MySystem(libaarhusxyz.XYZ("measured.xyz"), startmodel__n_layer=10)
    ```
    """
    
    
    def __init__(self, xyz, **kw):
        self._xyz = xyz
        self.options = kw
        if self.validate:
            self.do_validate()

    validate = True
    "Validate input data scaling etc. prior to inversion"
    def do_validate(self):
        if "dbdt_ch1gt" in self._xyz.layer_data:
            dbdt = -self._xyz.layer_data["dbdt_ch1gt"].values.flatten() * self._xyz.model_info.get("scalefactor", 1)
            assert np.nanmean(dbdt) < 1e-3, "Unit for dbdt is probably wrong. Please set scalefactor."
        
    def __getattribute__(self, name):
        options = object.__getattribute__(self, "options")
        if name in options: return options[name]
        return object.__getattribute__(self, name)


    sounding_filter = slice(None, None, None)

    @property
    def moment_channels(self):
        """The channel numbers this system models, in ``times_full`` order.

        The contract for subclasses: ``moment_channels[i]`` is the instrument
        channel whose gates ``times_full[i]`` and ``times_filter[i]`` describe.
        That pairing is what lets :attr:`gate_filter` turn a channel number into
        a moment index without assuming the two are the same number.

        This base description has a single channel, numbered 1 — the same
        assumption ``times_full`` makes when it reads ``'gate times for channel
        1'``. A subclass describing more than one channel, or channels numbered
        otherwise, must override this; see
        :attr:`~.dual.DualMomentTEMXYZSystem.moment_channels`, which resolves
        the pair from the instrument description rather than declaring it.
        """
        return (1,)

    #: Matches the channel number in a ``layer_data`` array name.
    #:
    #: Anchored on the ``ch`` token deliberately. Taking the first run of digits
    #: instead reads ``01`` out of a name like ``HM_X_G01`` and calls it channel
    #: 1, which is how an array belonging to no channel at all used to be handed
    #: another channel's gate filter.
    _channel_pattern = re.compile(r"ch(\d+)", re.IGNORECASE)

    @classmethod
    def layer_data_channel(cls, key):
        """The channel a ``layer_data`` array name declares, or None for none.

        Covers both namings in circulation for the same array — ``dbdt_ch1gt``
        and ``Gate_Ch01`` — since which one reaches here depends on the naming
        standard the file was normalized to. Override alongside
        :attr:`moment_channels` for an instrument naming its arrays some third
        way.
        """
        match = cls._channel_pattern.search(key)
        return None if match is None else int(match.group(1))

    @property
    def gate_filter(self):
        """Per-array gate masks, keyed by ``layer_data`` array name.

        Built from the channels this system declares. An array is filtered only
        where its name names a channel in :attr:`moment_channels`; the mask it
        gets is the one for that channel's *moment*, found by position in the
        pair rather than by subtracting one from the channel number. Those
        coincide only on an instrument numbering its moments 1 and 2, and
        assuming it raised ``IndexError`` on anything else.

        Arrays left out are not dropped. ``FilteredXYZ`` reads this dict with
        ``.get(key, None)`` and falls back to a full slice, so an absent key
        means "all gates kept" — the array still passes through the sounding
        filter and out the other side, simply un-narrowed. That is the only
        defensible default: applying one channel's gate mask to an array of
        another channel's width is what produced the shape errors this replaces.

        A name carrying a channel number outside the declared set is reported,
        because that is unambiguously measured data the inversion is about to
        ignore. A name carrying no channel number at all is passed over in
        silence: at this level ``resistivity``, ``dep_top`` and ``dep_bot`` are
        indistinguishable from an auxiliary component the GEX omits, and warning
        about all of them would fire on every dataset and teach the operator to
        ignore the warning that matters.
        """
        moment_of_channel = {channel: moment
                             for moment, channel in enumerate(self.moment_channels)}
        filt = {}
        undeclared = []
        for key in self._xyz.layer_data.keys():
            channel = self.layer_data_channel(key)
            if channel is None:
                continue
            if channel not in moment_of_channel:
                undeclared.append(key)
                continue
            filt[key] = self.times_filter[moment_of_channel[channel]]
        if undeclared:
            warnings.warn(
                "Layer data %s belongs to channels this system does not model "
                "(modeling channel(s) %s). Those arrays are carried through "
                "unfiltered and take no part in the inversion."
                % (", ".join(sorted(undeclared)),
                   ", ".join(str(c) for c in self.moment_channels)))
        return filt

    @property
    def xyz(self):
        return xyzfilter.FilteredXYZ(self._xyz, self.sounding_filter, self.gate_filter)
    
    def make_system(self, idx, location, times):
        """This method should return a list of instances of some
        SimPEG.survey.BaseSrc subclass, such as
        SimPEG.electromagnetics.time_domain.sources.MagDipole.

        idx is an index into self.xyz.flightlines
        location is a tuple (x, y, z) corresponding to the coordinates
            found at that index in self.xyz.flightlines
        times is whatever is returned by self.times, typically a list
            of gate times, or for a multi channel system, a tuple of
            such lists, one for each channel.
        """
        raise NotImplementedError("You must subclass XYZInversion and override make_system() with your own method!")

    @property
    def times_full(self):
        """Every gate time the instrument records, before any gate filtering.

        Read off the unfiltered dataset. Reading ``self.xyz`` instead closes a
        loop — the filtered view is built from :attr:`gate_filter`, which needs
        :attr:`times_filter`, which needs this — so any subclass not overriding
        this property recursed to death the moment anything touched its data.
        The two are interchangeable in every other respect: filtering narrows
        ``layer_data`` and ``flightlines``, and leaves ``model_info`` alone.
        """
        return [np.array(self._xyz.model_info['gate times for channel 1'])]

    @property
    def times_filter(self):
        return [np.ones(len(times), dtype=bool) for times in self.times_full]
    
    @property
    def times(self):
        return [times_full if times_filter is None else times_full[times_filter]
                for times_full, times_filter
                in zip(self.times_full, self.times_filter)]
    
    startmodel__n_layer = 30
    "Number of 1D model layers per sounding. More layers give finer depth resolution but increase computation time. Typical range: 20–35. Depth extent is controlled by 'top_depth_last_layer'."
    @property
    def n_layer_used(self):
        if "resistivity" in self.xyz.layer_data:
            return self.xyz.resistivity.shape[1]
        return self.startmodel__n_layer
    
    @property
    def data_array_nan(self):
        return self.xyz.dbdt_ch1gt.values.flatten()

    @property
    def data_array(self):
        dobs = self.data_array_nan
        return np.where(np.isnan(dobs), 9999., dobs)
    
    @property
    def data_uncert_array(self):
        return self.xyz.dbdt_std_ch1gt.values.flatten()

    @property
    def data_uncert_array_culled(self):
        dobs = self.data_array_nan
        return np.where(np.isnan(dobs) | np.isnan(self.data_uncert_array), np.inf, self.data_uncert_array)

    dipole_moments = [1]
    
    uncertainties__std_data = 0.03
    "Minimum relative noise floor as a fraction of data amplitude (e.g. 0.03 = 3%). When measured stacking noise is lower than this value, this floor is used instead. Prevents overfitting in low-noise windows. Typical range: 0.02–0.10."
    uncertainties__std_data_override = False
    "If true, ignore per-sounding noise from stacking and apply 'std_data' uniformly to all soundings. Use when data lacks measured STD (e.g. forward model output), or to impose a uniform noise floor across the survey."
    uncertainties__noise_level_1ms = 1e-9
    "Absolute noise floor amplitude at 1 ms gate time (V/Am²). Sets a practical lower bound on uncertainty for early-time gates. Scales with gate time as noise_level_1ms × (t × 1000)^noise_exponent. Typical range: 1e-13 (quiet system) to 1e-9 (noisy). Check system specs or late-time noise in your data."
    uncertainties__noise_exponent = -0.5
    "Power-law time exponent for the noise floor decay. Default -0.5 means noise scales as t^(-0.5), a common approximation for AEM systems. Combined with 'noise_level_1ms' to form the time-varying noise floor: N(t) = noise_level_1ms × (t × 1000)^noise_exponent."
    @property
    def uncert_array(self):
        n_sounding = len(self.xyz.flightlines)
        
        # 1e3 to compensate for noise level being at 1 millisecond
        noise = np.hstack([np.tile((times*1e3)**self.uncertainties__noise_exponent
                                   * (self.uncertainties__noise_level_1ms / moment),
                                   (n_sounding, 1))
                           for times, moment in zip(self.times, self.dipole_moments)]).flatten()

        if not self.uncertainties__std_data_override:
            stds = np.where(self.data_uncert_array_culled < self.uncertainties__std_data,
                            self.uncertainties__std_data,
                            self.data_uncert_array_culled)
            uncertainties = stds * np.abs(self.data_array_nan) + noise
        else:
            uncertainties = self.uncertainties__std_data*np.abs(self.data_array_nan) + noise
        
        return np.where(np.isnan(self.data_array_nan), np.inf, uncertainties)

    startmodel__thicknesses_type: typing.Literal['logspaced', 'geometric', 'time'] = "logspaced"
    "Layer thickness scheme. 'logspaced': layers increase in thickness logarithmically from top to bottom — recommended for most AEM surveys. 'geometric': each layer is a fixed ratio thicker than the one above (set ratio with 'thicknesses_geometric_factor'). 'time': layer boundaries are scaled to gate times (good for data-adaptive depth discretization)."
    startmodel__thicknesses_minimum_dz = 1
    "Thickness of the shallowest layer (m). Controls near-surface resolution. Used by 'logspaced' and 'geometric' thickness schemes. Typical: 1–5 m for shallow targets, 5–10 m for deep regional surveys."
    startmodel__top_depth_last_layer = 400
    "Depth to the top of the deepest layer (m), used by the 'logspaced' scheme. Should match the approximate depth of investigation (DOI) for the survey. Typical AEM DOI: 100–500 m depending on system moment and ground conductivity. Setting this too deep wastes model parameters on unresolved depths."
    startmodel__thicknesses_geomtric_factor = 1.15309
    "Layer thickness ratio for the 'geometric' scheme — each layer is this factor thicker than the one above. Default 1.153 gives approximately log-spaced layers. Increase for faster depth growth; decrease for more uniform thickness."

    def make_thicknesses(self):
        # If we already have thicknesses because input is a model, don't deviate from that
        if "dep_top" in self.xyz.layer_params:
            return np.diff(self.xyz.layer_params["dep_top"].values)
        if self.startmodel__thicknesses_type == "logspaced":
            thk = build_log_spaced_layer_thick(first_thk=self.startmodel__thicknesses_minimum_dz,
                                               last_dep_top=self.startmodel__top_depth_last_layer,
                                               numlay=self.n_layer_used)
            # print(thk)
            return thk
        elif self.startmodel__thicknesses_type == "geometric":
            return SimPEG.electromagnetics.utils.em1d_utils.get_vertical_discretization(self.n_layer_used - 1,
                                                                                        self.startmodel__thicknesses_minimum_dz,
                                                                                        self.startmodel__thicknesses_geomtric_factor)
        elif self.startmodel__thicknesses_type == "time":
            # FIX ME: if model is given it should use the resistivities in the model, not self.startmodel__res
            return SimPEG.electromagnetics.utils.em1d_utils.get_vertical_discretization_time(
                np.sort(np.concatenate(self.times)),
                sigma_background=1./self.startmodel__res,
                n_layer=self.n_layer_used-1
            )
        else:
            raise Exception("unknown thickness type")

    def make_survey(self):
        times = self.times
        xyz = self.xyz
        systems = [
            self.make_system(
                idx,
                xyz.flightlines.loc[
                    idx, [xyz.x_column, xyz.y_column, xyz.alt_column]
                ].astype(float).values,
                times)
            for idx in range(0, len(xyz.flightlines))]
        return tdem.Survey([
            source
            for sources in systems
            for source in sources])

    def n_param(self, thicknesses):
        return (len(thicknesses)+1)*len(self.xyz.flightlines)
    
    simulation__solver : typing.Literal['LU', 'pardiso'] = 'LU'
    "Linear solver backend for the forward simulation. 'LU' uses scipy sparse LU decomposition (default, no extra dependencies). 'pardiso' uses Intel MKL Pardiso via pymatsolver — significantly faster for large problems but requires compatible hardware and the pymatsolver package."
    simulation__parallel = True
    "Run forward simulations for each sounding in parallel. Strongly recommended for production runs. Set to False only for single-threaded debugging in a notebook."
    simulation__n_cpu = 0
    "Number of CPU threads for parallel simulation. Default 0 means auto-detect from the pod's CPU limit (CPU_LIMIT env var, then cgroup CFS quota, then node core count); None is treated the same way. A positive value pins that many worker processes verbatim (explicit override always wins). Increasing beyond the number of physical cores gives diminishing returns."
    def make_simulation(self, survey, thicknesses):
        n_cpu = self.simulation__n_cpu
        if n_cpu is None or n_cpu == 0:
            n_cpu = detect_cpu_availability()
        if 'pardiso' in self.simulation__solver.lower():
            print('Using Pardiso solver')
            return tdem.Simulation1DLayeredStitched(
                survey=survey,
                thicknesses=thicknesses,
                sigmaMap=maps.ExpMap(nP=self.n_param(thicknesses)),
                solver=PardisoSolver,
                parallel=self.simulation__parallel,
                n_cpu=n_cpu,
                n_layer=self.n_layer_used)
        else:
            print('Using default (spLU) solver')
            return tdem.Simulation1DLayeredStitched(
                survey=survey,
                thicknesses=thicknesses,
                sigmaMap=maps.ExpMap(nP=self.n_param(thicknesses)),
                parallel=self.simulation__parallel,
                n_cpu=n_cpu,
                n_layer=self.n_layer_used)

    
    def make_data(self, survey):
        return data.Data(
            survey,
            dobs=self.data_array,
            standard_deviation=self.uncert_array)
    
    def make_misfit_weights(self):
        return 1./self.uncert_array

    def make_misfit(self, thicknesses):
        survey = self.make_survey()

        dmis = data_misfit.L2DataMisfit(
            simulation=self.make_simulation(survey, thicknesses),
            data=self.make_data(survey))
        dmis.W = self.make_misfit_weights()
        return dmis
    
    startmodel__res=100.
    "Uniform starting resistivity (Ω·m). All soundings begin from a homogeneous halfspace at this value. Should be a reasonable estimate of the background resistivity — a poor choice increases iteration count. Typical values: 10 Ω·m (conductive settings, e.g. saline groundwater), 100 Ω·m (moderate), 1000 Ω·m (resistive, e.g. crystalline rock or dry alluvium)."
    def make_startmodel(self, thicknesses):
        startmodel=np.log(np.ones(self.n_param(thicknesses)) * 1/self.startmodel__res)
        return startmodel

    regularization__alpha_s = 1e-4
    """Smallness weight — anchors every model cell toward the reference resistivity (startmodel__res).

    Unlike alpha_r and alpha_z, which enforce smoothness between neighboring cells, alpha_s acts
    globally: it adds a cost for any cell that deviates from the reference model, regardless of its
    neighbors. This is important in poorly-sampled regions (deep layers, survey edges) where the
    data have little sensitivity — without alpha_s the model is free to drift to arbitrary values in
    those regions.

    IMPORTANT: alpha_s does NOT control lateral resolution. To suppress short-wavelength lateral
    structure, increase alpha_r instead. alpha_s only controls how strongly the whole model is pulled
    toward the starting resistivity.

    Scaling: alpha_s ≈ 1/h², where h is the geometric mean of sounding spacing and line spacing.
    This keeps alpha_s numerically comparable to the smoothness terms (alpha_r, alpha_z = 1.0 by default):
      - 25 m sounding / 100 m line spacing  →  h = 50 m  →  alpha_s ≈ 4e-4
      - 30 m sounding / 100 m line spacing  →  h = 55 m  →  alpha_s ≈ 3e-4
      - 100 m sounding / 100 m line spacing →  h = 100 m →  alpha_s ≈ 1e-4  (default)
      - 25 m sounding / 400 m line spacing  →  h = 100 m →  alpha_s ≈ 1e-4

    Too large: resistivity contrasts are suppressed and the model sits near startmodel__res everywhere.
    Too small: the model drifts freely in low-sensitivity regions, producing depth artifacts or
    spurious structure between flight lines.
    """
    regularization__alpha_r = 1.
    """Lateral smoothness weight — penalizes resistivity differences between neighboring soundings.

    This is the primary control on effective lateral resolution. The inversion links soundings via
    a Delaunay triangulation of their positions, and alpha_r scales how strongly adjacent soundings
    are pulled toward each other.

    To target an effective lateral resolution D with sounding spacing d, set:
        alpha_r ≈ (D / d)² × alpha_z
    For example, with 30 m sounding spacing and a target resolution of ~100 m:
        alpha_r ≈ (100/30)² × 1.0 ≈ 10

    alpha_r has no effect on vertical structure within a sounding — use alpha_z for that.
    The default (1.0, isotropic with alpha_z) is appropriate when the sounding spacing already
    matches your desired lateral resolution. Increase alpha_r for laterally continuous geology
    (e.g., sedimentary basins); keep it low if you expect sharp lateral boundaries.
    """
    regularization__alpha_z = 1.
    """Vertical smoothness weight — penalizes resistivity differences between adjacent layers within each sounding.

    Controls how sharply resistivity is allowed to change with depth. The ratio alpha_z / alpha_r
    sets the anisotropy of the regularization:
      - alpha_z = alpha_r (default): isotropic — lateral and vertical gradients are penalized equally.
      - alpha_z > alpha_r: enforces stronger vertical smoothness, useful for environments with
        well-defined horizontal layering and little lateral variation.
      - alpha_z < alpha_r: allows more vertical structure while enforcing lateral continuity, useful
        when you expect sharp boundaries at depth but a laterally uniform stratigraphy.

    Note that the absolute values of alpha_r and alpha_z matter less than their ratio — doubling
    both changes nothing about the model shape, only about how alpha_s balances against smoothness.
    """

    def make_regularization(self, thicknesses):
        if False:
            assert False, "LCI is currently broken"
            hz = np.r_[thicknesses, thicknesses[-1]]
            reg = LaterallyConstrained(
                get_2d_mesh(len(self.xyz.flightlines), hz),
                mapping=maps.IdentityMap(nP=self.n_param(thicknesses)),
                alpha_s = self.regularization__alpha_s,
                alpha_r = self.regularization__alpha_r,
                alpha_z = self.regularization__alpha_z)
            # reg.get_grad_horizontal(self.xyz.flightlines[["x", "y"]], hz, dim=2, use_cell_weights=True)
            # ps, px, py = 0, 0, 0
            # reg.norms = np.c_[ps, px, py, 0]
            reg.mref = self.make_startmodel(thicknesses)
            # reg.mrefInSmooth = False
            return reg
        else:
            coords = self.xyz.flightlines[[self.xyz.x_column, self.xyz.y_column]].astype(float).values
            if np.sum(np.abs(np.diff(coords[:,1]))) == 0:
                print('y-coordinate seems to be constant (synthetic data?), adding a small random number')
                coords[:,1] += np.random.randn(len(coords)) * 1e-6
            tri = Delaunay(coords)
            hz = np.r_[thicknesses, thicknesses[-1]]

            mesh_radial = SimplexMesh(tri.points, tri.simplices)
            mesh_vertical = SimPEG.electromagnetics.utils.em1d_utils.set_mesh_1d(hz)
            mesh_reg = [mesh_radial, mesh_vertical]
            n_param = int(mesh_radial.n_nodes * mesh_vertical.nC)
            reg_map = SimPEG.maps.IdentityMap(nP=n_param)    # Mapping between the model and regularization
            reg = SimPEG.regularization.LaterallyConstrained(
                mesh_reg, mapping=reg_map,
                alpha_s = self.regularization__alpha_s,
                alpha_r = self.regularization__alpha_r,
                alpha_z = self.regularization__alpha_z,
            )
            reg.mref = self.make_startmodel(thicknesses)
            return reg

    directives__beta__seed : int = 42
    "Random seed for the beta estimator. Fixed value ensures reproducible results across runs with identical parameters. Change to any integer for a different (but still reproducible) initialization, or clear to use a random seed each run."
    directives__beta__beta0_ratio : float = 10.
    "Initial regularization strength as a multiple of the estimated optimal beta. Higher values (10–100) start with a heavily smoothed model and relax regularization gradually — this is the standard Tikhonov approach and typically converges in 20–30 iterations. Values near 1 give the data too much control immediately, leading to slow or erratic convergence. Recommended: 10–50."
    directives__beta__cooling_factor=2
    "Factor by which the regularization weight (beta) is divided at each cooling step. Default 2 halves beta each step. Larger values (4–10) cool faster and may converge in fewer iterations but risk overshooting the data misfit target."
    directives__beta__cooling_rate=1
    "Number of Gauss-Newton outer iterations between each beta cooling step. Default 1 cools every iteration. Increase to 2–3 if the inversion is oscillating or if you want more iterations at each regularization level before reducing it."
    directives__irls__enable = False
    "Enable sparse (IRLS) inversion after the smooth L2 model converges. IRLS produces a model with sharper layer boundaries by iteratively reweighting the regularization. The smooth L2 model is always produced first and saved regardless."
    directives__irls__max_iterations = 30
    "Maximum IRLS iterations after L2 convergence. Each IRLS iteration updates the reweighting and re-inverts. Typical: 10–30."
    directives__irls__minGNiter = 1
    "Minimum Gauss-Newton iterations per IRLS step before the reweighting is updated. Default 1. Increase to 2–3 for more stable IRLS convergence."
    directives__irls__fix_Jmatrix = True
    "Fix the sensitivity matrix (Jacobian) during IRLS iterations. True is faster (avoids recomputing sensitivities) and recommended for most cases. Set False only if the model changes substantially between IRLS iterations."
    directives__irls__f_min_change = 1e-3
    "IRLS convergence tolerance — minimum fractional change in the objective function between iterations. Smaller values require tighter convergence before stopping."
    directives__irls__coolingRate = 1
    "Number of IRLS iterations between each update of the IRLS reweighting factors. Default 1 updates every iteration."
    def make_directives(self):
        if self.directives__beta__seed:
            BetaEstimate = directives.BetaEstimate_ByEig(beta0_ratio=self.directives__beta__beta0_ratio, 
                                                         seed=self.directives__beta__seed)
            print('setting manual random seed for repeatabillity')
        else:
            BetaEstimate = directives.BetaEstimate_ByEig(beta0_ratio=self.directives__beta__beta0_ratio)
        dirs = [
            BetaEstimate,
            SimPEG.directives.BetaSchedule(coolingFactor=self.directives__beta__cooling_factor, 
                                           coolingRate=self.directives__beta__cooling_rate),
            SimPEG.directives.TargetMisfit()]

        #            directives.SaveOutputEveryIteration(save_txt=False),
        if self.directives__irls__enable:
            dirs.append(
                directives.Update_IRLS(
                    max_irls_iterations = self.directives__irls__max_iterations,
                    minGNiter = self.directives__irls__minGNiter,
                    fix_Jmatrix = self.directives__irls__fix_Jmatrix,
                    f_min_change = self.directives__irls__f_min_change,
                    coolingRate = self.directives__irls__coolingRate))
            dirs.append(directives.UpdatePreconditioner())

        return dirs
        
    optimizer__max_iter=50
    "Maximum number of Gauss-Newton outer iterations. The inversion will stop early if the TargetMisfit directive is satisfied (data fit is good enough). With beta0_ratio=10 and cooling_rate=1, convergence in 20–35 iterations is typical. Increase to 60–80 only if the inversion is still improving at the limit."
    optimizer__max_iter_cg=20
    "Maximum conjugate gradient (CG) iterations for the inner linear solve at each Gauss-Newton step. Increase if you see poor model updates per outer iteration, which can indicate a poorly conditioned problem. Default 20 is sufficient for most AEM problems."
    def make_optimizer(self):
        return optimization.InexactGaussNewton(maxIter = self.optimizer__max_iter, maxIterCG=self.optimizer__max_iter_cg)
    
    def make_inversion(self):
        thicknesses = self.make_thicknesses()

        return inversion.BaseInversion(
            inverse_problem.BaseInvProblem(
                self.make_misfit(thicknesses),
                self.make_regularization(thicknesses),
                self.make_optimizer()),
            self.make_directives())

    def make_forward(self):
        return self.make_simulation(self.make_survey(), self.make_thicknesses())
        
    def inverted_model_to_xyz(self, model, thicknesses):
        xyzsparse = libaarhusxyz.XYZ()
        xyzsparse.model_info.update(self.xyz.model_info)
        xyzsparse.flightlines = self.xyz.flightlines
        xyzsparse.layer_data["resistivity"] = 1 / np.exp(pd.DataFrame(
            model.reshape((len(self.xyz.flightlines),
                           len(model) // len(self.xyz.flightlines)))))

        dep_top = np.cumsum(np.concatenate(([0], thicknesses)))
        dep_bot = np.concatenate((dep_top[1:], [np.inf]))

        xyzsparse.layer_data["dep_top"] = pd.DataFrame(np.meshgrid(dep_top, self.xyz.flightlines.index)[0])
        xyzsparse.layer_data["dep_bot"] = pd.DataFrame(np.meshgrid(dep_bot, self.xyz.flightlines.index)[0])

        return self.xyz.unfilter(xyzsparse, layerfilter=False)
    
    def invert(self, **kw):
        """Invert the data from the XYZ file using this system description and
        inversion parameters.

        Returns a sparse model and an l2 (smooth model), both in xyz format.
        """

        self.options.update(kw)
        
        self.inv = self.make_inversion()        
        self.inv.run(self.make_startmodel(self.inv.invProb.dmisfit.simulation.thicknesses))
        self.make_inversion_outputs()
        return self.sparse, self.l2
    
    def make_inversion_outputs(self):
        last_model = self.inverted_model_to_xyz(self.inv.invProb.model, self.inv.invProb.dmisfit.simulation.thicknesses)
        last_pred = self.forward_data_to_xyz(self.inv.invProb.dpred, inversion=True)

        self.corrected = self.forward_data_to_xyz(self.inv.invProb.dmisfit.data.dobs, inversion=True)

        if hasattr(self.inv.invProb, "l2model"):
            self.sparse = last_model
            self.sparsepred = last_pred
            self.l2 = self.inverted_model_to_xyz(self.inv.invProb.l2model, self.inv.invProb.dmisfit.simulation.thicknesses)
            self.l2pred = self.forward_data_to_xyz(self.inv.invProb.l2dpred, inversion=True)

        else:
            self.sparse = None
            self.sparsepred = None
            self.l2 = last_model
            self.l2pred = last_pred

    def split_moments(self, resp):
        moments = []
        pos = 0
        for times in self.times:
            moments.append(resp[:,pos:pos+len(times)])
            pos += len(times)
        return moments

    def pad_times(self, xyz, times, positions):
        """Pad data in xyz with NaN:s, to have the list of gate times be
        times. times must be a superset of the times already present
        for each moment. positions must be the positions in times
        where the existing times in xyz are located.

        """
        
        new_xyz = copy.deepcopy(xyz)

        for idx, (moment_new_times, pos) in enumerate(zip(times, positions)):
            idx += 1
            times = xyz.info['gate times for channel %s' % idx]
            new_xyz.info['gate times for channel %s' % idx] = moment_new_times

            for col in xyz.layer_data.keys():
                if col.endswith("_ch%sgt" % idx):
                    new_xyz.layer_data[col] = pd.DataFrame(
                        np.nan,
                        index=new_xyz.flightlines.index,
                        columns=np.arange(len(moment_new_times)),
                        dtype=float)
                    new_xyz.layer_data[col].loc[:,pos] = xyz.layer_data[col]

        return new_xyz

    
    def forward_data_to_xyz(self, dpred, inversion=False):
        def reshape_nosplit(data):
            return data.reshape((len(self.xyz.flightlines),
                                  len(data) // len(self.xyz.flightlines)))
        def reshape(data):
            return self.split_moments(reshape_nosplit(data))
        
        xyzresp = libaarhusxyz.XYZ()
        xyzresp.model_info.update(self.xyz.model_info)
        xyzresp.flightlines = self.xyz.flightlines
        xyzresp.layer_data = {}

        if inversion:
            uncertfilt = np.isinf(self.data_uncert_array_culled)
            
            derr = (self.inv.invProb.dmisfit.data.dobs-dpred) * self.inv.invProb.dmisfit.W.diagonal()
            with np.errstate(divide='ignore'):
                std = np.abs(1 / self.inv.invProb.dmisfit.W.diagonal() / self.inv.invProb.dmisfit.data.dobs)

            # dpred, dobs etc contain dummy values where uncertainty
            # is inf. Don't let them through to the file or it will
            # look funny when plotting.
            dpred = np.where(uncertfilt, np.nan, dpred)
            derr = np.where(uncertfilt, np.nan, derr)
            std = np.where(uncertfilt, np.nan, std)
            
            for idx, moment in enumerate(reshape(derr)):
                xyzresp.layer_data["dbdt_err_ch%sgt" % (idx + 1)] = moment

            for idx, moment in enumerate(reshape(std)):
                xyzresp.layer_data["dbdt_std_ch%sgt" % (idx + 1)] = moment

            derrall = reshape_nosplit(derr)
            with np.errstate(divide='ignore'):
                xyzresp.flightlines['resdata'] = np.sqrt(np.nansum(derrall**2, axis=1) / (~np.isnan(derrall)).sum(axis=1))
            
        dpred = -dpred / self.xyz.model_info.get("scalefactor", 1)
        
        for idx, moment in enumerate(reshape(dpred)):
            xyzresp.layer_data["dbdt_ch%sgt" % (idx + 1)] = moment
                            
        # XYZ assumes all receivers have the same times
        for idx, t in enumerate(self.times):
            xyzresp.model_info["gate times for channel %s" % (idx + 1)] = list(t)

        return self.xyz.unfilter(self.pad_times(xyzresp, self.times_full, self.times_filter), layerfilter=False)
    
    def forward(self, **kw):
        """Does a forward modelling of the model in the XYZ file using
        this system description. Returns data in xyz format."""
        # self.inv.invProb.dmisfit.simulation

        self.options.update(kw)

        self.sim = self.make_forward()

        model_cond=np.log(1/self.xyz.resistivity.values)
        resp = self.sim.dpred(model_cond.flatten())

        return self.forward_data_to_xyz(resp)
