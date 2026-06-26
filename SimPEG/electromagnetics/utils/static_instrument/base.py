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

import warnings

try:
    from pymatsolver import PardisoSolver as Solver
    print("pymatsolver.PardisoSolver available for fwd modelling")
except:
    print("Could not import PardisoSolver, only default (spLU) available")

try:
    from kneed import KneeLocator
except ImportError:
    KneeLocator = None

import scipy.stats
import copy
import re
import typing

from . import xyzfilter


class XYZClusterer:
    """Cluster AEM soundings by data similarity to build a clustered start model.

    Takes a filtered XYZ object (``system.xyz``) and produces per-cluster
    representative soundings that can be inverted cheaply before the full
    inversion starts.  K is selected automatically via the Kneedle algorithm
    (requires the *kneed* package).

    Parameters
    ----------
    xyz : FilteredXYZ
        The (filtered) sounding data — pass ``system.xyz``.
    k_range : iterable of int
        K values to scan when choosing the number of clusters.
    random_seed : int or None
        Random seed for K-means reproducibility.
    valid_gate_threshold : float
        Gates present in fewer than this fraction of soundings are excluded
        from the feature vector.
    """

    def __init__(self, xyz, k_range=range(2, 20),
                 random_seed=None, valid_gate_threshold=0.5):
        self.xyz = xyz
        self.k_range = k_range
        self.random_seed = random_seed
        self.valid_gate_threshold = valid_gate_threshold
        self.cluster_ids_ = None
        self.n_clusters_ = None

    def build_feature_matrix(self):
        """Return z-scored ``(n_soundings, n_features)`` matrix for K-means.

        Features: ``log|data|`` for gates present in ≥ threshold of soundings,
        plus flight altitude.  NaN entries are zero-filled after scaling.
        """
        data_arrays = [
            df.values.astype(float)
            for col, df in sorted(self.xyz.layer_data.items())
            if re.match(r'^dbdt_ch\d+gt$', col)
        ]
        if not data_arrays:
            raise ValueError("No dbdt_ch*gt columns found; cannot build feature matrix")
        data_2d = np.hstack(data_arrays)

        valid_gates = np.mean(~np.isnan(data_2d), axis=0) >= self.valid_gate_threshold
        data_2d = data_2d[:, valid_gates]

        with np.errstate(divide='ignore', invalid='ignore'):
            log_data = np.where(data_2d != 0, np.log(np.abs(data_2d)), np.nan)

        altitude = self.xyz.flightlines[self.xyz.alt_column].values.astype(float).reshape(-1, 1)
        features = np.hstack([log_data, altitude])

        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0)
        std[std == 0] = 1.0
        return np.where(np.isnan((features - mean) / std), 0.0, (features - mean) / std)

    def _select_k(self, features):
        """Run K-means over k_range, apply Kneedle, return optimal K."""
        from sklearn.cluster import KMeans
        if KneeLocator is None:
            raise ImportError("kneed package required for cluster count auto-detection: pip install kneed")
        k_list = list(self.k_range)
        inertias = [
            KMeans(n_clusters=k, random_state=self.random_seed, n_init=10).fit(features).inertia_
            for k in k_list
        ]
        return KneeLocator(k_list, inertias, curve='convex', direction='decreasing').knee

    def fit(self):
        """Detect K via Kneedle, run K-means, store results; return cluster_ids."""
        from sklearn.cluster import KMeans
        features = self.build_feature_matrix()
        k = self._select_k(features)
        print(f"Clustering: kneedle selected K={k}")
        km = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
        km.fit(features)
        self.cluster_ids_ = km.labels_
        self.n_clusters_ = k
        self.medoid_indices_ = self.find_medoids(features)
        return self.cluster_ids_

    def find_medoids(self, features):
        """Return the index (into self.xyz / features rows) of the medoid per cluster.

        The medoid is the actual sounding that minimises mean distance to all
        other cluster members in feature space.  Returns an int array of length
        n_clusters_ where entry k is the sounding index for cluster k.
        """
        medoids = np.zeros(self.n_clusters_, dtype=int)
        for k in range(self.n_clusters_):
            mask = self.cluster_ids_ == k
            cluster_features = features[mask]
            global_indices = np.where(mask)[0]
            diffs = cluster_features[:, np.newaxis, :] - cluster_features[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            local_idx = dists.mean(axis=1).argmin()
            medoids[k] = global_indices[local_idx]
        return medoids

    def cluster_models_to_startmodel(self, cluster_l2, thicknesses, n_soundings, default_res,
                                     raw_medoid_indices=None):
        """Map cluster inverted resistivities into a ``(n_soundings * n_layers,)`` start model.

        Must call ``fit()`` first.  raw_medoid_indices maps cluster k to its row
        in cluster_l2 (needed when cluster_l2 spans the full dataset after unfilter).
        """
        n_layers = len(thicknesses) + 1
        startmodel = np.full(n_soundings * n_layers, np.log(1.0 / default_res))
        if 'resistivity' not in cluster_l2.layer_data:
            return startmodel
        cluster_res = cluster_l2.layer_data['resistivity'].values
        for i, k in enumerate(self.cluster_ids_):
            row = raw_medoid_indices[int(k)] if raw_medoid_indices is not None else int(k)
            res_k = cluster_res[row]
            valid = np.isfinite(res_k) & (res_k > 0)
            fallback = np.full_like(res_k, default_res)
            startmodel[i * n_layers:(i + 1) * n_layers] = np.log(1.0 / np.where(valid, res_k, fallback))
        return startmodel


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
    def gate_filter(self):
        filt = {}
        for key in self._xyz.layer_data.keys():
            match = re.match(r"^[^0-9]*([0-9]+).*", key)
            if match is None: continue
            channel = int(match.groups()[0]) - 1
            n_gates = self._xyz.layer_data[key].shape[1]
            filt[key] = self.times_filter[channel][:n_gates]
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
        return [np.array(self.xyz.model_info['gate times for channel 1'])]

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
    simulation__n_cpu = 3
    "Number of CPU threads for parallel simulation. Set to the number of available cores on the machine (minus 1–2 for OS headroom). Increasing beyond the number of physical cores gives diminishing returns."
    def make_simulation(self, survey, thicknesses):
        if 'pardiso' in self.simulation__solver.lower():
            print('Using Pardiso solver')
            return tdem.Simulation1DLayeredStitched(
                survey=survey,
                thicknesses=thicknesses,
                sigmaMap=maps.ExpMap(nP=self.n_param(thicknesses)), 
                solver=PardisoSolver,
                parallel=self.simulation__parallel,
                n_cpu=self.simulation__n_cpu,
                n_layer=self.n_layer_used)
        else:
            print('Using default (spLU) solver')
            return tdem.Simulation1DLayeredStitched(
                survey=survey,
                thicknesses=thicknesses,
                sigmaMap=maps.ExpMap(nP=self.n_param(thicknesses)), 
                parallel=self.simulation__parallel,
                n_cpu=self.simulation__n_cpu,
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

    clustering__enabled = False
    "Set to True to cluster soundings before inversion. K is selected automatically via the Kneedle algorithm (requires the kneed package)."
    clustering__k_range = range(2, 20)
    "K values scanned when selecting the number of clusters automatically."
    clustering__random_seed = None
    "Random seed for K-means reproducibility."
    clustering__valid_gate_threshold = 0.5
    "Gates present in fewer than this fraction of soundings are excluded from the feature vector."

    def _make_clusterer(self):
        return XYZClusterer(
            xyz=self.xyz,
            k_range=self.clustering__k_range,
            random_seed=self.clustering__random_seed,
            valid_gate_threshold=self.clustering__valid_gate_threshold,
        )

    def make_startmodel(self, thicknesses):
        if not self.clustering__enabled:
            return np.log(np.ones(self.n_param(thicknesses)) * 1/self.startmodel__res)

        # Cached: make_regularization → make_mref → make_startmodel triggers the cluster
        # inversion; inv.run(make_startmodel(...)) then returns the same array cheaply.
        if hasattr(self, '_cached_cluster_startmodel'):
            return self._cached_cluster_startmodel

        clusterer = self._make_clusterer()
        self._clusterer = clusterer
        print(f"Clustering: fitting {len(self.xyz.flightlines)} soundings")
        self._cluster_ids = clusterer.fit()

        # Medoid indices are into self.xyz (filtered, 0-based).  Map them to
        # raw _xyz indices so the child can filter exactly once, the same way
        # the parent does, avoiding any double-filter width mismatch.
        medoid_filtered = clusterer.medoid_indices_
        sf = self.sounding_filter
        if isinstance(sf, slice):
            raw_medoid_indices = medoid_filtered
        elif hasattr(sf, 'dtype') and sf.dtype == bool:
            raw_medoid_indices = np.where(sf)[0][medoid_filtered]
        else:
            raw_medoid_indices = np.asarray(sf)[medoid_filtered]

        print(f"Clustering: inverting {clusterer.n_clusters_} medoid soundings")

        # Inherit the parent's options (n_layer, n_cpu, gate_filter__*, etc.),
        # excluding clustering namespaces.  The child uses self._xyz directly with
        # sounding_filter selecting only the K medoid rows; its gate filter is
        # applied once, identically to the parent — no special time overrides needed.
        cluster_opts = {
            key: val for key, val in self.options.items()
            if not (key.startswith('clustering__') or key.startswith('cluster_inversion__'))
        }
        cluster_opts['clustering__enabled'] = False  # prevent recursion
        cluster_opts['validate'] = False
        cluster_opts['regularization__alpha_r'] = 0
        cluster_opts['sounding_filter'] = raw_medoid_indices
        for key, val in self.options.items():
            if key.startswith('cluster_inversion__'):
                cluster_opts[key[len('cluster_inversion__'):]] = val

        cluster_system = type(self)(self._xyz, **cluster_opts)
        self._cluster_system = cluster_system  # expose for diagnostics (iteration count)
        _, cluster_l2 = cluster_system.invert()

        startmodel = clusterer.cluster_models_to_startmodel(
            cluster_l2, thicknesses, len(self.xyz.flightlines), self.startmodel__res,
            raw_medoid_indices=raw_medoid_indices)
        self._cached_cluster_startmodel = startmodel
        return startmodel

    regularization__mref = 'startmodel'
    "Reference model for regularization: 'startmodel' (default) uses the cluster-derived start model; 'halfspace' uses a flat halfspace at startmodel__res."

    def make_mref(self, thicknesses):
        """Return the reference model for regularization."""
        if self.regularization__mref == 'halfspace':
            return np.log(np.ones(self.n_param(thicknesses)) * 1/self.startmodel__res)
        return self.make_startmodel(thicknesses)

    # FIXME!!! Should alpha_s's default be set to something based off the model domain?
    #  https://giftoolscookbook.readthedocs.io/en/latest/content/fundamentals/Alphas.html
    #  Here it talks about how how alpha_s is often set to
    #  alpha_s = 1/(h**2),
    #  where h is the cell size dimension for the core region.
    #  for us h could be
    #    1) the height of the last, non-halfspace layer,
    #    2) the average thickness of our model domain,
    #    3) the average sounding spacing.
    #    4) 1e-4 as proposed in the link above
    #    5) line spacing (if 100m then alpha_s = 1e-4, if 400m then 6.3e-6)
    #    6) geomean of the linespacing and sounding spacing: sqrt(line_space * sound_space)
    #        - 25m sounding spacing, 100m line spacing: h=50, alpha_s=4e-4
    regularization__alpha_s = 1e-4
    "Smallness weight — penalizes deviation of each layer from the reference (starting) model. Larger values anchor the model more strongly to 'startmodel__res'. A rule of thumb: alpha_s ≈ 1 / (sounding_spacing × line_spacing) in m⁻². For 25 m sounding spacing and 100 m line spacing: alpha_s ≈ 4e-4. Too large: model is too smooth and resistivity extremes are suppressed. Too small: unconstrained model, may fit noise."
    regularization__alpha_r = 1.
    "Lateral (along-line) smoothness weight — penalizes resistivity differences between neighboring soundings. The ratio alpha_r / alpha_z controls lateral vs. vertical smoothing. Default 1:1 is isotropic. Increase alpha_r relative to alpha_z to enforce more lateral continuity (useful for layered geology)."
    regularization__alpha_z = 1.
    "Vertical smoothness weight — penalizes resistivity differences between adjacent layers in a sounding. The ratio alpha_z / alpha_r controls vertical vs. lateral smoothing. Increase alpha_z relative to alpha_r to enforce more layered structure (i.e., smoother depth profiles)."

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
            reg.mref = self.make_mref(thicknesses)
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
            reg.mref = self.make_mref(thicknesses)
            return reg

    directives__beta__seed : int = None
    "Random seed for the beta estimator. Set to a fixed integer for reproducible results across runs. Leave blank (None) for random initialization."
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

        if hasattr(self, '_cluster_ids'):
            # Map cluster IDs back to the full (unfiltered) sounding set
            cluster_id_full = pd.Series(np.nan, index=self._xyz.flightlines.index, dtype=float)
            cluster_id_full.loc[self.xyz.flightlines.index] = self._cluster_ids.astype(float)
            for obj in [self.sparse, self.l2, self.l2pred, self.sparsepred, self.corrected]:
                if obj is not None:
                    obj.flightlines = obj.flightlines.copy()
                    obj.flightlines['cluster_id'] = cluster_id_full.values

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
