"""Christiansen-Auken (2012) depth of investigation for 1-D EM models.

Method, from Christiansen, A. V., and E. Auken, 2012, A global measure for depth of
investigation: Geophysics 77(4), WB171-WB177:

- eq. 2: G_ij = d ln|d_i| / d ln(m_j), the Jacobian of the final model in log data and log
  model space. Here m is conductivity; the sign flips for resistivity and |G| removes it.
- eq. 3: s_j = sum_i |G_ij| / dd_i over the data used in the inversion, dd_i being the
  relative standard deviation of datum i. Dimensionless.
- eq. 4: s*_j = s_j / t_j, the same per meter of layer, for plotting only (not computed here).
- eq. 5: S_j = sum_{k >= j} s_k, cumulated from the bottom layer (the halfspace included) up.
- The DOI is the depth where S falls to a threshold. The paper uses one threshold (0.8;
  0.6-1.2 tested). Aarhus Workbench (AarhusInv con file) reports two: DOIAbsHigh 1.5 for the
  shallow "upper" DOI and DOIAbsLow 0.75 for the deep "lower" DOI.

Scale: by default s_j = sum_i |G_ij|, without the 1/dd_i of eq. 3 as printed
(``weighted=False``). The paper's own figures and the AarhusInv manual are on that scale, and
the thresholds above were tuned to it: the manual's DC example (20 data at 2 %, whose log
sensitivities sum to exactly 1 per datum) tabulates a total of 19.4, where eq. 3 as printed
gives at least 1000; the paper's Fig. 1 (30 gates at 5 %) totals about 20, not several
hundred. With the 1/dd_i, the DOIs of a VTEM survey at 3-5 % noise land on the base of a
500 m model. Noise still enters through which data are used (``inf`` uncertainty = not
used). ``weighted=True`` applies eq. 3 as printed.

The absolute value is taken per element of G before summing. Eq. 3 is written without it,
but TEM sensitivities change sign across gates, and a signed sum would let gates cancel.

S is placed at layer tops (S_j at the top of layer j) and interpolated linearly in depth
between them, which reproduces the "DOI absolute" depths of the AarhusInv 8.11 manual's emo
example (Figure 15.1: 88.34 m at threshold 1.5, 133.47 m at 0.75).

The functions here need numpy only; the two ``stitched_*`` / ``doi_christiansen_auken``
functions take a stitched 1-D simulation
(:class:`simpeg.electromagnetics.base_1d_stitched.BaseStitchedEM1DSimulation`).
"""

import numpy as np


def log_sensitivity(J_sigma, sigma, d_pred):
    """Jacobian of one sounding in log space, G = d ln|d| / d ln(sigma) (C-A 2012 eq. 2).

    Parameters
    ----------
    J_sigma : ndarray (n_data, n_layer)
        d d / d sigma for one sounding [data units per S/m], as SimPEG's 1-D ``getJ`` returns
        it under ``"ds"`` (with respect to the physical conductivity, before any model map).
    sigma : ndarray (n_layer,)
        Layer conductivity of the final model [S/m].
    d_pred : ndarray (n_data,)
        Predicted data of the final model [data units].

    Returns
    -------
    G : ndarray (n_data, n_layer)
        Dimensionless log-log sensitivity.
    """
    return J_sigma * sigma[None, :] / d_pred[:, None]


def layer_sensitivity(G, relative_uncertainty, weighted=False):
    """Sensitivity per layer to all used data, s_j = sum_i |G_ij| (C-A 2012 eq. 3, see the module notes).

    Parameters
    ----------
    G : ndarray (n_data, n_layer)
        Log-log sensitivity of one sounding (see ``log_sensitivity``).
    relative_uncertainty : ndarray (n_data,)
        Relative standard deviation of each datum [fraction, 0.03 = 3 %]. ``inf`` marks data
        not used in the inversion; they contribute nothing.
    weighted : bool
        False (default): Workbench scale, sum of |G_ij| over the used data. True: eq. 3 as
        printed, each term divided by the datum's relative standard deviation.

    Returns
    -------
    s : ndarray (n_layer,)
        Dimensionless sensitivity of each layer to all used data.

    Raises
    ------
    ValueError
        If a used datum has a non-finite sensitivity (e.g. a zero predicted datum).
    """
    used = np.isfinite(relative_uncertainty)
    G = G[used]
    if not np.isfinite(G).all():
        raise ValueError("non-finite sensitivity in a used datum")
    if weighted:
        return (np.abs(G) / relative_uncertainty[used, None]).sum(axis=0)
    return np.abs(G).sum(axis=0)


def cumulative_sensitivity(s):
    """Sensitivity of each layer and everything below it, S_j = sum_{k >= j} s_k (C-A 2012 eq. 5).

    Parameters
    ----------
    s : ndarray (..., n_layer)
        Layer sensitivities, top layer first, halfspace last [dimensionless].

    Returns
    -------
    S : ndarray (..., n_layer)
        Cumulated from the bottom, so S[..., 0] is the total and S is non-increasing with depth.
    """
    return np.cumsum(s[..., ::-1], axis=-1)[..., ::-1]


def doi_from_cumulative(S, depth_top, threshold):
    """Depth where the cumulative sensitivity falls to a threshold.

    Parameters
    ----------
    S : ndarray (n_sounding, n_layer) or (n_layer,)
        Cumulative sensitivity at each layer top [dimensionless] (see ``cumulative_sensitivity``).
    depth_top : ndarray (n_layer,)
        Layer top depths [m, positive down, 0 at the surface].
    threshold : float
        Absolute threshold [dimensionless]; larger gives a shallower DOI.

    Returns
    -------
    doi : ndarray (n_sounding,)
        DOI [m below the surface], interpolated linearly in depth between the layer tops that
        bracket the threshold. 0 where even the total sensitivity is below the threshold;
        ``depth_top[-1]`` (a lower bound) where the halfspace alone exceeds it; NaN where S is NaN.

    Examples
    --------
    Relative DOI, as AarhusInv reports it beside the absolute one:

    >>> doi_rel = doi_from_cumulative(S / S[:, :1], depth_top, 0.05)
    """
    S = np.atleast_2d(S)
    depth_top = np.asarray(depth_top, dtype=float)
    above = S >= threshold                              # non-increasing with depth, so a prefix
    k = above.sum(axis=1) - 1                           # last layer top still at or above threshold
    doi = np.full(len(S), np.nan)
    doi[k < 0] = 0.0
    doi[k == S.shape[1] - 1] = depth_top[-1]
    mid = (k >= 0) & (k < S.shape[1] - 1)
    i = np.flatnonzero(mid)
    s0, s1 = S[i, k[i]], S[i, k[i] + 1]
    z0, z1 = depth_top[k[i]], depth_top[k[i] + 1]
    doi[i] = z0 + (s0 - threshold) / (s0 - s1) * (z1 - z0)
    doi[np.isnan(S).any(axis=1)] = np.nan
    return doi


def stitched_cumulative_sensitivity(simulation, model, relative_uncertainty, weighted=False,
                                    d_pred=None):
    """Cumulative sensitivity per sounding and layer for a stitched 1-D simulation's final model.

    The stitched problem is block diagonal (each sounding's data depend only on its own
    layers), so SimPEG's per-sounding Jacobians are used directly. Costs one ``getJ`` at
    ``model``, plus one ``dpred`` unless ``d_pred`` is given.

    Parameters
    ----------
    simulation : simpeg.electromagnetics.base_1d_stitched.BaseStitchedEM1DSimulation
        The simulation the model was inverted with (same survey, thicknesses and mapping).
        Its model is set to ``model``.
    model : ndarray (n_sounding * n_layer,)
        Final inversion model in the simulation's model space, sounding-major.
    relative_uncertainty : ndarray (n_data,)
        Relative standard deviation of each datum in survey order [fraction]; ``inf`` for data
        not used in the inversion.
    weighted : bool
        Divide by the relative standard deviation (eq. 3 as printed); see ``layer_sensitivity``.
    d_pred : ndarray (n_data,), optional
        Predicted data at ``model`` [data units], e.g. ``inv_prob.dpred`` after an inversion.
        Computed when not given.

    Returns
    -------
    S : ndarray (n_sounding, n_layer)
        Cumulative sensitivity at each layer top [dimensionless].
    depth_top : ndarray (n_layer,)
        Layer top depths [m, positive down].

    Raises
    ------
    ValueError
        If the simulation has per-sounding thicknesses or a fixed Jacobian, or the data count
        does not match.
    """
    if getattr(simulation, "fix_Jmatrix", False):
        raise ValueError("simulation.fix_Jmatrix is set, so getJ may return the J of an earlier model")
    thicknesses = np.asarray(simulation.thicknesses, dtype=float)
    if thicknesses.ndim != 1 or len(thicknesses) != simulation.n_layer - 1:
        raise ValueError("one shared set of layer thicknesses is assumed")
    if d_pred is None:
        d_pred = simulation.dpred(model)
    d_pred = np.asarray(d_pred, dtype=float)
    J = simulation.getJ(model)                          # also sets simulation.model = model
    sigma = np.reshape(simulation.sigma, (simulation.n_sounding, simulation.n_layer))
    depth_top = np.r_[0.0, np.cumsum(thicknesses)]
    n_data = [simulation.survey.vnD_by_sounding[i] for i in range(simulation.n_sounding)]
    if sum(n_data) != len(d_pred) or len(relative_uncertainty) != len(d_pred):
        raise ValueError(f"data count mismatch: {sum(n_data)} by sounding, {len(d_pred)} predicted, "
                         f"{len(relative_uncertainty)} uncertainties")
    edges = np.r_[0, np.cumsum(n_data)]
    s = np.empty((simulation.n_sounding, simulation.n_layer))
    for i in range(simulation.n_sounding):
        rows = slice(edges[i], edges[i + 1])
        G = log_sensitivity(np.asarray(J[i]["ds"]), sigma[i], d_pred[rows])
        s[i] = layer_sensitivity(G, relative_uncertainty[rows], weighted)
    return cumulative_sensitivity(s), depth_top


def doi_christiansen_auken(simulation, model, relative_uncertainty, thresholds=(1.5, 0.75),
                           weighted=False, d_pred=None):
    """Christiansen-Auken (2012) DOI of every sounding of a stitched 1-D inversion.

    Parameters
    ----------
    simulation : simpeg.electromagnetics.base_1d_stitched.BaseStitchedEM1DSimulation
        The simulation the model was inverted with. Its model is set to ``model``.
    model : ndarray (n_sounding * n_layer,)
        Final inversion model in the simulation's model space, sounding-major.
    relative_uncertainty : ndarray (n_data,)
        Relative standard deviation of each datum [fraction]; ``inf`` for unused data. For an
        inversion with absolute uncertainties, divide them by |d_obs|.
    thresholds : sequence of float
        Absolute thresholds [dimensionless]. Default: Aarhus Workbench's DOIAbsHigh 1.5
        (upper, shallow) and DOIAbsLow 0.75 (lower, deep). The paper's single value is 0.8.
    weighted : bool
        False (default): Workbench scale, which the thresholds above belong to. True: eq. 3 as
        printed, dividing by the relative standard deviation; see the module notes.
    d_pred : ndarray (n_data,), optional
        Predicted data at ``model`` [data units]; computed when not given.

    Returns
    -------
    doi : ndarray (n_sounding, len(thresholds))
        DOI [m below the surface] per threshold (see ``doi_from_cumulative`` for the edge cases).
    S : ndarray (n_sounding, n_layer)
        Cumulative sensitivity at each layer top [dimensionless], for plots and QC.
    depth_top : ndarray (n_layer,)
        Layer top depths [m, positive down].

    Examples
    --------
    >>> doi, S, z = doi_christiansen_auken(sim, inv_model, uncert / np.abs(dobs))
    >>> doi_upper, doi_lower = doi.T
    """
    S, depth_top = stitched_cumulative_sensitivity(simulation, model, relative_uncertainty,
                                                   weighted, d_pred)
    doi = np.column_stack([doi_from_cumulative(S, depth_top, t) for t in thresholds])
    return doi, S, depth_top


def _line_key(value):
    """A line id as a string key, "101" rather than "101.0"."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _base_figures(doi_upper, doi_lower, z_first, z_at_base, z_base):
    """The DOI-at-base shares of one group of soundings; None where it has no DOI."""
    has_doi = np.isfinite(doi_upper) & np.isfinite(doi_lower)
    up, low = doi_upper[has_doi], doi_lower[has_doi]
    out = {"n_soundings": int(len(doi_lower)), "n_with_doi": int(has_doi.sum())}
    if not has_doi.any():
        return dict(out, lower_at_base=None, upper_at_base=None, lower_p90_of_base=None,
                    upper_in_first_layer=None)
    return dict(out,
                lower_at_base=float(np.mean(low >= z_at_base)),
                upper_at_base=float(np.mean(up >= z_at_base)),
                lower_p90_of_base=float(np.percentile(low, 90) / z_base),
                upper_in_first_layer=float(np.mean(up < z_first)))


def doi_base_check(doi_upper, doi_lower, depth_top, line=None, threshold=0.05):
    """How often the DOI sits at the model base, per line and for the whole run.

    A DOI at the base means the model is cut off there: the half-space top, a modelling choice,
    is too shallow for the system and the ground. A DOI counts as at the base when it lies
    within half a bottom-layer thickness of the half-space top, ``doi >= z_base - t_bottom / 2``.
    A DOI equal to ``z_base`` is the case where the half-space alone is above the threshold.

    Parameters
    ----------
    doi_upper, doi_lower : ndarray (n_sounding,)
        Upper (shallow) and lower (deep) DOI [m below the surface, positive down]. NaN marks a
        sounding without a DOI; it is counted, and left out of every share.
    depth_top : ndarray (n_layer,)
        Layer top depths [m, positive down, 0 at the surface], the half-space top last. The
        DOI arrays are in the same length unit.
    line : ndarray (n_sounding,), optional
        Line id of each sounding. Without it, only the run figures are computed.
    threshold : float
        Share of a line's soundings with a DOI [fraction, 0.05 = 5 %] above which the lower DOI
        at the base puts the line in ``lines_over_threshold``.

    Returns
    -------
    check : dict
        Plain Python values, so it can go into ``model_info``, msgpack or JSON as it is:

        - ``threshold``; ``base_depth``, ``bottom_layer`` and ``first_layer``: the half-space
          top, the thickness of the layer above it and the bottom of the first layer [m].
        - ``run`` and, per line (string keys, in order of first appearance), ``lines``: each
          with ``n_soundings``, ``n_with_doi``, ``lower_at_base`` and ``upper_at_base`` (shares
          of the soundings with a DOI), ``lower_p90_of_base`` (90th percentile of the lower DOI
          as a fraction of ``base_depth``) and ``upper_in_first_layer`` (share whose upper DOI
          is inside the first layer, so shallower than the earliest gates resolve). Shares are
          None for a group without a DOI.
        - ``lines_over_threshold``: lines whose ``lower_at_base`` is above ``threshold``.

    Raises
    ------
    ValueError
        If there are fewer than two layer tops, or the arrays differ in length.

    Examples
    --------
    >>> fl = model.flightlines
    >>> check = doi_base_check(fl.doi_upper.values, fl.doi_lower.values,
    ...                        model.layer_data["dep_top"].values[0], fl.line_no.values)
    >>> print("\\n".join(doi_check_messages(check)))
    """
    doi_upper = np.asarray(doi_upper, dtype=float)
    doi_lower = np.asarray(doi_lower, dtype=float)
    depth_top = np.asarray(depth_top, dtype=float)
    if len(depth_top) < 2:
        raise ValueError("at least one layer above the half-space is needed")
    if len(doi_upper) != len(doi_lower) or (line is not None and len(line) != len(doi_lower)):
        raise ValueError("doi_upper, doi_lower and line differ in length")
    z_base, t_bottom = depth_top[-1], depth_top[-1] - depth_top[-2]
    z_at_base = z_base - t_bottom / 2

    def figures(mask):
        return _base_figures(doi_upper[mask], doi_lower[mask], depth_top[1], z_at_base, z_base)

    lines = {}
    if line is not None:
        keys = [_line_key(v) for v in line]
        lines = {k: figures(np.array(keys) == k) for k in dict.fromkeys(keys)}
    over = [k for k, f in lines.items()
            if f["lower_at_base"] is not None and f["lower_at_base"] > threshold]
    return {"threshold": float(threshold), "base_depth": float(z_base),
            "bottom_layer": float(t_bottom), "first_layer": float(depth_top[1]),
            "run": figures(slice(None)), "lines": lines, "lines_over_threshold": over}


def doi_check_messages(check, name=None):
    """Log lines for a ``doi_base_check`` result: a summary, and a warning when lines are over.

    Parameters
    ----------
    check : dict
        As returned by ``doi_base_check``.
    name : str, optional
        Which model it is ("l2", "sparse"), put in front of each line.

    Returns
    -------
    messages : list of str
        The ``##DOI_CHECK##`` summary, then a ``WARNING:`` line when the lower DOI is at the
        base on more than ``threshold`` of the soundings of any line (of the run, without lines).
    """
    run, lines, over = check["run"], check["lines"], check["lines_over_threshold"]
    base, pct = f"{check['base_depth']:.0f} m", f"{100 * check['threshold']:g}%"
    who = f"{name} model: " if name else ""
    if run["n_with_doi"] == 0:
        return [f"##DOI_CHECK## {who}no sounding has a DOI ({run['n_soundings']} soundings)"]
    summary = (f"##DOI_CHECK## {who}lower DOI at the base on {100 * run['lower_at_base']:.1f}% "
               f"of {run['n_with_doi']} soundings with a DOI (upper {100 * run['upper_at_base']:.1f}%), "
               f"lower DOI p90 at {100 * run['lower_p90_of_base']:.0f}% of the {base} base, "
               f"upper DOI in the first layer on {100 * run['upper_in_first_layer']:.1f}%")
    if lines:
        summary += f"; {len(over)} of {len(lines)} lines over {pct}"
    messages = [summary]
    warning = f"WARNING: {who}the model base ({base}) cuts off the lower DOI on more than {pct} of soundings"
    if over:
        messages.append(f"{warning} on these lines: {', '.join(over)}; consider a deeper base")
    elif not lines and run["lower_at_base"] > check["threshold"]:
        messages.append(f"{warning}; consider a deeper base")
    return messages
