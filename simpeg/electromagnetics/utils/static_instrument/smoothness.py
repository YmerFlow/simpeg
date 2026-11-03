"""Lateral links and smoothness length scales for stitched 1-D inversions.

With distance weighting (``regularization__distance_weighted``), a smoothness weight alpha
and its length scale L are two forms of one setting, alpha = alpha_s * L**2. The functions
here resolve such a pair, build the two kinds of lateral link (along a flight line and
between lines), and give the type that links a length parameter to its alpha in the
parameter schema.
"""

import typing

import numpy as np


def length_scale_of(alpha):
    """Type of a length-scale parameter, linked in the schema to the alpha it sets.

    swaggerspect copies ``swaggerspect_``-prefixed metadata into the parameter's schema, so
    the field carries ``"x-length-scale": {"alpha": alpha, "alpha_s": "alpha_s"}``, naming
    its sibling fields. A form can then keep the two in step with alpha = alpha_s * L**2.

    Parameters
    ----------
    alpha : str
        Name of the alpha field in the same parameter group, e.g. ``"alpha_r_inline"``.

    Returns
    -------
    typing.Annotated
        ``float`` annotated with the link.
    """
    return typing.Annotated[
        float, {"swaggerspect_x-length-scale": {"alpha": alpha, "alpha_s": "alpha_s"}}
    ]


def resolve_pair(alpha, length, alpha_s, alpha_name, length_name):
    """The alpha and length scale of one smoothness weight, from whichever was given.

    A given length sets the alpha (alpha = alpha_s * L**2); an alpha given alone sets the
    length; both given must agree to one part in a million.

    Parameters
    ----------
    alpha : float or None
        The weight as given [dimensionless]; None if not given.
    length : float or None
        Its length scale as given [CRS linear unit]; None if not given.
    alpha_s : float
        Smallness weight [dimensionless].
    alpha_name, length_name : str
        Parameter names, for the error messages.

    Returns
    -------
    alpha : float or None
        None only if neither was given.
    length : float or None
        None if neither was given, or if alpha_s is 0 (no length then).

    Raises
    ------
    ValueError
        If both are given and disagree, or a length is given with alpha_s not positive.
    """
    if length is None:
        if alpha is None or alpha_s <= 0:
            return alpha, None
        return float(alpha), float(np.sqrt(alpha / alpha_s))
    if alpha_s <= 0:
        raise ValueError(f"{length_name} = {length:g} needs a positive alpha_s, not {alpha_s:g}")
    from_length = alpha_s * length**2
    if alpha is not None and abs(alpha - from_length) > 1e-6 * max(abs(alpha), from_length):
        raise ValueError(
            f"{alpha_name} = {alpha:g} and {length_name} = {length:g} disagree: with alpha_s = "
            f"{alpha_s:g} that length gives {alpha_name} = {from_length:g}. Give one of them, "
            f"or both in agreement (alpha = alpha_s * L**2)."
        )
    return float(from_length), float(length)


def inline_links(line, order):
    """Links between consecutive soundings of each line, in flight order.

    Parameters
    ----------
    line : ndarray (n_sounding,)
        Line of each sounding.
    order : ndarray (n_sounding,)
        Position along the line in flight order, e.g. along-line distance [m] or time;
        ties keep the soundings' order.

    Returns
    -------
    ndarray (n_links, 2) of int
        Sounding index pairs, the earlier sounding first.
    """
    line = np.asarray(line)
    idx = np.lexsort((np.asarray(order), line))         # by line, then order; stable
    same = line[idx[1:]] == line[idx[:-1]]
    return np.column_stack([idx[:-1][same], idx[1:][same]])


def otherline_links(edges, line):
    """The triangulation edges whose two soundings are on different lines.

    Parameters
    ----------
    edges : ndarray (n_edges, 2) of int
        Sounding index pairs, e.g. the Delaunay edges of the sounding positions.
    line : ndarray (n_sounding,)
        Line of each sounding.

    Returns
    -------
    ndarray (n_links, 2) of int
    """
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    line = np.asarray(line)
    return edges[line[edges[:, 0]] != line[edges[:, 1]]]


def crs_unit(epsg):
    """Short name of a projected CRS's linear unit, from its EPSG code.

    Parameters
    ----------
    epsg : int or None
        EPSG code, e.g. from ``model_info["projection"]``.

    Returns
    -------
    str
        ``"m"``, ``"ft"`` or pyproj's unit name; ``"CRS units"`` if the CRS is unknown.
    """
    if epsg is None:
        return "CRS units"
    import pyproj

    try:
        name = pyproj.CRS.from_user_input(epsg).axis_info[0].unit_name
    except pyproj.exceptions.CRSError:
        return "CRS units"
    if name == "metre":
        return "m"
    return "ft" if "foot" in name else name
