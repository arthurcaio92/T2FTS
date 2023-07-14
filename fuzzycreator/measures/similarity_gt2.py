"""This module contains similarity measures for general type-2 fuzzy sets."""

from collections import defaultdict
from numpy import linspace
from decimal import Decimal

from ..fuzzy_sets.discrete_t1_fuzzy_set import DiscreteT1FuzzySet
from .. import global_settings as gs


def jaccard(fs1, fs2):
    """Calculate the weighted average of the jaccard similarity on zslices."""
    top = 0
    bottom = 0
    for z in gs.get_z_points():
        j_top = 0
        j_bottom = 0
        for x in gs.get_x_points():
            fs1_l, fs1_u = fs1.calculate_membership(x, z)
            fs2_l, fs2_u = fs2.calculate_membership(x, z)
            j_top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
            j_bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
        if j_top == 0 and j_bottom == 0:
            return 0
        top += z * (j_top / j_bottom)
        bottom += z
    if top == 0 and bottom == 0:
        return 0
    return gs.rnd(top / bottom)


def _zslice_jaccard(fs1, fs2, z):
    """Measure the jaccard similarity between two zslices."""
    j_top = 0
    j_bottom = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x, z)
        fs2_l, fs2_u = fs2.calculate_membership(x, z)
        j_top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
        j_bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
    if j_bottom == 0:
        return Decimal(0)
    return gs.rnd(j_top / j_bottom)


def zhao_crisp(fs1, fs2):
    """Like jaccard, but the result is the standard average; not weighted."""
    sim = 0
    for z in gs.get_z_points():
        sim += _zslice_jaccard(fs1, fs2, z)
    return gs.rnd(sim / gs.global_zlevel_disc)


def hao_fuzzy(fs1, fs2):
    """Calculate the jaccard similarity given as type-1 fuzzy set."""
    sim_fs = defaultdict(int)
    for z in gs.get_z_points():
        sim = _zslice_jaccard(fs1, fs2, z)
        sim_fs[sim] = max(sim_fs[sim], z)
    fs = DiscreteT1FuzzySet(sim_fs)
    return fs


def hao_crisp(fs1, fs2):
    """Calculate the centroid of hao_fuzzy(fs1, fs2)."""
    sim_fs = hao_fuzzy(fs1, fs2)
    top = 0
    bottom = 0
    for y in sim_fs.points.keys():
        z = sim_fs.points[y]
        y = y.quantize(Decimal(10) ** -2)
        top += y * z
        bottom += z
    return top / bottom


def yang_lin(fs1, fs2):
    """Calculate the average jaccard similarity for each vertical slice."""
    result = 0
    n = 0
    for x in gs.get_x_points():
        top = 0
        bottom = 0
        y_points = gs.get_y_points()
        for y in y_points:
            z1 = fs1.calculate_secondary_membership(x, y)
            z2 = fs2.calculate_secondary_membership(x, y)
            top += min(y*z1, y*z2)
            bottom += max(y*z1, y*z2)
        top /= sum(y_points)
        bottom /= sum(y_points)
        if bottom > 0:  # if z values were present
            result += top/bottom
            n += 1
    if n == 0:
        return 0
    return gs.rnd(result / n)


def mohamed_abdaala(fs1, fs2):
    """Based on the the jaccard similarity for each vertical slice."""
    result = 0
    for x in gs.get_x_points():
        fs1_slices = 0
        fs2_slices = 0
        for y in gs.get_y_points():
            fs1_slices += 1 - y * fs1.calculate_secondary_membership(x, y)
            fs2_slices += 1 - y * fs2.calculate_secondary_membership(x, y)
        result += min(fs1_slices, fs2_slices) / max(fs1_slices, fs2_slices)
    return gs.rnd(result / gs.global_x_disc)


def hung_yang(fs1, fs2):
    """Based on the Hausdorff distance between vertical slice pairs."""
    distance = Decimal(0)
    for x in gs.get_x_points():
        top = Decimal(0)
        bottom = Decimal(0)
        for z in gs.get_z_points():
            y1_l, y1_u = fs1.calculate_membership(x, z)
            y2_l, y2_u = fs2.calculate_membership(x, z)
            hausdorff = max(abs(y1_l - y2_l), abs(y1_u - y2_u))
            top += hausdorff * z
            bottom += z
        try:
            distance += top / bottom
        except ZeroDivisionError:
            pass
    return gs.rnd(1 - (distance / gs.global_x_disc))


def wu_mendel(fs1, fs2):
    """Geometric approach."""
    top = 0
    bottom = 0
    for z in gs.get_z_points():
        for x in gs.get_x_points():
            fs1_l, fs1_u = fs1.calculate_membership(x, z)
            fs2_l, fs2_u = fs2.calculate_membership(x, z)
            top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
            bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
    return gs.rnd(top / bottom)
