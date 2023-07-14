"""This module contains similarity measures for interval type-2 fuzzy sets."""

from numpy import linspace, e
from decimal import Decimal

from .. import global_settings as gs


def zeng_li(fs1, fs2):
    """Based on the average distance between the membership values."""
    result = 0
    # restrict the universe of discourse because the
    # measure doesn't follow the property overlapping.
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        result += abs(fs1_l - fs2_l) + abs(fs1_u - fs2_u)
    result /= (2 * gs.global_x_disc)
    result = 1 - result
    return gs.rnd(result)


def gorzalczany(fs1, fs2):
    """Based on the highest membership where the fuzzy sets overlap."""
    max_of_min_lower_values = 0
    max_of_min_upper_values = 0
    max_lower_fs1 = 0
    max_upper_fs1 = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        max_of_min_lower_values = max(max_of_min_lower_values,
                                      min(fs1_l, fs2_l))
        max_of_min_upper_values = max(max_of_min_upper_values,
                                      min(fs1_u, fs2_u))
        max_lower_fs1 = max(max_lower_fs1, fs1_l)
        max_upper_fs1 = max(max_upper_fs1, fs1_u)
    measure1 = gs.rnd(max_of_min_lower_values / max_lower_fs1)
    measure2 = gs.rnd(max_of_min_upper_values / max_upper_fs1)
    return min(measure1, measure2), max(measure1, measure2)


def bustince(fs1, fs2, t_norm_min=True):
    """Based on the inclusion of one fuzzy set within the other."""
    yl_ab = 1
    yl_ba = 1
    yu_ab = 1
    yu_ba = 1
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        yl_ab = min(yl_ab, min(1 - fs1_l + fs2_l,
                               1 - fs1_u + fs2_u))
        yl_ba = min(yl_ba, min(1 - fs2_l + fs1_l,
                               1 - fs2_u + fs1_u))
        yu_ab = min(yu_ab, max(1 - fs1_l + fs2_l,
                               1 - fs1_u + fs2_u))
        yu_ba = min(yu_ba, max(1 - fs2_l + fs1_l,
                               1 - fs2_u + fs1_u))
    return min(yl_ab, yl_ba), min(yu_ab, yu_ba)


def jaccard(fs1, fs2):
    """Ratio between the intersection and union of the fuzzy sets."""
    top = 0
    bottom = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        top += min(fs1_u, fs2_u) + min(fs1_l, fs2_l)
        bottom += max(fs1_u, fs2_u) + max(fs1_l, fs2_l)
    return gs.rnd(top / bottom)


def zheng(fs1, fs2):
    """Similar to jaccard; based on the intersection and union of the sets."""
    top_a = 0
    top_b = 0
    bottom_a = 0
    bottom_b = 0
    for x in gs.get_x_points():
        fs1_l, fs1_u = fs1.calculate_membership(x)
        fs2_l, fs2_u = fs2.calculate_membership(x)
        top_a += min(fs1_u, fs2_u)
        top_b += min(fs1_l, fs2_l)
        bottom_a += max(fs1_u, fs2_u)
        bottom_b += max(fs1_l, fs2_l)
    return gs.rnd(Decimal('0.5') * ((top_a / bottom_a) + (top_b / bottom_b)))


def vector(fs1, fs2):
    """Vector similarity based on the distance and similarity of shapes."""
    fs1_c = fs1.calculate_overall_centre_of_sets()
    fs2_c = fs2.calculate_overall_centre_of_sets()
    dist = fs1_c - fs2_c
    # temporarily align the centroid of fs2 with fs1 to compare shapes
    fs2.mf1.shift_membership_function(dist)
    fs2.mf2.shift_membership_function(dist)
    # find out the support of the union of the fuzzy sets
    # and weight the absolute distance by this support
    x_min = min(max(fs1.uod[0], fs1.mf1.x_min),
                max(fs1.uod[0], fs1.mf2.x_min),
                max(fs2.uod[0], fs2.mf1.x_min),
                max(fs2.uod[0], fs2.mf2.x_min))
    x_max = max(min(fs1.uod[1], fs1.mf1.x_max),
                min(fs1.uod[1], fs1.mf2.x_max),
                min(fs2.uod[1], fs2.mf1.x_max),
                min(fs2.uod[1], fs2.mf2.x_max))
    r = Decimal(4) / (x_max - x_min)
    proximity = pow(Decimal(e), -r * abs(dist))
    shape_difference = jaccard(fs1, fs2)
    # put the fuzzy set fs2 back to where it was
    fs2.mf1.shift_membership_function(-dist)
    fs2.mf2.shift_membership_function(-dist)
    return gs.rnd(shape_difference * proximity)
