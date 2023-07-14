"""This module contains distance measures for type-1 fuzzy set."""

import functools
from copy import deepcopy
from decimal import Decimal
from numpy import linspace, sqrt
from scipy import integrate

from .. import global_settings as gs
from ..fuzzy_exceptions import AlphaCutError


def _hausdorff(fs1, fs2, alpha):
    """Calculate the Hausdorff distance at the given alpha cut."""
    fs1_min, fs1_max = fs1.calculate_alpha_cut(alpha)
    fs2_min, fs2_max = fs2.calculate_alpha_cut(alpha)
    return max(abs(fs1_min - fs2_min), abs(fs1_max - fs2_max))


def _minkowski_r1(fs1, fs2, alpha):
    """Calculate the minkwoski distance where r = 1 at the given alpha cut."""
    fs1_min, fs1_max = fs1.calculate_alpha_cut(alpha)
    fs2_min, fs2_max = fs2.calculate_alpha_cut(alpha)
    return abs(fs1_min - fs2_min), abs(fs1_max - fs2_max)


def centroid(fs1, fs2):
    """Calculate the diffefence between the centroids of the fuzzy sets."""
    return abs(fs1.calculate_centroid() - fs2.calculate_centroid())


def ralescu1(fs1, fs2):
    """Calculate the average Hausdorff distance over all alpha-cuts."""
    def haus(alpha):
        return _hausdorff(fs1, fs2, alpha)
    a, b = integrate.quad(haus, 0, 1)
    return gs.rnd(a)


def ralescu2(fs1, fs2):
    """Calculate the maximum Hausdorff distance over all alpha-cuts."""
    result = Decimal(0)
    for alpha in gs.get_y_points():
        dist = _hausdorff(fs1, fs2, alpha)
        result = max(result, dist)
    return gs.rnd(result)


def chaudhuri_rosenfeld(fs1, fs2):
    """Calculate the weighted average of Hausdorff distances."""
    top = Decimal(0)
    bottom = Decimal(0)
    for alpha in gs.get_y_points():
        dist = _hausdorff(fs1, fs2, alpha)
        top += alpha * dist
        bottom += alpha
    return gs.rnd(top / bottom)


def chaudhuri_rosenfeld_nn(fs1, fs2, e=0.5):
    """Calculate the weighted average of Hausdorff distances for non-normal."""
    def _normalise(fs):
        if fs.__class__.__name__ == 'DiscreteT1FuzzySet':
            fs.points = dict((x, y/fs.height) for x, y in fs.points.items())
        else:
            fs.membership_function.height = 1
    dist1 = 0
    n = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist1 += abs(y1 - y2)
        n += 1
    # normalise the fuzzy sets, use deepcopy to originals aren't altered
    fs1n = deepcopy(fs1)
    fs2n = deepcopy(fs2)
    _normalise(fs1n)
    _normalise(fs2n)
    dist2 = chaudhuri_rosenfeld(fs1n, fs2n)
    return gs.rnd(dist2 + (Decimal(e) * (dist1 / n)))


def _grzegorzewski_non_inf_p(fs1, fs2, p=2):
    """Use for Grzegorzewski distance where 1 <= p < infty."""
    def get_left_dist(alpha):
        fs1_l, fs1_u = fs1.calculate_alpha_cut(alpha)
        fs2_l, fs2_u = fs2.calculate_alpha_cut(alpha)
        return pow(fs1_l - fs2_l, p)

    def get_right_dist(alpha):
        fs1_l, fs1_u = fs1.calculate_alpha_cut(alpha)
        fs2_l, fs2_u = fs2.calculate_alpha_cut(alpha)
        return pow(fs1_u - fs2_u, p)
    left_dist, b = integrate.quad(get_left_dist, 0, 1)
    right_dist, b = integrate.quad(get_right_dist, 0, 1)
    return Decimal(left_dist), Decimal(right_dist)


def _grzegorzewski_inf_p(fs1, fs2):
    """Use for Grzegorzewski distance where p is infinity."""
    left_dist = Decimal(0)
    right_dist = Decimal(0)
    for alpha in gs.get_y_points():
        l, r = _minkowski_r1(fs1, fs2, alpha)
        left_dist = max(left_dist, l)
        right_dist = max(right_dist, r)
    return left_dist, right_dist


def grzegorzewski_non_inf_pq(fs1, fs2, p=2, q=0.5):
    """Grzegorzewski distance where 1 <= p < infty and q is used.

    q is used to weight the distance at alpha cuts.
    (1-q) weight for left distance, (q) weight for right distance.
    """
    p = Decimal(p)
    left_dist, right_dist = _grzegorzewski_non_inf_p(fs1, fs2, p)
    left_dist = (1 - Decimal(q)) * left_dist
    right_dist = 1 * right_dist
    distance = (left_dist + right_dist) ** (1 / p)
    return gs.rnd(distance)


def grzegorzewski_non_inf_p(fs1, fs2, p=2):
    """Grzegorzewski distance where 1 <= p < infty and q is not used."""
    p = Decimal(p)
    left_dist, right_dist = _grzegorzewski_non_inf_p(fs1, fs2, p)
    left_dist = left_dist ** (1 / p)
    right_dist = right_dist ** (1 / p)
    return gs.rnd(max(left_dist, right_dist))


def grzegorzewski_inf_q(fs1, fs2, q=0.5):
    """Grzegorzewski distance where p is infinity and q is used.

    q is used to weight the distance at alpha cuts.
    (1-q) weight for left distance, (q) weight for right distance.
    """
    q = Decimal(q)
    left_dist, right_dist = _grzegorzewski_inf_p(fs1, fs2)
    left_dist = (1 - q) * left_dist
    right_dist = q * right_dist
    return gs.rnd(left_dist + right_dist)


def grzegorzewski_inf(fs1, fs2):
    """Grzegorzewski distance where p is infinity and q is not used."""
    left_dist, right_dist = _grzegorzewski_inf_p(fs1, fs2)
    return gs.rnd(max(left_dist, right_dist))


def ban(fs1, fs2):
    """Minkowski based distance."""
    left_dist, right_dist = _grzegorzewski_non_inf_p(fs1, fs2, p=2)
    return gs.rnd((left_dist + right_dist) ** Decimal('0.5'))


def allahviranloo(fs1, fs2, c=0.5, f=lambda a: a):
    """Distance based on the average width and centre of the fuzzy sets."""
    def average_value(fs, alpha):
        l, r = fs.calculate_alpha_cut(alpha)
        return (Decimal(c) * l) + (Decimal(c) * r)

    def width(fs, alpha):
        l, r = fs.calculate_alpha_cut(alpha)
        return (r - l) * Decimal(f(alpha))
    fs1_average, b = integrate.quad(lambda y: average_value(fs1, y),
                                    0, 1, limit=50)
    fs2_average, b = integrate.quad(lambda y: average_value(fs2, y),
                                    0, 1, limit=50)
    fs1_width, b = integrate.quad(lambda y: width(fs1, y), 0, 1, limit=50)
    fs2_width, b = integrate.quad(lambda y: width(fs2, y), 0, 1, limit=50)
    centre_difference = (Decimal(fs1_average) - Decimal(fs2_average)) ** 2
    width_difference = (Decimal(fs1_width) - Decimal(fs2_width)) ** 2
    return gs.rnd(sqrt(centre_difference + width_difference))


def yao_wu(fs1, fs2):
    """Calculate the average Minkowski (r=1) distance."""
    def dist(alpha):
        fs1_min, fs1_max = fs1.calculate_alpha_cut(alpha)
        fs2_min, fs2_max = fs2.calculate_alpha_cut(alpha)
        diff = fs1_min + fs1_max - fs2_min - fs2_max
        return diff
    a, b = integrate.quad(dist, 0, 1)
    return gs.rnd(Decimal('0.5') * Decimal(a))


def _get_directional_distance(cut1, cut2):
    """Calcualte the directional minkwoski distance between cuts."""
    # Check they're continuous (i.e. not a tuple of tuples)
    if isinstance(cut1[0], Decimal) and isinstance(cut2[0], Decimal):
        return (cut2[0] + cut2[1] - cut1[0] - cut1[1]) / 2
    else:
        # Attempt to flatten the alpha cuts in case they are non-convex
        try:
            cut1 = [val for sublist in cut1 for val in sublist]
        except TypeError:
            pass
        try:
            cut2 = [val for sublist in cut2 for val in sublist]
        except TypeError:
            pass
        diffs = []
        # Take the lists in pairs
        for i in zip(cut1[0::2], cut1[1::2]):
            for j in zip(cut2[0::2], cut2[1::2]):
                diffs.append((j[0] + j[1] - i[0] - i[1]) / 2)
        return functools.reduce(lambda x, y: x + y, diffs) / len(diffs)


def mcculloch(fs1, fs2):
    """Calculate the weighted Minkowski (r=1) directional distance."""
    top = 0
    bottom = 0
    diff = 0
    for alpha in gs.get_y_points():
        try:
            cut1 = fs1.calculate_alpha_cut(alpha)
            cut2 = fs2.calculate_alpha_cut(alpha)
            diff = _get_directional_distance(cut1, cut2)
            top += alpha * diff
            bottom += alpha
        except AlphaCutError:
            pass
    return gs.rnd(top / bottom)
