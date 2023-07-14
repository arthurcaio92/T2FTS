"""This module contains similarity measures for type-1 fuzzy set."""

from numpy import linspace, e
from decimal import Decimal

import global_settings as gs


def pappis1(fs1, fs2):
    """Based on the maximum distance between membership values."""
    dist = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist = max(dist, abs(y1 - y2))
    return gs.rnd(1 - dist)


def pappis2(fs1, fs2):
    """The ratio between the negation and addition of membership values."""
    dist1 = 0
    dist2 = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist1 += abs(y1 - y2)
        dist2 += y1 + y2
    return gs.rnd(1 - (dist1 / dist2))


def pappis3(fs1, fs2):
    """Based on the average difference between membership values."""
    dist = 0
    n = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        dist += abs(y1 - y2)
        n += 1
    return gs.rnd(1 - (dist / n))


def jaccard(fs1, fs2):
    """Ratio between the intersection and union of the fuzzy sets."""
    sim1 = 0
    sim2 = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        sim1 += min(y1, y2)
        sim2 += max(y1, y2)
    return gs.rnd(sim1 / sim2)


def dice(fs1, fs2):
    """Based on the ratio between the intersection and cardinality."""
    sim1 = 0
    sim2 = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        sim1 += Decimal(2) * (min(y1, y2))
        sim2 += y1 + y2
    return gs.rnd(sim1 / sim2)


def zwick(fs1, fs2):
    """The maximum membership of the intersection of the fuzzy sets."""
    sim = Decimal(0)
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        sim = max(sim, min(y1, y2))
    return gs.rnd(sim)


def chen(fs1, fs2):
    """Ratio between the product of memberships and the cardinality."""
    top = 0
    fs1_squares = 0
    fs2_squares = 0
    for x in gs.get_x_points():
        y1 = fs1.calculate_membership(x)
        y2 = fs2.calculate_membership(x)
        top += (y1 * y2)
        fs1_squares += (y1 * y1)
        fs2_squares += (y2 * y2)
    return gs.rnd(top / max(fs1_squares, fs2_squares))


def vector(fs1, fs2):
    """Vector similarity based on the distance and similarity of shapes."""
    x_min = min(fs1.membership_function.x_min, fs2.membership_function.x_min)
    x_max = max(fs1.membership_function.x_max, fs2.membership_function.x_max)
    r = Decimal(4) / (x_max - x_min)
    dist = fs1.calculate_centroid() - fs2.calculate_centroid()
    # temporarily align the centroid of fs2 with fs1 to compare shapes
    fs2.membership_function.shift_membership_function(dist)
    sim1 = jaccard(fs1, fs2)
    # put the fuzzy set fs2 back to where it was
    fs2.membership_function.shift_membership_function(-dist)
    sim2 = pow(Decimal(e), -r * abs(dist))
    return gs.rnd(sim1 * sim2)
