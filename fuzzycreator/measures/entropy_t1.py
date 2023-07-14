"""This module contains inclusion (subsethood) measures for type-1 sets."""

from .. import global_settings as gs


def kosko(fs):
    """Calculate the degree to which the fuzzy set is fuzzy."""
    inc1 = 0
    inc2 = 0
    for x in gs.get_x_points():
        mu = fs.calculate_membership(x)
        inc1 += min(mu, 1 - mu)
        inc2 += max(mu, 1 - mu)
    return gs.rnd(inc1 / inc2)
