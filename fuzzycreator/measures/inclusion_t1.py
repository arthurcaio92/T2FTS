"""This module contains inclusion (subsethood) measures for type-1 sets."""

from .. import global_settings as gs


def sanchez(fs1, fs2):
    """Calcualte the degree to which fs1 is contained within fs2."""
    inc1 = 0
    inc2 = 0
    for x in gs.get_x_points():
        mu1 = fs1.calculate_membership(x)
        inc1 += min(mu1, fs2.calculate_membership(x))
        inc2 += mu1
    return gs.rnd(inc1 / inc2)
