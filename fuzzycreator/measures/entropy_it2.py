"""This module contains inclusion (subsethood) measures for type-1 sets."""

from decimal import Decimal

from .. import global_settings as gs


def szmidt_pacprzyk(fs):
    """Calculate the ratio between the upper & lower membership functions."""
    ent1 = 0
    ent2 = 0
    for x in gs.get_x_points():
        l, u = fs.calculate_membership(x)
        ent1 += 1 - max(1 - u, l)
        ent2 += 1 - min(1 - u, l)
    return gs.rnd((ent1 / ent2) / Decimal(gs.global_x_disc))


def zeng_li(fs):
    """Calculate entroyp based on the sum of upper and lower memberships."""
    result = 0
    for x in gs.get_x_points():
        l, u = fs.calculate_membership(x)
        result += abs(u + l - 1)
    return gs.rnd(1 - (result / Decimal(gs.global_x_disc)))
