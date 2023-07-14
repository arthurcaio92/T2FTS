"""This module contains distance measures for interval type-2 fuzzy set."""

from decimal import Decimal
from numpy import linspace
from scipy import integrate

from ..fuzzy_sets.fuzzy_set import FuzzySet
from . import distance_t1
from .. import global_settings as gs


def figueroa_garcia_alpha(fs1, fs2):
    """Calculate the absolute difference between alpha-cuts."""
    def dist(alpha):
        fs1_cut_lower = fs1.calculate_alpha_cut_lower(alpha)
        fs1_cut_upper = fs1.calculate_alpha_cut_upper(alpha)
        fs2_cut_lower = fs2.calculate_alpha_cut_lower(alpha)
        fs2_cut_upper = fs2.calculate_alpha_cut_upper(alpha)
        return (Decimal(str(alpha)) *
                (abs(fs1_cut_upper[0] - fs2_cut_upper[0]) +
                 abs(fs1_cut_lower[0] - fs2_cut_lower[0]) +
                 abs(fs1_cut_upper[1] - fs2_cut_upper[1]) +
                 abs(fs1_cut_lower[1] - fs2_cut_lower[1])))
    a, b = integrate.quad(dist, 0, 1)
    return gs.rnd(a)


def figueroa_garcia_centres_hausdorff(fs1, fs2):
    """Calculate the hausdorff distance between the centre-of-sets."""
    fs1_centre = fs1.calculate_centre_of_sets()
    fs2_centre = fs2.calculate_centre_of_sets()
    return max(abs(fs1_centre[0] - fs2_centre[0]),
               abs(fs1_centre[1] - fs2_centre[1]))


def figueroa_garcia_centres_minkowski(fs1, fs2):
    """Calculate the absolute difference between the centre-of-sets."""
    fs1_centre = fs1.calculate_centre_of_sets()
    fs2_centre = fs2.calculate_centre_of_sets()
    return (abs(fs1_centre[0] - fs2_centre[0]) +
            abs(fs1_centre[1] - fs2_centre[1]))


def mcculloch(fs1, fs2):
    """Calculate the weighted Minkowski (r=1) directional distance."""
    def order_lower_upper(fs):
        if ((fs.mf1.x_min <= fs.mf2.x_min and
                fs.mf1.x_max >= fs.mf2.x_max and
                fs.mf1.height >= fs.mf2.height)):
            return fs.mf1, fs.mf2
        else:
            return fs.mf2, fs.mf1
    fs1_lower_mf, fs1_upper_mf = order_lower_upper(fs1)
    fs2_lower_mf, fs2_upper_mf = order_lower_upper(fs2)
    return gs.rnd((distance_t1.mcculloch(FuzzySet(fs1_lower_mf),
                                         FuzzySet(fs2_lower_mf)) +
                   distance_t1.mcculloch(FuzzySet(fs1_upper_mf),
                                         FuzzySet(fs2_upper_mf))) /
                Decimal(2))
