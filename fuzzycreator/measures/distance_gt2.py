"""This module contains distance measures for general type-2 fuzzy set."""

from decimal import Decimal

from . import distance_t1
from .. import global_settings as gs
from ..fuzzy_exceptions import ZLevelError, AlphaCutError


def _zslice_distance(fs1, fs2, z):
    """Calcualte the directional minkwoski distance between zslices."""
    top = bottom = Decimal(0)
    count = True
    z1 = fs1.validate_zlevel(z)
    z2 = fs2.validate_zlevel(z)
    mfs = ((fs1.calculate_alpha_cut_lower, fs2.calculate_alpha_cut_lower),
           (fs1.calculate_alpha_cut_upper, fs2.calculate_alpha_cut_upper))
    mf_maxes = ((fs1.zslice_primary_heights[z1][0],
                 fs2.zslice_primary_heights[z2][0]),
                (fs1.zslice_primary_heights[z1][1],
                 fs2.zslice_primary_heights[z2][1]))
    if (fs1.zslice_primary_heights[z1][1] == 0 or
            fs2.zslice_primary_heights[z1][1] == 0):
        raise ZLevelError('Empty zSlice.')
    for mf_index in (0, 1):
        mf_pair = mfs[mf_index]
        for alpha in gs.get_y_points():
            try:
                count = True
                fs1_cut = mf_pair[0](alpha, z)
                fs2_cut = mf_pair[1](alpha, z)
                diff = distance_t1._get_directional_distance(fs1_cut, fs2_cut)
            except AlphaCutError:
                # If both sets have a null alpha cut then ignore it
                # (count=False). If only one is empty then replace it with
                # the cut at its height.
                count = False
                try:
                    fs1_cut = mf_pair[0](alpha, z)
                    count = True
                except AlphaCutError:
                    alpha2 = mf_maxes[mf_index][0]
                    fs1_cut = mf_pair[0](alpha2, z)
                try:
                    fs2_cut = mf_pair[1](alpha, z)
                    count = True
                except AlphaCutError:
                    alpha2 = mf_maxes[mf_index][1]
                    fs2_cut = mf_pair[1](alpha2, z)
                if count:
                    diff = distance_t1._get_directional_distance(fs1_cut,
                                                                 fs2_cut)
            # count is zero if the alpha cut is empty for both fuzzy sets
            if count:
                top += Decimal(str(alpha)) * diff
                bottom += alpha
    return gs.rnd(top / bottom)


def mcculloch(fs1, fs2):
    """Calculate the weighted Minkowski (r=1) directional distance."""
    top = 0
    bottom = 0
    for z in gs.get_z_points():
        try:
            # Note: You can't just compare zslice_functions using distance_it2
            # as IAA and discrete sets aren't easily constructed that way.
            dist = _zslice_distance(fs1, fs2, z)
            top += z * dist
            bottom += z
        except ZLevelError:
            pass  # there is nothing at this zSlice
    return gs.rnd(top / bottom)
