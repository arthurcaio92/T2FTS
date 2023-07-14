"""This module is used to create a general type-2 fuzzy set."""

import inspect
from decimal import Decimal
from numpy import diff, linspace

from .interval_t2_fuzzy_set import IntervalT2FuzzySet
from .polling_t1_fuzzy_set import PollingT1FuzzySet
from .. import global_settings as gs
from .. import visualisations
from .. import visualisations_3d
from ..fuzzy_exceptions import AlphaCutError, ZLevelError


class GeneralT2FuzzySet():
    """Create a zSlices (alpha-plane) based general type-2 fuzzy set."""

    def __init__(self, mf1, mf2, zlevels_total=None, uod=None):
        """Create a general type-2 fuzzy set.

        mf1: first membership function object of lowest zslice
        mf2: second membership function object of lowest zslice
        zlevels_total: total number of zlevels
        uod: the universe of discourse indicated by a two-tuple.
        Note, the lower and upper membership functions may be assiged
        in any order to mf1 and mf2.
        """
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        if zlevels_total is None:
            self.zlevels_total = gs.global_zlevel_disc
        else:
            self.zlevels_total = zlevels_total
        self.zlevel_coords = []
        self._calculate_zlevel_coords()
        # Create the first zSlice to check it's valid,
        # then automatically generate the rest based on this.
        self.zslice_functions = {self.zlevel_coords[0]:
                                 IntervalT2FuzzySet(mf1, mf2)}
        self._generate_zslices()
        # A dict of z:mu pairs for the height of each zslice
        self.zslice_primary_heights = {}
        self._set_zslice_primary_heights()

    def _calculate_zlevel_coords(self):
        """Calculate the zlevel coordinates for each zslice."""
        # it's easier to construct the z-coordinates in reverse order
        self.zlevel_coords = [gs.rnd(Decimal(i)/self.zlevels_total)
                              for i in range(self.zlevels_total, 0, -1)]
        self.zlevel_coords.reverse()

    def _set_zslice_primary_heights(self):
        for z in self.zlevel_coords:
            lower_mf_height = min(self.zslice_functions[z].mf1.height,
                                  self.zslice_functions[z].mf2.height)
            upper_mf_height = max(self.zslice_functions[z].mf1.height,
                                  self.zslice_functions[z].mf2.height)
            self.zslice_primary_heights[z] = (lower_mf_height,
                                              upper_mf_height)

    def _generate_zslices(self):
        """Generate the zslices fuzzy sets."""
        # Find out what parameters the mf uses that will change
        # for each zslice.
        lowest_zslice = self.zslice_functions[self.zlevel_coords[0]]
        arg_names = inspect.getargspec(lowest_zslice.mf1.__init__)[0]
        arg_names.remove('self')
        args1 = [lowest_zslice.mf1.__dict__[var] for var in arg_names]
        args2 = [lowest_zslice.mf2.__dict__[var] for var in arg_names]
        shape = lowest_zslice.mf1.__class__
        # Calculate how much the MFs shift for each zslice.
        # coord_skew defines how much the coordinates of the UMF and
        # LMF change at each zSlice; 1 means no skew and 0 means
        # maximum skew. Think of it as a percentage of how far apart
        # the UMF and LMF will be from each other between each zLevel.
        if self.zlevels_total == 1:
            coord_skew = [1]
        else:
            x = self.zlevels_total - 1
            coord_skew = [Decimal(i)/x for i in range(x, -1, -1)]
        # Create the zslices IT2 FSs for each zlevel.
        # start from index 1 because __init__ already added the lowest zslice
        for z_index in range(1, len(coord_skew)):
            mf1_points = []
            mf2_points = []
            for i in range(len(args1)):
                spread = (args2[i] - args1[i]) / Decimal(2)
                altered_value = gs.rnd(spread * (1 - coord_skew[z_index]))
                mf1_points.append(args1[i] + altered_value)
                mf2_points.append(args2[i] - altered_value)
            try:
                zslice = IntervalT2FuzzySet(shape(*mf1_points),
                                            shape(*mf2_points),
                                            self.uod)
            except:
                #rounding error
                new_rounding = Decimal(str(gs.DECIMAL_ROUNDING*10)[:-1])
                mf1_points = [value.quantize(new_rounding)
                              for value in mf1_points]
                mf2_points = [value.quantize(new_rounding)
                              for value in mf2_points]
                zslice = IntervalT2FuzzySet(shape(*mf1_points),
                                            shape(*mf2_points),
                                            self.uod)
            self.zslice_functions[self.zlevel_coords[z_index]] = zslice

    def validate_zlevel(self, z):
        """Find the closest valid zlevel.

        Checks if the zlevel at z exists. If it exists then return z.
        If not, then return the closest zlevel that encompasses z.
        """
        z = gs.rnd(z)
        if z in self.zlevel_coords:
            return z
        else:
            points = self.zlevel_coords[:]
            if z > max(points):
                raise ZLevelError('zLevel ' + str(z) +
                                  ' is higher than the greatest zLevel at ' +
                                  str(max(points)))
            points.sort()
            for i in points:
                if i > z:
                    return i

    def calculate_membership(self, x, z):
        """Calculate the primary membership of x at the zlevel z."""
        z = self.validate_zlevel(z)
        return self.zslice_functions[z].calculate_membership(x)

    def calculate_secondary_membership(self, x, y):
        """Calculate the secondary membership of x at primary membership y."""
        y = Decimal('%.4f' % y)
        for z in sorted(self.zlevel_coords, reverse=True):
            y1, y2 = self.zslice_functions[z].calculate_membership(x)
            if y1 <= y <= y2:
                return z
        return Decimal(0)

    def calculate_alpha_cut_lower(self, alpha, z):
        """Calculate the alpha-cut of the lower membership function at z.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        z = self.validate_zlevel(z)
        return self.zslice_functions[z].calculate_alpha_cut_lower(alpha)

    def calculate_alpha_cut_upper(self, alpha, z):
        """Calculate the alpha-cut of the upper membership function at z.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        z = self.validate_zlevel(z)
        return self.zslice_functions[z].calculate_alpha_cut_upper(alpha)

    def calculate_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Uses the Karnik Mendel algorithm.
        Returns a two-tuple indicating the boundaries of the type-reduced set.
        """
        type_reduced_set = {}
        for z in self.zlevel_coords:
            l, r = self.zslice_functions[z].calculate_centre_of_sets()
            type_reduced_set[z] = (l, r)
        return type_reduced_set

    def calculate_overall_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Returns the centroid of the centre-of-sets type reduced result.
        """
        intervals = self.calculate_centre_of_sets()
        top = 0
        bottom = 0
        for z in intervals.keys():
            centre = (intervals[z][0] + intervals[z][1]) / Decimal(2)
            top += z * centre
            bottom += z
        return top / bottom

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        visualisations.plot_sets((self,), filename)

    def plot_set_3d(self):
        """Plot a 3-dimensional graph of the fuzzy set."""
        visualisations_3d.plot_sets((self,))
