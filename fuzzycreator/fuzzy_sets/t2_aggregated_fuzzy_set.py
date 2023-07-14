"""This module is for aggregating type-1 fuzzy sets into a type-2 fuzzy set.

The method of aggregation is the same as that used by the
interval aggrement approach, details of which are within
C. Wagner, S. Miller, J. M. Garibaldi, D. T. Anderson and T. C. Havens,
"From Interval-Valued Data to General Type-2 Fuzzy Sets,"
in IEEE Transactions on Fuzzy Systems, vol. 23, no. 2,
pp. 248-269, April 2015.
doi: 10.1109/TFUZZ.2014.2310734
"""

import inspect
from decimal import Decimal
from numpy import linspace

from .. import global_settings as gs
from .. import visualisations
from .. import visualisations_3d
from ..fuzzy_exceptions import AlphaCutError, ZLevelError


class T2AggregatedFuzzySet():
    """This class is for type-2 Interval Agreement Approach fuzzy sets."""

    def __init__(self, uod=None):
        """Initate a type-2 interval agreement approach fuzzy set."""
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        self.membership_functions = []
        self._total_membership_functions = 0
        self.zlevel_coords = []
        # A dict of z:mu pairs for the height of each zslice
        self.zslice_primary_heights = {}

    def add_membership_function(self, mf):
        """Add a type-1 membership function to the fuzzy set."""
        self.membership_functions.append(mf)
        self._total_membership_functions += 1
        # It's easier to calculate the zlevels in reverse order
        self.zlevel_coords = [gs.rnd(Decimal(i)/self._total_membership_functions)
                              for i in range(self._total_membership_functions,
                                             0, -1)]
        self.zlevel_coords.reverse()
        self._calculate_zslice_primary_heights()

    def _calculate_zslice_primary_heights(self):
        """Calculate the primary height of each zslice."""
        self.zslice_primary_heights = dict((z, (0, 0))
                                           for z in self.zlevel_coords)
        for z in self.zlevel_coords:
            self.zslice_primary_heights[z] = (
                    0, max([self.calculate_membership(x, z)[1]
                            for x in linspace(self.uod[0], self.uod[1],
                                              gs.global_x_disc)]))

    def validate_zlevel(self, z):
        """Find the closest valid zlevel.

        Checks if the zlevel at z exists. If it exists then return z.
        If not, then return the closest zlevel that encompasses z.
        """
        # Default to an interval type-2 fuzzy set if no zlevel is given.
        if z is None:
            return min(self.zlevel_coords)
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

    def calculate_membership(self, x, z=None):
        """Calculate the primary membership of x at the zlevel z.

        For an interval type-2 fuzzy set, leave z as None.
        """
        z = self.validate_zlevel(z)
        x = Decimal(str(x))
        y_values = sorted([mf.calculate_membership(x)
                           for mf in self.membership_functions])
        y_max = 0
        for y in y_values:
            if self.calculate_secondary_membership(x, y) >= z:
                y_max = y
        # lower membership is 0 as a type-2 IAA doesn't strictly have a
        # proper lower membership function.
        return Decimal(0), y_max

    def calculate_secondary_membership(self, x, y):
        """Calculate the secondary membership value for the given x and y."""
        if y == 0:
            return 0
        x = Decimal(str(x))
        y = Decimal(str(y))
        mfs_within = 0
        for mf in self.membership_functions:
            if y <= mf.calculate_membership(x):
                mfs_within += 1
        return gs.rnd(Decimal(mfs_within) / self._total_membership_functions)

    def calculate_alpha_cut_upper(self, alpha, z=None):
        """Calculate the alpha-cut of the lower membership function at z.

        alpha must be greater than 0 and less than the function height.
        For an interval type-2 fuzzy set, leave z as None.
        Returns a list containing two-tuples
        (a list of cuts is always given as any alpha-cut may be non-convex)
        """
        z = self.validate_zlevel(z)
        if alpha > self.zslice_primary_heights[z][1]:
            raise AlphaCutError('alpha level', alpha, 'is above max y level',
                                self.zslice_primary_heights[z][1])
        x_values = []
        for mf in self.membership_functions:
            x_values.extend(mf.intervals.singleton_keys())
        x_values = sorted(x_values)
        test_values = x_values[:]
        # add inbetween values to spot discontinous intervals
        for i in range(len(x_values)-1):
            test_values.insert((i+1) * 2-1,
                               ((x_values[i+1] + x_values[i]) / 2))
        alpha_intervals = []
        current_interval = []
        for x in test_values:
            if self.calculate_membership(x, z)[1] >= alpha:
                current_interval.append(Decimal(str(x)))
            else:
                if len(current_interval) != 0:
                    alpha_intervals.append((Decimal(current_interval[0]),
                                            Decimal(current_interval[-1])))
                    current_interval = []
        if len(current_interval) != 0:
            alpha_intervals.append((Decimal(current_interval[0]),
                                    Decimal(current_interval[-1])))
        return alpha_intervals

    def calculate_alpha_cut_lower(self, alpha, z):
        """Calculate the alpha-cut of the lower membership function at z.

        alpha must be greater than 0 and less than the function height.
        """
        # Type-2 IAA doesn't strictly have a proper lower membership function.
        return (Decimal(0), Decimal(0))

    def calculate_centroid(self):
        """Calculate the centroid of the fuzzy set.

        Calculate the centroid of each zslice and take the weighted average.
        """
        result_top = 0
        result_bottom = 0
        for z in self.zlevel_coords:
            slice_top = 0
            slice_bottom = 0
            for x in linspace(self.uod[0], self.uod[1], gs.global_x_disc):
                x = Decimal(str(x))
                mu = self.calculate_membership(x, z)[1]
                slice_top += Decimal(x) * mu
                slice_bottom += mu
            result_top += z * (slice_top / slice_bottom)
            result_bottom += z
        return gs.rnd(result_top / result_bottom)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        visualisations.plot_sets((self,), filename)

    def plot_set_3d(self):
        """Plot a graph of the fuzzy set."""
        visualisations_3d.plot_sets((self,))
