"""This module is used to create a discrete type-2 fuzzy set."""

from decimal import Decimal
from collections import defaultdict

from .discrete_t1_fuzzy_set import DiscreteT1FuzzySet
from .. import global_settings as gs
from .. import visualisations
from ..fuzzy_exceptions import AlphaCutError, ZLevelError


class DiscreteT2FuzzySet():
    """Create a discrete type-2 fuzzy set."""

    def __init__(self, points, uod=None):
        """Create a discrete type-1 fuzzy set using a dict as {x: {mu: z}}."""
        self.points = points
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        # A dict of z:mu pairs for the height of each zslice
        self.zlevel_coords = set([])
        for x in self.points.keys():
            self.zlevel_coords.update(self.points[x].values())
        self.zlevel_coords = sorted(list(self.zlevel_coords))
        self.zslice_primary_heights = dict((z, [0,0]) for z in self.zlevel_coords)
        for x in sorted(self.points.keys()):
            for z in self.zlevel_coords:
                l, u = self.calculate_membership(x, z)
                self.zslice_primary_heights[z][0] = max(self.zslice_primary_heights[z][0], l)
                self.zslice_primary_heights[z][1] = max(self.zslice_primary_heights[z][1], u)

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
        x = gs.rnd(x)
        try:
            y_list = []
            for y in self.points[x].keys():
                if self.points[x][y] >= z:
                    y_list.append(y)
            if len(y_list) == 0:
                return 0, 0
            else:
                return (gs.rnd(min(y_list)), gs.rnd(max(y_list)))
        except KeyError:
            return 0, 0

    def calculate_secondary_membership(self, x, mu):
        """Calculate the secondary membership of x at primary membership y."""
        try:
            return self.points[x][mu]
        except KeyError:
            return 0

    def calculate_alpha_cut_lower(self, alpha, z=0):
        """Calculate the alpha-cut of the lower membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        t1_points = {}
        for x in sorted(self.points.keys()):
            yl, yu = self.calculate_membership(x, z)
            if yu >= 0:
                t1_points[x] = yl
        t1_set = DiscreteT1FuzzySet(t1_points)
        return t1_set.calculate_alpha_cut(alpha)

    def calculate_alpha_cut_upper(self, alpha, z=0):
        """Calculate the alpha-cut of the lower membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        t1_points = {}
        for x in self.points.keys():
            yl, yu = self.calculate_membership(x, z)
            if yu >= 0:
                t1_points[x] = yu
        t1_set = DiscreteT1FuzzySet(t1_points)
        return t1_set.calculate_alpha_cut(alpha)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        visualisations.plot_sets((self,), filename)
