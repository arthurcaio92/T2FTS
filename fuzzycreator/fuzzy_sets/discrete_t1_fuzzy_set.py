"""This module is used to create a discrete type-1 fuzzy set."""

from decimal import Decimal
import numpy as np

from .. import global_settings as gs
from ..fuzzy_exceptions import AlphaCutError, ZLevelError
from .. import visualisations


class DiscreteT1FuzzySet():
    """Create a discrete type-1 fuzzy set."""

    def __init__(self, points):
        """Create a discrete type-1 fuzzy set using a dict of x,mu pairs."""
        self.points = points
        self.x_min = min(points.keys())
        self.x_max = max(points.keys())
        self.height = max(points.values())

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        Returns a Decimal value.
        """
        #x = Decimal(str(x))
        try:
            return gs.rnd(self.points[x])
        except KeyError:
            return 0

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function within the uod.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        X_over_alpha = []
        alpha = gs.rnd(alpha)
        if alpha > max(self.points.values()):
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level',
                    max(self.points.values()))
        x_values = self.points.keys()
        x_values = sorted(x_values)
        # Record when the membership value passes the alpha cut
        results = []
        previous_hit = False
        for i in range(len(x_values)):
            if self.calculate_membership(x_values[i]) >= alpha:
                if not previous_hit:
                    results.append(gs.rnd(x_values[i]))
                    previous_hit = True
            else:
                if previous_hit:
                    results.append(gs.rnd(x_values[i-1]))
                previous_hit = False
        if self.calculate_membership(x_values[-1]) >= alpha and x_values[-1] not in results:
            results.append(gs.rnd(x_values[-1]))
        if len(results) == 1:
            results.append(results[0])
        if len(results) == 2:
            return results
        return list(zip(results[0::2], results[1::2]))


    def shift_membership_function(self, x):
        """Move the membership function along the x-axis by x-amount."""
        new_points = {}
        for k, v in self.points.items():
            new_points[gs.rnd(k + x)] = v
        self.points = new_points
        self.x_min = self.x_min + x
        self.x_max = self.x_max + x

    def calculate_centroid(self):
        """Calculate the centroid x-value of the fuzzy set."""
        top = 0
        bottom = 0
        for x in self.points.keys():
            mu = self.points[x]
            top += x * mu
            bottom += mu
        return gs.rnd(top / bottom)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        visualisations.plot_sets((self,), filename)
