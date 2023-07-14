"""This module is used to create a continuous version of a type-1 set."""

from bisect import bisect
from decimal import Decimal
from numpy import linspace
from scipy.interpolate import lagrange

from .. import global_settings as gs
from ..fuzzy_exceptions import AlphaCutError
from .. import visualisations

LAGRANGE = 0
LINEAR = 1

class PollingT1FuzzySet():
    """Create a type-1 fuzzy set using the polling technique."""

    def __init__(self, points, uod=None):
        """Create a discrete type-1 fuzzy set using a dict of x,mu pairs."""
        self.points = points
        self.x_min = min(self.points.keys())
        self.x_max = max(self.points.keys())
        self.height = max(self.points.values())
        self.interp_method = LINEAR
        # get lagrange polynomial
        X = [float(x) for x in sorted(points.keys())]
        Y = [float(points[x]) for x in sorted(points.keys())]
        self.lagrange_poly = lagrange(X, Y)
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        If x is not in self.points but exists between known x-values
        then linear interpolation is used to calculate its membership.
        Returns a Decimal value.
        """
        try:
            return gs.rnd(self.points[x])
        except KeyError:
            if self.interp_method is LINEAR:
                x_values = sorted(self.points.keys())
                if x < min(x_values) or x > max(x_values):
                    return Decimal(0)
                i = bisect(x_values, x)
                xl = x_values[i-1]
                xr = x_values[i]
                yl = self.points[xl]
                yr = self.points[xr]
                # cast as compatible data type
                if isinstance(xl, Decimal):
                    x = Decimal(x)
                else:
                    x = float(x)
                # Calculate the slope of the line from (xl, yl) to (xr, yr)
                slope = (yr - yl) / (xr - xl)
                return gs.rnd(slope * (x - xl) + yl)
            else:
                y = self.lagrange_poly(float(x))  # will not work with Decimal
                return gs.rnd(str(y))
                #return y

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function within the uod.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        alpha = gs.rnd(alpha)
        if len(self.points) == 0:
            raise Exception('There are no elements within this fuzzy set')
        if alpha > max(self.points.values()):
            raise AlphaCutError('alpha level', alpha, 'is above max y level',
                                max(self.points.values()))
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        # create x_values based on the global settings and include the
        # x values used in the fuzzy set
        x_values = list(gs.get_x_points())
        for x in self.points.keys():
            if x not in x_values:
                x_values.append(x)
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
        if len(results) == 1:
            results.append(results[0])
        if len(results) == 2:
            return results
        return list(zip(results[0::2], results[1::2]))

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
