"""This module is used to create a type-1 fuzzy set."""

from numpy import linspace
from decimal import Decimal

from .. import global_settings as gs
from .. import visualisations


class FuzzySet():
    """Create a type-1 fuzzy set."""

    def __init__(self, membership_function, uod=None):
        """Create a type-1 fuzzy set.

        membership_function: a membership function object.
        uod: the universe of discourse indicated by a two-tuple.
        """
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod
        self.membership_function = membership_function

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        Returns a Decimal value.
        """
        if x < self.uod[0] or x > self.uod[1]:
            return Decimal(0)
        mu = self.membership_function.calculate_membership(x)
        return mu

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function within the uod.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        cut = self.membership_function.calculate_alpha_cut(alpha)
        # make sure the alpha-cut is within the universe of discourse
        if isinstance(cut[0], Decimal):  # convex alpha-cut
            return max(cut[0], self.uod[0]), min(cut[1], self.uod[1])
        elif isinstance(cut[0], tuple):  # non-convex alpha-cut
            return [(max(segment[0], self.uod[0]),
                     min(segment[1], self.uod[1]))
                    for segment in cut]

    def calculate_centroid(self):
        """Calculate the centroid x-value of the fuzzy set."""
        top = 0
        bottom = 0
        for x in linspace(self.uod[0], self.uod[1], gs.global_x_disc):
            x = Decimal(str(x))
            mu = self.membership_function.calculate_membership(x)
            top += x * mu
            bottom += mu
        return gs.rnd(top / bottom)

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        visualisations.plot_sets((self,), filename)
