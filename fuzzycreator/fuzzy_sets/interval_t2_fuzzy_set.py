"""This module is used to create an interval type-2 fuzzy set."""

from decimal import Decimal
from numpy import diff, linspace

from .. import global_settings as gs
from .. import visualisations
from ..fuzzy_exceptions import AlphaCutError


class IntervalT2FuzzySet():
    """Create an interval type-2 fuzzy set."""

    def __init__(self, mf1, mf2, uod=None):
        """Create an interval type-2 fuzzy set.

        mf1: first membership function object
        mf2: second membership function object
        uod: the universe of discourse indicated by a two-tuple.
        Note, the lower and upper membership functions may be assigned
        in any order to mf1 and mf2.
        """
        if mf1.__class__ != mf2.__class__:
            raise Exception('Both membership functions ' +
                            'must be of the same type.')
        # special care has to be taken if using two Gaussian functions
        # with different means (unlike other cases the MFs will overlap)
        if (mf1.__class__.__name__ == 'Gaussian' and
                mf1.mean != mf2.mean):
            if mf1.height != mf2.height:
                raise Exception('Gaussian functions with different ' +
                                'mean values must have the same height.')
            self.gauss_diff_mean = True
            self.gauss_mean_values = (min(mf1.mean, mf2.mean),
                                      max(mf1.mean, mf2.mean))
            self.gauss_height = mf1.height
        else:
            self.gauss_diff_mean = False
            # check that one mf is a subset of the other
            if not ((mf1.x_min <= mf2.x_min and
                     mf1.x_max >= mf2.x_max and
                     mf1.height >= mf2.height) or
                    (mf1.x_min >= mf2.x_min and
                     mf1.x_max <= mf2.x_max and
                     mf2.height >= mf1.height)):
                raise Exception('One membership function must be a subset ' +
                                'of the other.')
        self.mf1 = mf1
        self.mf2 = mf2
        if uod is None:
            self.uod = gs.global_uod
        else:
            self.uod = uod

    def calculate_membership(self, x):
        """Calculate the membership of x within the uod.

        Returns a two-tuple (lower, upper) of Decimal values.
        """
        if x < self.uod[0] or x > self.uod[1]:
            return (Decimal(0), Decimal(0))
        y1 = self.mf1.calculate_membership(x)
        y2 = self.mf2.calculate_membership(x)
        if (self.gauss_diff_mean and
                x >= self.gauss_mean_values[0] and
                x <= self.gauss_mean_values[1]):
            return min(y1, y2), self.gauss_height
        else:
            return min(y1, y2), max(y1, y2)

    def calculate_alpha_cut_lower(self, alpha):
        """Calculate the alpha-cut of the lower membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        if self.gauss_diff_mean:
            if self.mf1.mean < self.mf2.mean:
                lower_r = self.mf1.calculate_alpha_cut(alpha)[1]
                lower_l = self.mf2.calculate_alpha_cut(alpha)[0]
        else:
            if self.mf1.x_min < self.mf2.x_min:
                lower_l, lower_r = self.mf2.calculate_alpha_cut(alpha)
            else:
                lower_l, lower_r = self.mf1.calculate_alpha_cut(alpha)
        return max(lower_l, self.uod[0]), min(lower_r, self.uod[1])

    def calculate_alpha_cut_upper(self, alpha):
        """Calculate the alpha-cut of the upper membership function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        if self.gauss_diff_mean:
            if self.mf1.mean < self.mf2.mean:
                cut = (self.mf1.calculate_alpha_cut(alpha)[0],
                       self.mf2.calculate_alpha_cut(alpha)[1])
            else:
                cut = (self.mf2.calculate_alpha_cut(alpha)[0],
                       self.mf1.calculate_alpha_cut(alpha)[1])
        else:
            if self.mf1.x_min < self.mf2.x_min:
                cut = self.mf1.calculate_alpha_cut(alpha)
            else:
                cut = self.mf2.calculate_alpha_cut(alpha)
        if isinstance(cut[0], Decimal):
            # convex cut
            return max(cut[0], self.uod[0]), min(cut[1], self.uod[1])
        else:
            # non-convex cut
            return [(max(subcut[0], self.uod[0]), min(subcut[1], self.uod[1]))
                    for subcut in cut]

    def plot_set(self, filename=None):
        """Plot a graph of the fuzzy set.

        If filename is None, the plot is displayed.
        If a filename is given, the plot is saved to the given location.
        """
        visualisations.plot_sets((self,), filename)

    def calculate_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Uses the Karnik Mendel algorithm.
        Returns a dict of two-tuples {z:(l, r)} indicating the
        boundaries  of the type-reduced set at each zlevel.
        """
        l = self._calculate_cos_boundary(right=False)
        r = self._calculate_cos_boundary(right=True)
        return l, r

    def calculate_overall_centre_of_sets(self):
        """Calculate centre-of-sets type reduction.

        Returns the centroid of the centre-of-sets type reduced result.
        """
        l, r = self.calculate_centre_of_sets()
        return (l + r) / Decimal(2)

    def _calculate_cos_boundary(self, right=True):
        """Compute the left or right boundary of the centre of sets.

        Uses the Karnik-Mendel centre-of-sets algorithm.
        right = True computes the right boundary,
        right = False computes the left centroid
        Process steps are as detailed in H. Hagras, "A hierarchical type-2
        fuzzy logic control architecture", IEEE Trans. Fuzz. Sys. 2004
        """
        x_values = [gs.rnd(x)
                    for x in linspace(self.uod[0], self.uod[1],
                                      gs.global_x_disc)]

        def h(x):
            return sum(self.calculate_membership(x)) / Decimal(2)

        def tri(x):
            y1, y2 = self.calculate_membership(x)
            return abs(y1 - y2) / Decimal(2)

        def find_e():
            """Find the index e where y_prime lies between e and e+1."""
            for e in range(len(x_values)-1):
                if x_values[e] <= y_prime and y_prime <= x_values[e+1]:
                    return e

        def get_double_prime():
            """Find the value of y_double_prime using steps 2 and 3."""
            # step 2
            e = find_e()
            # step 3
            top = Decimal(0)
            bottom = Decimal(0)
            for i in range(e+1):
                if right:
                    theta_value = (h(x_values[i]) - tri(x_values[i]))
                else:
                    theta_value = (h(x_values[i]) + tri(x_values[i]))
                top += (x_values[i] * theta_value)
                bottom += theta_value
            for i in range(e+1, len(x_values)):
                if right:
                    theta_value = (h(x_values[i]) + tri(x_values[i]))
                else:
                    theta_value = (h(x_values[i]) - tri(x_values[i]))
                top += (x_values[i] * theta_value)
                bottom += theta_value
            return gs.rnd(top / bottom)

        # step 1
        top = Decimal(0)
        bottom = Decimal(0)
        for x in x_values:
            top += (x * h(x))
            bottom += h(x)
        y_prime = gs.rnd(top / bottom)
        y_double_prime = 0
        while True:
            y_double_prime = get_double_prime()
            # step 4
            if y_prime == y_double_prime:
                return y_double_prime
            else:
                # step 5
                y_prime = y_double_prime
