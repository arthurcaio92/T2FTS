"""This module is for applying the type-1 Interval Agreement Approach.

Details of the interval agreement approach are within
C. Wagner, S. Miller, J. M. Garibaldi, D. T. Anderson and T. C. Havens,
"From Interval-Valued Data to General Type-2 Fuzzy Sets,"
in IEEE Transactions on Fuzzy Systems, vol. 23, no. 2,
pp. 248-269, April 2015.
doi: 10.1109/TFUZZ.2014.2310734
"""

from decimal import Decimal
from numpy import linspace

from ..interval_dict import IntervalDict
from ..fuzzy_exceptions import AlphaCutError
from .. import visualisations
from .. import global_settings as gs


class IntervalAgreementApproach():
    """This class type-1 interval agreement approach membership function."""

    def __init__(self, normalise=False):
        """Create a membership function by the interval agreement approach."""
        self.normalise = normalise
        self.intervals = IntervalDict(overwrite_with_max=False)
        self._total_intervals = 0
        self._largest_value = 0  # largest value in the dict after summing
        self.height = 1

    def add_interval(self, interval):
        """Add an interval to the fuzzy set."""
        self.intervals[interval[0]:interval[1]] = 1
        self._total_intervals += 1
        self._largest_value = max([self.intervals[point] for point in
                                   self.intervals.singleton_keys()])
        self.height = self._largest_value / Decimal(self._total_intervals)

    def calculate_membership(self, x):
        """Calculate the membership of x. Returns a Decimal value."""
        if self.normalise:
            mu = Decimal(self.intervals[x]) / self._largest_value
        else:
            mu = Decimal(self.intervals[x]) / self._total_intervals
        return gs.rnd(mu)

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function.

        alpha must be greater than 0 and less than the function height.
        Returns a list containing two-tuples
        (a list of cuts is always given as any alpha-cut may be non-convex)
        """
        if not self.normalise and alpha > self.height:
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level', self.height)
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        x_values = sorted(self.intervals.singleton_keys())
        test_values = x_values[:]
        # add inbetween values to spot discontinous intervals
        for i in range(len(x_values)-1):
            test_values.insert((i+1) * 2-1,
                               ((x_values[i+1] + x_values[i]) / 2))
        alpha_intervals = []
        current_interval = []
        for x in test_values:
            if self.calculate_membership(x) >= alpha:
                current_interval.append(Decimal(x))
            else:
                if len(current_interval) != 0:
                    alpha_intervals.append((Decimal(current_interval[0]),
                                            Decimal(current_interval[-1])))
                    current_interval = []
        if len(current_interval) != 0:
            alpha_intervals.append((Decimal(current_interval[0]),
                                    Decimal(current_interval[-1])))
        return alpha_intervals
