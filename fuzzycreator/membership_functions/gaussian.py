"""This module is used to create Gaussian membership functions."""

from numpy import e, power, log, sqrt
from decimal import Decimal

from .. import global_settings as gs
from ..fuzzy_exceptions import AlphaCutError


class Gaussian():
    """Create a Gaussian distribution."""

    def __init__(self, mean, std_dev, height=1):
        """Set the Gaussian membership function.

        height scales the height of the mean.
        """
        if height <= 0 or height > 1:
            raise Exception('height must be within the range (0, 1]')
        if std_dev <= 0:
            raise Exception('std_dev must be greater than 0')
        self.height = Decimal(str(height))
        self.std_dev = Decimal(str(std_dev))
        self.mean = Decimal(str(mean))
        self.x_min = None
        self.x_max = None
        self._set_function_end_points()

    def _set_function_end_points(self):
        """Set self.x_min and self.x_max.

        A gaussian function never approaches zero but it's helpful to
        define the end points of the function for various calculations.
        Typically, the spread is std_dev * 4.
        """
        self.x_min = self.mean - (self.std_dev * 4)
        self.x_max = self.mean + (self.std_dev * 4)

    def calculate_membership(self, x):
        """Calculate the membership of x. Returns a Decimal value."""
        x = Decimal(str(x))
        if x < self.x_min or x > self.x_max:
            return Decimal(0)
        y = self.height * power(
                    Decimal(e),
                    Decimal('-0.5') * (power(
                                       (x - self.mean) / self.std_dev,
                                       Decimal(2))))
        return gs.rnd(y)

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        if alpha > self.height:
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level', self.height)
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        # set alpha as float because log won't work with Decimal
        part = Decimal(sqrt(abs(log(float(alpha)) / 0.5))) * self.height
        left_point = (self.std_dev * -part) + self.mean
        right_point = (self.std_dev * part) + self.mean
        return (gs.rnd(left_point), gs.rnd(right_point))

    def shift_membership_function(self, x):
        """Move the membership function along the x-axis by x-amount."""
        self.mean += x
        self._set_function_end_points()
