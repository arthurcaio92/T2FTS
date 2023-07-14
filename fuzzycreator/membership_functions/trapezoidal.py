"""This module is used to create trapezoidal membership functions."""

from decimal import Decimal

from .. import global_settings as gs
from ..fuzzy_exceptions import AlphaCutError


class Trapezoidal():
    """Create a trapezoidal membership function."""

    def __init__(self, x_min, x_top_left, x_top_right, x_max, height=1):
        """Set the Trapezoidal membership function.

        x_min_base: bottom left coordinate
        x_top_left: top left coordinate
        x_top_right: top right coordinate
        x_max_base: bottom right coordinate
        height: scale the maximum membership value
        """
        if height <= 0 or height > 1:
            raise Exception('height must be within the range (0, 1]')
        if not (x_min <= x_top_left <= x_top_right <= x_max):
            raise Exception('Values must be ordered such that ' +
                            'x_min_base <= x_top_left <= ' +
                            'x_top_right <= x_max_base')
        # First two variables are renamed so that every membership
        # function has x_min and x_max defining the boundaries.
        self.x_min = Decimal(str(x_min))
        self.x_max = Decimal(str(x_max))
        self.x_top_left = Decimal(str(x_top_left))
        self.x_top_right = Decimal(str(x_top_right))
        self.height = Decimal(str(height))

    def calculate_membership(self, x):
        """Calculate the membership of x. Returns a Decimal value."""
        x = Decimal(str(x))
        if self.x_min <= x and x <= self.x_top_left:
            try:
                return gs.rnd(self.height *
                              ((x - self.x_min) /
                               (self.x_top_left - self.x_min)))
            except:# ZeroDivisionError, DivisionByZero:
                # the base and top are the same resulting in a vertical line
                return self.height
        elif self.x_top_left <= x and x <= self.x_top_right:
            return self.height
        elif self.x_top_right <= x and x <= self.x_max:
            try:
                return gs.rnd(self.height *
                              ((self.x_max - x) /
                               (self.x_max - self.x_top_right)))
            except:# ZeroDivisionError, DivisionByZero:
                # the base and top are the same resulting in a vertical line
                return self.height
        else:
            return Decimal(0)

    def calculate_alpha_cut(self, alpha):
        """Calculate the alpha-cut of the function.

        alpha must be greater than 0 and less than the function height.
        Returns a two-tuple.
        """
        alpha = Decimal(str(alpha))
        if alpha > self.height:
            raise AlphaCutError(
                    'alpha level', alpha, 'is above max y level', self.height)
        if alpha == 0:
            raise AlphaCutError('There can be no alpha-cut where alpha=0.')
        left_point = (self.x_min +
                      (self.x_top_left - self.x_min) *
                      (alpha / self.height))
        right_point = (self.x_max +
                       (self.x_top_right - self.x_max) *
                       (alpha / self.height))
        return (gs.rnd(left_point), gs.rnd(right_point))

    def shift_membership_function(self, x):
        """Move the membership function along the x-axis by x-amount."""
        self.x_min += x
        self.x_max += x
        self.x_top_left += x
        self.x_top_right += x
