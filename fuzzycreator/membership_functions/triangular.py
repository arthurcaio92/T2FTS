"""This module is used to create triangular membership functions."""

from decimal import Decimal

from .trapezoidal import Trapezoidal


class Triangular(Trapezoidal):
    """Create a triangular membership function."""

    def __init__(self, x_min, centre, x_max, height=1):
        """Create a triangular membership function.

        x_min: bottom left coordinate
        centre: x coordinate at the peak of the triangle
        x_max: bottom right coordinate
        height: highest membership at the centre.
        """
        self.centre = Decimal(centre)
        Trapezoidal.__init__(self, x_min, centre, centre, x_max, height)
