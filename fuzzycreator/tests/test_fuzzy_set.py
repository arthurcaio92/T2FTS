"""This module is to test FuzzySet."""

import unittest

from fuzzycreator.membership_functions.trapezoidal import Trapezoidal
from fuzzycreator.fuzzy_sets.fuzzy_set import FuzzySet


class FuzzySetTest(unittest.TestCase):
    """Test FuzzySet."""

    def test_uod_vertically(self):
        """Test membership values."""
        fs = FuzzySet(Trapezoidal(-2, -1, 1, 2))
        fs.uod = (0, 5)
        self.assertTrue(fs.calculate_membership(-2) == 0)
        self.assertTrue(fs.calculate_membership(-1) == 0)
        self.assertTrue(fs.calculate_membership(0) == 1)
        self.assertTrue(fs.calculate_membership(1) == 1)
        self.assertTrue(fs.calculate_membership(2) == 0)

    def test_uod_horizontally(self):
        """Test alpha-cuts."""
        fs = FuzzySet(Trapezoidal(-2, -1, 1, 2))
        fs.uod = (0, 5)
        self.assertTrue(fs.calculate_alpha_cut(1) == (0, 1))

    def test_centroid(self):
        """Test centroid."""
        fs = FuzzySet(Trapezoidal(1, 2, 3, 4))
        fs.uod = (0, 5)
        fs.global_x_disc = 51
        self.assertTrue(fs.calculate_centroid() == 2.5)


if __name__ == '__main__':
    unittest.main()
