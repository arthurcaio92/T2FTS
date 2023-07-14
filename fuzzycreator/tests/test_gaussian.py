"""This module is to test Gaussian membership function."""

import unittest

from fuzzycreator.membership_functions.gaussian import Gaussian


class GaussianTest(unittest.TestCase):
    """Test Gaussian membership function."""

    def test_basic_membership_functions(self):
        """Test basic membership values."""
        t = Gaussian(5, 1)
        self.assertTrue(t.calculate_membership(5) == 1)

    def test_alpha_cuts(self):
        """Test alpha-cut."""
        t = Gaussian(5, 1)
        self.assertTrue(t.calculate_alpha_cut(1) == (5, 5))

    def test_zeros(self):
        """Test where values should be out of function range."""
        t = Gaussian(5, 1)
        self.assertTrue(t.calculate_membership(0) == 0)
        self.assertTrue(t.calculate_membership(0.5) == 0)
        self.assertTrue(t.calculate_membership(0.9) == 0)
        self.assertTrue(t.calculate_membership(9.1) == 0)
        self.assertTrue(t.calculate_membership(9.5) == 0)

    def test_shift(self):
        """Test shifting the function along the x-axis."""
        t = Gaussian(5, 1)
        t.shift_membership_function(2)
        self.assertTrue(t.calculate_membership(2.9) == 0)
        self.assertTrue(t.calculate_membership(4) > 0)
        self.assertTrue(t.calculate_membership(7) == 1)
        self.assertTrue(t.calculate_membership(10) > 0)
        self.assertTrue(t.calculate_membership(11.1) == 0)
        t.shift_membership_function(-4)
        self.assertTrue(t.calculate_membership(-1.1) == 0)
        self.assertTrue(t.calculate_membership(1) > 0)
        self.assertTrue(t.calculate_membership(3) == 1)
        self.assertTrue(t.calculate_membership(5) > 0)
        self.assertTrue(t.calculate_membership(7.1) == 0)

if __name__ == '__main__':
    unittest.main()
