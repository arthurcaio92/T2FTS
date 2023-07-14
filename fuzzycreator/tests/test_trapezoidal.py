"""This module is to test Trapezoidal membership function."""

import unittest

from fuzzycreator.membership_functions.trapezoidal import Trapezoidal
from fuzzycreator import global_settings as gs

t = Trapezoidal(1, 2, 2, 3)


class TrapezoidalTest(unittest.TestCase):
    """Test Trapezoidal membership function."""

    def setUp(cls):
        """Ensure correct rounding."""
        gs.set_rounding(4)

    def test_basic_membership_functions(self):
        """Test calculating membership."""
        self.assertTrue(t.calculate_membership(1.25) == 0.25)
        self.assertTrue(t.calculate_membership(1.5) == 0.5)
        self.assertTrue(t.calculate_membership(1.75) == 0.75)
        self.assertTrue(t.calculate_membership(2) == 1)
        self.assertTrue(t.calculate_membership(2.25) == 0.75)
        self.assertTrue(t.calculate_membership(2.5) == 0.5)
        self.assertTrue(t.calculate_membership(2.75) == 0.25)

    def test_alpha_cuts(self):
        """Test calculating alpha-cuts."""
        self.assertTrue(t.calculate_alpha_cut(0.25) == (1.25, 2.75))
        self.assertTrue(t.calculate_alpha_cut(0.5) == (1.5, 2.5))
        self.assertTrue(t.calculate_alpha_cut(0.75) == (1.75, 2.25))
        self.assertTrue(t.calculate_alpha_cut(1) == (2.0, 2.0))

    def test_zeros(self):
        """Test where values should be out of function range."""
        for i in range(-5, 2):
            self.assertTrue(t.calculate_membership(i) == 0)
        for i in range(3, 10):
            self.assertTrue(t.calculate_membership(i) == 0)

if __name__ == '__main__':
    unittest.main()
