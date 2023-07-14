"""This module is to test PollingT1FuzzySet."""

import unittest
from decimal import Decimal

from fuzzycreator import global_settings as gs
from fuzzycreator.fuzzy_sets.polling_t1_fuzzy_set import PollingT1FuzzySet
from fuzzycreator.fuzzy_sets.fuzzy_set import FuzzySet
from fuzzycreator.membership_functions.gaussian import Gaussian
from fuzzycreator.measures import distance_t1
from fuzzycreator.fuzzy_exceptions import AlphaCutError


class PollingTest(unittest.TestCase):
    """Test PollingT1FuzzySet."""

    def test_vertical(self):
        """Test linear interpolation."""
        points = {0: 0.1, 1: 0.5, 2: 1, 3: 0.5, 4: 1, 5: 0.5, 6: 0.1}
        fs1 = PollingT1FuzzySet(points)
        self.assertEqual(fs1.calculate_membership(1.5), Decimal('0.75'))
        self.assertEqual(fs1.calculate_membership(7), Decimal('0'))

    def test_alpha_cut(self):
        """Test convex and non-convex alpha-cuts."""
        gs.global_uod = (0, 10)
        gs.global_x_disc = 101
        points = {0: 0.1, 1: 0.5, 2: 1, 3: 0.5, 4: 1, 5: 0.5, 6: 0.1}
        fs1 = PollingT1FuzzySet(points)
        self.assertEqual(fs1.calculate_alpha_cut(1), [(2, 2), (4, 4)])
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('0.75')),
                          [(Decimal('1.5000'), Decimal('2.5000')),
                           (Decimal('3.5000'), Decimal('4.5000'))])
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('0.5')), [1, 5])
        fs2 = FuzzySet(Gaussian(5, 1))
        self.assertEqual(distance_t1.mcculloch(fs1, fs2), 2)

if __name__ == '__main__':
    unittest.main()
