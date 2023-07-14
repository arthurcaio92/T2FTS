"""This module is to test DiscreteT1FuzzySet."""

import unittest
from decimal import Decimal

from fuzzycreator.fuzzy_sets.discrete_t1_fuzzy_set import DiscreteT1FuzzySet
from fuzzycreator.fuzzy_exceptions import AlphaCutError


class DiscreteT1Test(unittest.TestCase):
    """Test DiscreteT1FuzzySet."""

    def test_fuzzy_set(self):
        """Test basic calculations."""
        points = {1: Decimal('0.33'),
                  2: Decimal('0.66'),
                  3: 1,
                  4: Decimal('0.66'),
                  5: Decimal('0.33')}
        fs = DiscreteT1FuzzySet(points)
        #fs.plot_set()
        self.assertEqual(fs.calculate_membership(3), 1)
        self.assertEqual(fs.calculate_alpha_cut(Decimal('0.33')), [1, 5])
        self.assertEqual(fs.calculate_alpha_cut(Decimal('0.5')), [2, 4])
        self.assertEqual(fs.calculate_alpha_cut(Decimal('0.66')), [2, 4])
        self.assertEqual(fs.calculate_alpha_cut(1), [3, 3])
        self.assertEqual(fs.calculate_centroid(), 3)

if __name__ == '__main__':
    unittest.main()
