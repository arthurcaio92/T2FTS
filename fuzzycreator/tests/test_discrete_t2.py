"""This module is to test DiscreteT2FuzzySet."""

import unittest
from decimal import Decimal

from fuzzycreator.fuzzy_sets.discrete_t2_fuzzy_set import DiscreteT2FuzzySet
from fuzzycreator.fuzzy_exceptions import AlphaCutError


class DiscreteT2Test(unittest.TestCase):
    """Test DiscreteT2FuzzySet."""

    def test_fuzzy_set(self):
        """Test basic calculations."""
        points = {1: {Decimal('0.6'): Decimal('0.3'),
                      Decimal('0.8'): Decimal('0.9'),
                      Decimal('0.9'): Decimal('0.7')},
                  2: {Decimal('0.4'): Decimal('0.6'),
                      Decimal('0.7'): Decimal('0.9')},
                  3: {Decimal('0.3'): Decimal('0.5'),
                      Decimal('0.4'): Decimal('1.0'),
                      Decimal('0.6'): Decimal('0.4')}}
        fsA = DiscreteT2FuzzySet(points)
        self.assertEqual(fsA.calculate_membership(1, 0.7),
                          (Decimal('0.8000'), Decimal('0.9000')))
        self.assertEqual(fsA.calculate_secondary_membership(1,
                                                             Decimal('0.8')),
                          Decimal('0.9'))
        self.assertRaises(AlphaCutError, fsA.calculate_alpha_cut_lower, 0.7)
        self.assertRaises(AlphaCutError, fsA.calculate_alpha_cut_upper, 1)
        self.assertEqual(fsA.calculate_alpha_cut_upper(0.6), [1, 3])
        self.assertEqual(fsA.calculate_alpha_cut_upper(0.7), [1, 2])
        self.assertEqual(fsA.calculate_alpha_cut_upper(0.8), [1, 1])

if __name__ == '__main__':
    unittest.main()
