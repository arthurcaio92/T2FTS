"""This module is to test distance_it2."""

import unittest
from decimal import Decimal

from fuzzycreator.membership_functions.triangular import Triangular
from fuzzycreator.fuzzy_sets.interval_t2_fuzzy_set import IntervalT2FuzzySet
from fuzzycreator.fuzzy_sets.polling_t1_fuzzy_set import PollingT1FuzzySet
from fuzzycreator.measures import distance_it2
from fuzzycreator import global_settings as gs


class DistanceIT2Test(unittest.TestCase):
    """Test distance_it2."""

    @classmethod
    def setUp(cls):
        """Initiate fuzzy sets for tests."""
        cls.A1 = IntervalT2FuzzySet(Triangular(2, 5, 9), Triangular(3, 5, 6))
        cls.A2 = IntervalT2FuzzySet(Triangular(0, 6, 9), Triangular(3, 6, 7))
        cls.A3 = IntervalT2FuzzySet(Triangular(0, 3, 7), Triangular(1, 3, 5))
        cls.A4 = IntervalT2FuzzySet(Triangular(0, 5, 9), Triangular(1, 5, 6))
        cls.A5 = IntervalT2FuzzySet(Triangular(0, 2, 8), Triangular(1, 2, 6))
        # The paper does calculations (including centre of sets) to 3 d.p.
        # but it gives results in 2 d.p.
        gs.set_rounding(3)
        gs.global_uod = (0, 10)
        gs.global_alpha_disc = 20

    def test_figueroa_garcia_centres_minkowski(self):
        """Test minkowski based approach."""
        results = [
            [Decimal('0'), Decimal('0.36'), Decimal('3.67'),
             Decimal('1.35'), Decimal('3.66')],
            [None, Decimal('0'), Decimal('3.96'),
             Decimal('1.63'), Decimal('3.94')],
            [None, None, Decimal('0'), Decimal('2.32'), Decimal('0.02')],
            [None, None, None, Decimal('0'), Decimal('2.31')],
            [None, None, None, None, Decimal('0')]]
        sets = [self.A1, self.A2, self.A3, self.A4, self.A5]
        for i in range(5):
            for j in range(i, 5):
                self.assertEqual(
                    (distance_it2.figueroa_garcia_centres_minkowski(sets[i],
                                                                    sets[j])
                     .quantize(Decimal('0.01'))),
                    results[i][j])

    def test_figueroa_garcia_alpha(self):
        """Test alpha-cut based approach."""
        results = [
            [Decimal('0'), Decimal('1.46'), Decimal('3.83'),
             Decimal('0.67'), Decimal('4.83')],
            [None, Decimal('0'), Decimal('5'), Decimal('1.83'), Decimal('6')],
            [None, None, Decimal('0'), Decimal('3.17'), Decimal('1.17')],
            [None, None, None, Decimal('0'), Decimal('4.17')],
            [None, None, None, None, Decimal('0')]]
        sets = [self.A1, self.A2, self.A3, self.A4, self.A5]
        for i in range(5):
            for j in range(i, 5):
                self.assertEqual(
                    (distance_it2.figueroa_garcia_alpha(sets[i], sets[j])
                     .quantize(Decimal('0.01'))),
                    results[i][j])

    def test_figueroa_garcia_centres_hausdorff(self):
        """Test Hausdorff based approach."""
        results = [
            [Decimal('0'), Decimal('0.32'), Decimal('2.01'),
             Decimal('0.68'), Decimal('1.99')],
            [None, Decimal('0'), Decimal('2.33'),
             Decimal('1'), Decimal('2.31')],
            [None, None, Decimal('0'), Decimal('1.33'), Decimal('0.02')],
            [None, None, None, Decimal('0'), Decimal('1.31')],
            [None, None, None, None, Decimal('0')]]
        sets = [self.A1, self.A2, self.A3, self.A4, self.A5]
        for i in range(5):
            for j in range(i, 5):
                self.assertEqual(
                    (distance_it2.figueroa_garcia_centres_hausdorff(sets[i],
                                                                    sets[j])
                     .quantize(Decimal('0.01'))),
                    results[i][j])

    def test_mcculloch(self):
        """Test own (alpha-cut based) approach."""
        A = IntervalT2FuzzySet(Triangular(1, 3, 5), Triangular(2, 3, 4))
        B = IntervalT2FuzzySet(Triangular(6, 8, 10), Triangular(7, 8, 9))
        self.assertEqual(distance_it2.mcculloch(A, B), Decimal('5'))
        B = IntervalT2FuzzySet(Triangular(5, 7, 9, 0.8),
                               Triangular(6, 7, 8, 0.7))
        self.assertEqual(distance_it2.mcculloch(A, B), Decimal('4'))
        B = IntervalT2FuzzySet(Triangular(5, 7, 10), Triangular(6, 7, 9))
        self.assertEqual(distance_it2.mcculloch(A, B), Decimal('4.158'))
        B = IntervalT2FuzzySet(Triangular(5, 7, 10, 0.7),
                               Triangular(6, 7, 9, 0.6))
        self.assertEqual(distance_it2.mcculloch(A, B), Decimal('4.154'))
        A = IntervalT2FuzzySet(Triangular(2, 6, 10), Triangular(3, 6, 9))
        B = IntervalT2FuzzySet(Triangular(4, 6, 8), Triangular(5, 6, 7))
        self.assertEqual(distance_it2.mcculloch(A, B), Decimal('0'))
        # test non-convex
        A = IntervalT2FuzzySet(Triangular(1, 3, 5), Triangular(2, 3, 4))
        BU_points = {5.25: 0, 6.25: 1, 7.5: 0.9, 8.75: 1, 9.75: 0}
        BU = PollingT1FuzzySet(BU_points)
        BL_points = {5.75: 0, 6.75: 0.8, 7.5: 0.7, 8.25: 0.8, 9.25: 0}
        BL = PollingT1FuzzySet(BL_points)
        B = IntervalT2FuzzySet(BU, BL)
        self.assertEqual(distance_it2.mcculloch(A, B), Decimal('4.500'))

if __name__ == '__main__':
    unittest.main()
