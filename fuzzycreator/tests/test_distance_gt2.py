"""This module is to test distance_gt2."""

import unittest
from decimal import Decimal

from fuzzycreator.membership_functions.triangular import Triangular
from fuzzycreator.fuzzy_sets.general_t2_fuzzy_set import GeneralT2FuzzySet
from fuzzycreator.membership_functions.iaa import IntervalAgreementApproach
from fuzzycreator.fuzzy_sets.t2_aggregated_fuzzy_set import T2AggregatedFuzzySet
from fuzzycreator.measures import distance_gt2
from fuzzycreator import global_settings as gs


class DistanceGT2Test(unittest.TestCase):
    """Test distance_gt2."""

    def test_gt2_regular(self):
        """Test normal, convex fuzzy sets."""
        fs1 = GeneralT2FuzzySet(Triangular(1, 3, 5), Triangular(2, 3, 4))
        fs2 = GeneralT2FuzzySet(Triangular(5, 7, 9), Triangular(6, 7, 8))
        self.assertEqual(distance_gt2.mcculloch(fs1, fs2), 4)

    def test_gt2_non_normal(self):
        """Test non-normal, convex fuzzy sets."""
        fs1 = GeneralT2FuzzySet(Triangular(1, 3, 5, 0.9),
                                Triangular(2, 3, 4, 0.8))
        fs2 = GeneralT2FuzzySet(Triangular(5, 7, 9, 0.8),
                                Triangular(6, 7, 8, 0.7))
        self.assertEqual(distance_gt2.mcculloch(fs1, fs2), 4)

    def test_iaa(self):
        """Test IAA generated fuzzy set (non-normal and non-convex)."""
        mf1 = IntervalAgreementApproach(normalise=False)
        mf1.add_interval((Decimal('0.51'), Decimal('0.84')))
        mf1.add_interval((Decimal('0.42'), Decimal('0.77')))
        mf2 = IntervalAgreementApproach(normalise=False)
        mf2.add_interval((Decimal('0.23'), Decimal('0.87')))
        mf2.add_interval((Decimal('0.17'), Decimal('0.87')))
        mf3 = IntervalAgreementApproach(normalise=False)
        mf3.add_interval((Decimal('0.62'), Decimal('0.7')))
        mf3.add_interval((Decimal('0.31'), Decimal('0.91')))
        t2_fs = T2AggregatedFuzzySet()
        t2_fs.add_membership_function(mf1)
        t2_fs.add_membership_function(mf2)
        t2_fs.add_membership_function(mf3)
        self.assertEqual(distance_gt2.mcculloch(t2_fs, t2_fs), 0)

if __name__ == '__main__':
    unittest.main()
