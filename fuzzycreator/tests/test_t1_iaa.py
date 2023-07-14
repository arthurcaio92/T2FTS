"""This module is to test IntervalAgreementApproach."""

import unittest
from decimal import Decimal

from fuzzycreator.membership_functions.iaa import IntervalAgreementApproach
from fuzzycreator.fuzzy_sets.fuzzy_set import FuzzySet
from fuzzycreator import global_settings as gs




class Type1IAATest(unittest.TestCase):
    """Test IntervalAgreementApproach."""
    def setUp(cls):
        gs.global_uod = (0, 1)
        gs.global_x_disc = 101
        gs.set_rounding(4)

    def test_membership_normal_convex(self):
        """Test membership of a normal, convex fuzzy set."""
        mf1 = IntervalAgreementApproach(normalise=False)
        mf1.add_interval((Decimal('0.51'), Decimal('0.84')))
        mf1.add_interval((Decimal('0.42'), Decimal('0.77')))
        fs1 = FuzzySet(mf1)
        self.assertEqual(fs1.calculate_membership(0.6), 1)
        self.assertEqual(fs1.calculate_membership(0.48), Decimal('0.5'))

    def test_alpha_cut_convex(self):
        """Test alpha-cuts of a normal, convex fuzzy set."""
        mf1 = IntervalAgreementApproach(normalise=False)
        mf1.add_interval((Decimal('0.51'), Decimal('0.84')))
        mf1.add_interval((Decimal('0.42'), Decimal('0.77')))
        fs1 = FuzzySet(mf1)
        # Note: alpha cuts are always return witnin a list in case they
        # are non-convex.
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('0.25')),
                          [(Decimal('0.42'), Decimal('0.84'))])
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('0.5')),
                          [(Decimal('0.42'), Decimal('0.84'))])
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('0.75')),
                          [(Decimal('0.51'), Decimal('0.77'))])
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('1')),
                          [(Decimal('0.51'), Decimal('0.77'))])

    def test_alpha_cut_non_convex(self):
        """Test discontinuous alpha-cuts of non-convex fuzzy set."""
        mf1 = IntervalAgreementApproach(normalise=True)
        mf1.add_interval((Decimal('0.2'), Decimal('0.8')))
        mf1.add_interval((Decimal('0.3'), Decimal('0.5')))
        mf1.add_interval((Decimal('0.7'), Decimal('0.9')))
        fs1 = FuzzySet(mf1)
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('0.5')),
                          [(Decimal('0.2'), Decimal('0.9'))])
        self.assertEqual(fs1.calculate_alpha_cut(Decimal('1.0')),
                          [(Decimal('0.3'), Decimal('0.5')),
                           (Decimal('0.7'), Decimal('0.8'))])

    def test_membership_normality(self):
        """Test membership for normal and non-normal fuzzy sets."""
        mf1 = IntervalAgreementApproach(normalise=True)
        mf1.add_interval((Decimal('0.2'), Decimal('0.8')))
        mf1.add_interval((Decimal('0.3'), Decimal('0.5')))
        mf1.add_interval((Decimal('0.7'), Decimal('0.9')))
        fs1 = FuzzySet(mf1)
        self.assertEqual(fs1.calculate_membership(Decimal('0.15')), 0)
        self.assertEqual(fs1.calculate_membership(Decimal('0.25')),
                          Decimal('0.5'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.35')), 1)
        self.assertEqual(fs1.calculate_membership(Decimal('0.6')),
                          Decimal('0.5'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.7')), 1)
        self.assertEqual(fs1.calculate_membership(Decimal('0.85')),
                          Decimal('0.5'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.95')), 0)
        mf1.normalise = False
        self.assertEqual(fs1.calculate_membership(Decimal('0.15')), 0)
        self.assertEqual(fs1.calculate_membership(Decimal('0.25')),
                          Decimal('0.3333'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.35')),
                          Decimal('0.6667'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.6')),
                          Decimal('0.3333'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.7')),
                          Decimal('0.6667'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.85')),
                          Decimal('0.3333'))
        self.assertEqual(fs1.calculate_membership(Decimal('0.95')), 0)

    def test_centroid(self):
        """Test centroid calculation."""
        mf1 = IntervalAgreementApproach(normalise=True)
        mf1.add_interval((Decimal('0.2'), Decimal('0.8')))
        mf1.add_interval((Decimal('0.3'), Decimal('0.4')))
        mf1.add_interval((Decimal('0.6'), Decimal('0.7')))
        fs1 = FuzzySet(mf1)
        self.assertEqual(fs1.calculate_centroid(), Decimal('0.5'))
        fs1.normalise = False
        self.assertEqual(fs1.calculate_centroid(), Decimal('0.5'))
        mf1 = IntervalAgreementApproach(normalise=True)
        mf1.add_interval((Decimal('0.2'), Decimal('0.8')))
        mf1.add_interval((Decimal('0.4'), Decimal('0.6')))
        fs1 = FuzzySet(mf1)
        self.assertEqual(fs1.calculate_centroid(), Decimal('0.5'))
        fs1.normalise = False
        self.assertEqual(fs1.calculate_centroid(), Decimal('0.5'))


if __name__ == '__main__':
    unittest.main()
