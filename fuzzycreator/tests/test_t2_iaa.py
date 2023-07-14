"""This module is to test T2AggregatedFuzzySet."""

import unittest
from decimal import Decimal

from fuzzycreator.membership_functions.iaa import IntervalAgreementApproach
from fuzzycreator.fuzzy_sets.t2_aggregated_fuzzy_set import T2AggregatedFuzzySet


class Type2IAATest(unittest.TestCase):
    """Test T2AggregatedFuzzySet."""

    def setUp(cls):
        """Set up fuzzy sets for tests."""
        cls.mf1 = IntervalAgreementApproach(normalise=False)
        cls.mf1.add_interval((Decimal('0.51'), Decimal('0.84')))
        cls.mf1.add_interval((Decimal('0.42'), Decimal('0.77')))
        cls.mf2 = IntervalAgreementApproach(normalise=False)
        cls.mf2.add_interval((Decimal('0.23'), Decimal('0.87')))
        cls.mf2.add_interval((Decimal('0.17'), Decimal('0.87')))
        cls.mf3 = IntervalAgreementApproach(normalise=False)
        cls.mf3.add_interval((Decimal('0.62'), Decimal('0.7')))
        cls.mf3.add_interval((Decimal('0.31'), Decimal('0.91')))
        cls.t2_fs = T2AggregatedFuzzySet()
        cls.t2_fs.add_membership_function(cls.mf1)
        cls.t2_fs.add_membership_function(cls.mf2)
        cls.t2_fs.add_membership_function(cls.mf3)

    def test_fuzzy_set(self):
        """Test alpha-cuts."""
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('0.5'), Decimal('0.3333')),
                          [(Decimal('0.17'), Decimal('0.91'))])
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('1'), Decimal('0.3333')),
                          [(Decimal('0.23'), Decimal('0.87'))])
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('0.5'), Decimal('0.6667')),
                          [(Decimal('0.31'), Decimal('0.87'))])
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('1'), Decimal('0.6667')),
                          [(Decimal('0.51'), Decimal('0.77'))])
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('0.5'), Decimal('1')),
                          [(Decimal('0.42'), Decimal('0.84'))])
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('1'), Decimal('1')),
                          [(Decimal('0.62'), Decimal('0.7'))])
        # default to lowest zlevel if none is given
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('0.5')),
                          [(Decimal('0.17'), Decimal('0.91'))])
        self.assertEqual(self.t2_fs.calculate_alpha_cut_upper(
                                    Decimal('1')),
                          [(Decimal('0.23'), Decimal('0.87'))])

    def test_zlevel_validation(self):
        """Test calculating closest zlevel when non-existent one is chosen."""
        self.assertEqual(self.t2_fs.validate_zlevel(Decimal('0.25')),
                          Decimal('0.3333'))
        self.assertEqual(self.t2_fs.validate_zlevel(Decimal('0.3333')),
                          Decimal('0.3333'))
        self.assertEqual(self.t2_fs.validate_zlevel(Decimal('0.5')),
                          Decimal('0.6667'))
        self.assertEqual(self.t2_fs.validate_zlevel(Decimal('0.6667')),
                          Decimal('0.6667'))
        self.assertEqual(self.t2_fs.validate_zlevel(Decimal('0.75')),
                          Decimal('1'))
        self.assertEqual(self.t2_fs.validate_zlevel(Decimal('1')),
                          Decimal('1'))

    def test_it2_membership(self):
        """Test calculating membership for IT2 FS."""
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.18')),
                          (Decimal(0), Decimal('0.5')))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.32')),
                          (Decimal(0), Decimal('1')))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.5')),
                          (Decimal(0), Decimal('1')))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.9')),
                          (Decimal(0), Decimal('0.5')))

    def test_gt2_membership(self):
        """Test calculating membership for GT2 FS."""
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.18'),
                                                          Decimal('0.6667')),
                          (Decimal(0), Decimal(0)))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.32'),
                                                          Decimal('0.6667')),
                          (Decimal(0), Decimal('0.5')))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.5'),
                                                          Decimal('0.6667')),
                          (Decimal(0), Decimal('0.5')))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.52'),
                                                          Decimal('0.6667')),
                          (Decimal(0), Decimal('1')))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.18'),
                                                          Decimal('1')),
                          (Decimal(0), Decimal(0)))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.32'),
                                                          Decimal('1')),
                          (Decimal(0), Decimal(0)))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.52'),
                                                          Decimal('1')),
                          (Decimal(0), Decimal('0.5')))
        self.assertEqual(self.t2_fs.calculate_membership(Decimal('0.65'),
                                                          Decimal('1')),
                          (Decimal(0), Decimal(1)))


if __name__ == '__main__':
    unittest.main()
