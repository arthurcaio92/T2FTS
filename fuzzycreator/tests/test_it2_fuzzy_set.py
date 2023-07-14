"""This module is to test IntervalT2FuzzySet."""

import unittest
from decimal import Decimal

from fuzzycreator.membership_functions.triangular import Triangular
from fuzzycreator.membership_functions.trapezoidal import Trapezoidal
from fuzzycreator.membership_functions.gaussian import Gaussian
from fuzzycreator.fuzzy_sets.interval_t2_fuzzy_set import IntervalT2FuzzySet
from fuzzycreator import global_settings as gs


class IT2FuzzySetTest(unittest.TestCase):
    """Test IntervalT2FuzzySet."""

    def setUp(cls):
        """Ensure correct rounding and UOD."""
        gs.set_rounding(4)
        gs.global_uod = (0, 10)

    def test_validation(self):
        """Test validation that one function is a subset of the other."""
        self.assertRaises(Exception,
                          IntervalT2FuzzySet,
                          Triangular(1, 3, 5),
                          Triangular(2, 3, 6))
        self.assertRaises(Exception,
                          IntervalT2FuzzySet,
                          Triangular(1, 3, 5),
                          Triangular(0, 3, 4))
        self.assertRaises(Exception,
                          IntervalT2FuzzySet,
                          Gaussian(5, 1, 1),
                          Gaussian(6, 1, 0.8))

    def test_membership(self):
        """Test membership values."""
        fs = IntervalT2FuzzySet(Triangular(1, 3, 5),
                                Triangular(2, 3, 4))
        self.assertEqual(fs.calculate_membership(2), (Decimal('0'),
                                                      Decimal('0.5')))
        self.assertEqual(fs.calculate_membership(2.5), (Decimal('0.5'),
                                                        Decimal('0.75')))
        self.assertEqual(fs.calculate_membership(3.5), (Decimal('0.5'),
                                                        Decimal('0.75')))

    def test_alpha_cut(self):
        """Test alpha-cuts."""
        fs = IntervalT2FuzzySet(Triangular(1, 3, 5),
                                Triangular(2, 3, 4))
        self.assertEqual(fs.calculate_alpha_cut_upper(0.5), (Decimal(2),
                                                             Decimal(4)))
        self.assertEqual(fs.calculate_alpha_cut_lower(0.5), (Decimal('2.5'),
                                                             Decimal('3.5')))

    def test_gauss_same_mean(self):
        """Test calculations for two gaussian MFs with same mean."""
        fs = IntervalT2FuzzySet(Gaussian(5, 0.5),
                                Gaussian(5, 1.0))
        self.assertEqual(fs.calculate_membership(5), (1.0, 1.0))
        self.assertEqual(fs.calculate_membership(4), (Decimal('0.1353'),
                                                      Decimal('0.6065')))
        self.assertEqual(fs.calculate_alpha_cut_upper(0.5),
                         (Decimal('3.8226'), Decimal('6.1774')))
        self.assertEqual(fs.calculate_alpha_cut_lower(0.5),
                         (Decimal('4.4113'), Decimal('5.5887')))

    def test_gauss_different_mean(self):
        """Test calculations for two gaussian MFs with different means."""
        # test that they must have the same scale
        self.assertRaises(Exception,
                          IntervalT2FuzzySet,
                          Gaussian(5, 0.5, 1.0),
                          Gaussian(6, 0.5, 0.8))
        fs = IntervalT2FuzzySet(Gaussian(5, 0.5),
                                Gaussian(6, 0.5))
        self.assertEqual(fs.calculate_membership(4),
                         (Decimal('0.0003'), Decimal('0.1353')))
        self.assertEqual(fs.calculate_membership(5.5),
                         (Decimal('0.6065'), Decimal('1.0')))
        self.assertEqual(fs.calculate_alpha_cut_lower(0.5),
                         (Decimal('5.4113'), Decimal('5.5887')))
        self.assertEqual(fs.calculate_alpha_cut_upper(0.5),
                         (Decimal('4.4113'), Decimal('6.5887')))

    def test_centre_of_sets(self):
        """Test centre-of-sets type reduction."""
        fs = IntervalT2FuzzySet(Trapezoidal(4, 7, 9, 10),
                                Trapezoidal(6, 7, 8, 9))
        l, r = fs.calculate_centre_of_sets()
        self.assertEqual(l, Decimal('6.8889'))
        self.assertEqual(r, Decimal('8.0000'))
        fs = IntervalT2FuzzySet(Triangular(0, 2, 4), Triangular(1, 2, 3))
        l, r = fs.calculate_centre_of_sets()
        self.assertEqual(l, Decimal('1.6646'))
        self.assertEqual(r, Decimal('2.3354'))
        # gauss different means
        fs = IntervalT2FuzzySet(Gaussian(4.5, 1), Gaussian(5.5, 1))
        l, r = fs.calculate_centre_of_sets()
        self.assertEqual(l, Decimal('4.4928'))
        self.assertEqual(r, Decimal('5.5072'))
        # gauss different std dev
        fs = IntervalT2FuzzySet(Gaussian(5, 1), Gaussian(5, 0.5))
        l, r = fs.calculate_centre_of_sets()
        self.assertEqual(l, Decimal('4.5997'))
        self.assertEqual(r, Decimal('5.4003'))

if __name__ == '__main__':
    unittest.main()
