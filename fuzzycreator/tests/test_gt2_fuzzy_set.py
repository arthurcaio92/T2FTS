"""This module is to test GeneralT2FuzzySet."""

import unittest
from decimal import Decimal

from fuzzycreator.membership_functions.triangular import Triangular
from fuzzycreator.membership_functions.trapezoidal import Trapezoidal
from fuzzycreator.membership_functions.gaussian import Gaussian
from fuzzycreator.fuzzy_sets.general_t2_fuzzy_set import GeneralT2FuzzySet
from fuzzycreator import global_settings as gs


class GT2FuzzySetTest(unittest.TestCase):
    """Test GeneralT2FuzzySet."""

    def setUp(cls):
        """Ensure rounding is correct for tests."""
        gs.set_rounding(4)

    def test_zlevels(self):
        """Test correct zlevels are calculated."""
        fs = GeneralT2FuzzySet(Triangular(2, 3, 4), Triangular(1, 3, 5), 3)
        self.assertEqual(fs.zlevel_coords, [Decimal('0.3333'),
                                            Decimal('0.6667'),
                                            Decimal('1.0000')])

    def test_membership(self):
        """Test membership values."""
        fs = GeneralT2FuzzySet(Triangular(1, 3, 5), Triangular(2, 3, 4), 3)
        self.assertEqual(fs.zslice_functions[1].mf1.x_min, Decimal('1.5'))
        self.assertEqual(fs.zslice_functions[1].mf1.centre, Decimal('3'))
        self.assertEqual(fs.zslice_functions[1].mf1.x_max, Decimal('4.5'))
        self.assertEqual(fs.zslice_functions[1].mf2.x_min, Decimal('1.5'))
        self.assertEqual(fs.zslice_functions[1].mf2.centre, Decimal('3'))
        self.assertEqual(fs.zslice_functions[1].mf2.x_max, Decimal('4.5'))
        self.assertEqual(fs.zslice_functions[Decimal('0.6667')].mf1.x_min,
                         Decimal('1.25'))
        self.assertEqual(fs.zslice_functions[Decimal('0.6667')].mf1.centre,
                         Decimal('3'))
        self.assertEqual(fs.zslice_functions[Decimal('0.6667')].mf1.x_max,
                         Decimal('4.75'))
        self.assertEqual(fs.zslice_functions[Decimal('0.6667')].mf2.x_min,
                         Decimal('1.75'))
        self.assertEqual(fs.zslice_functions[Decimal('0.6667')].mf2.centre,
                         Decimal('3'))
        self.assertEqual(fs.zslice_functions[Decimal('0.6667')].mf2.x_max,
                         Decimal('4.25'))
        self.assertEqual(fs.zslice_functions[Decimal('0.3333')].mf1.x_min,
                         Decimal('1'))
        self.assertEqual(fs.zslice_functions[Decimal('0.3333')].mf1.centre,
                         Decimal('3'))
        self.assertEqual(fs.zslice_functions[Decimal('0.3333')].mf1.x_max,
                         Decimal('5'))
        self.assertEqual(fs.zslice_functions[Decimal('0.3333')].mf2.x_min,
                         Decimal('2'))
        self.assertEqual(fs.zslice_functions[Decimal('0.3333')].mf2.centre,
                         Decimal('3'))
        self.assertEqual(fs.zslice_functions[Decimal('0.3333')].mf2.x_max,
                         Decimal('4'))


if __name__ == '__main__':
    unittest.main()
