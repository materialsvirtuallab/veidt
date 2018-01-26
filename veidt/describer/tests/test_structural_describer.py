# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import os
import pandas as pd
from pymatgen import Structure
from veidt.describer.structural_describer import DistinctSiteProperty


class DistinctSitePropertyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                    "../../tests/Li2O.cif"))
        cls.na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                    "../../tests/Na2O.cif"))
        cls.describer = DistinctSiteProperty(['8c', '4a'],
                                              ["Z", "atomic_radius"])

    def test_describe(self):
        descriptor = self.describer.describe(self.li2o)
        self.assertAlmostEqual(descriptor.iloc[0]["8c-Z"], 3)
        self.assertAlmostEqual(descriptor.iloc[0]["8c-atomic_radius"], 1.45)
        descriptor = self.describer.describe(self.na2o)
        self.assertEqual(descriptor.iloc[0]["4a-Z"], 8)
        self.assertEqual(descriptor.iloc[0]["4a-atomic_radius"], 0.6)

    def test_describe_all(self):
        df = pd.DataFrame(self.describer.describe_all([self.li2o, self.na2o]))
        print(df)
        self.assertEqual(df.iloc[0]["8c-Z"], 3)
        self.assertEqual(df.iloc[0]["8c-atomic_radius"], 1.45)


if __name__ == "__main__":
    unittest.main()
