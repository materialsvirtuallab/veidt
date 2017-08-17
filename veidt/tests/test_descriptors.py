# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import os

import numpy as np
import pandas as pd
from pymatgen import Structure

from veidt.descriptors import Generator, DistinctSiteProperty


class GeneratorTest(unittest.TestCase):

    def setUp(self):
        self.obj = pd.DataFrame(np.random.rand(4, 2), columns=['x', 'y'])
        self.labels = ["sum", "diff", "prod", "quot"]
        funcs = []
        funcs.append(lambda d: d['x'] + d['y'])
        funcs.append(lambda d: d['x'] - d['y'])
        funcs.append(lambda d: d['x'] * d['y'])
        funcs.append(lambda d: d['x'] / d['y'])
        self.funcs = funcs
        self.generator = Generator(funcs=funcs, labels=self.labels)

    def test_describe(self):
        results = self.generator.describe(self.obj)
        np.testing.assert_array_almost_equal(self.obj['x'] + self.obj['y'],
                                             results['sum'])
        np.testing.assert_array_almost_equal(self.obj['x'] - self.obj['y'],
                                             results['diff'])
        np.testing.assert_array_almost_equal(self.obj['x'] * self.obj['y'],
                                             results['prod'])
        np.testing.assert_array_almost_equal(self.obj['x'] / self.obj['y'],
                                             results['quot'])


class DistinctSitePropertyTest(unittest.TestCase):

    def setUp(self):
        self.li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                     "Li2O.cif"))
        self.na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                     "Na2O.cif"))
        self.describer = DistinctSiteProperty(['8c', '4a'],
                                              ["Z", "atomic_radius"])

    def test_describe(self):
        descriptor = self.describer.describe(self.li2o)
        self.assertAlmostEqual(descriptor["8c-Z"], 3)
        self.assertAlmostEqual(descriptor["8c-atomic_radius"], 1.45)
        descriptor = self.describer.describe(self.na2o)
        self.assertEqual(descriptor["4a-Z"], 8)
        self.assertEqual(descriptor["4a-atomic_radius"], 0.6)

    def test_describe_all(self):
        df = pd.DataFrame(self.describer.describe_all([self.li2o, self.na2o]))
        self.assertEqual(df.iloc[0]["8c-Z"], 3)
        self.assertEqual(df.iloc[0]["8c-atomic_radius"], 1.45)


if __name__ == "__main__":
    unittest.main()