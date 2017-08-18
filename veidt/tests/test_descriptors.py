# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import os
import json

import numpy as np
import pandas as pd
from pymatgen import Structure

from veidt.descriptors import Generator, DistinctSiteProperty


class GeneratorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = np.random.rand(100, 3) * 10 - 5
        cls.df = pd.DataFrame(cls.data, columns=["x", "y", "z"])
        func_dict = {"sin": "np.sin",
                     "sum": "lambda d: d.sum(axis=1)",
                     "nest": "lambda d: np.log(np.exp(d['x']))"}
        cls.generator = Generator(func_dict=func_dict)

    def test_describe(self):
        results = self.generator.describe(self.df)
        np.testing.assert_array_equal(np.sin(self.data),
                                      results[["sin x", "sin y", "sin z"]])
        np.testing.assert_array_equal(np.sum(self.data, axis=1),
                                      results["sum"])
        np.testing.assert_array_almost_equal(self.data[:, 0],
                                             results["nest"])

    def test_serialize(self):
        json_str = json.dumps(self.generator.as_dict())
        recover = Generator.from_dict(json.loads(json_str))
        self.assert_(True)


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