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
        cls.obj = np.random.rand(10) * 10 - 5
        cls.func = "lambda x: x + 0.1"
        func_dict = {"np": "numpy.exp", "lambda": cls.func}
        cls.generator = Generator(func_dict=func_dict)

    def test_describe(self):
        results = self.generator.describe(self.obj)
        np.testing.assert_array_equal(np.exp(self.obj), results["np"])
        np.testing.assert_array_equal(self.obj + 0.1, results["lambda"])

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