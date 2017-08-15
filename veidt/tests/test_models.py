# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import os
import numpy as np

from pymatgen import Structure

from veidt.descriptors import DistinctSiteProperty
from veidt.models import NeuralNet, LRModel


class NeuralNetTest(unittest.TestCase):

    def setUp(self):
        self.model = NeuralNet(
            [20], describer=DistinctSiteProperty(['8c'], ["Z"]))

    def test_fit_evaluate(self):
        li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Li2O.cif"))
        na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Na2O.cif"))
        structures = [li2o] * 100 + [na2o] * 100
        energies = [3] * 100 + [4] * 100
        self.model.fit(inputs=structures, outputs=energies, epochs=100)
        # Given this is a fairly simple model, we should get close to exact.
        self.assertEqual(round(self.model.predict([na2o])[0][0]), 4, 3)



class LRModelTest(unittest.TestCase):

    def setUp(self):
        self.model = LRModel(
            describer=DistinctSiteProperty(['8c'], ["Z"]), fit_intercept=True)

    def test_fit_evaluate(self):
        li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Li2O.cif"))
        na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Na2O.cif"))
        structures = [li2o] * 100 + [na2o] * 100
        energies = [3] * 100 + [4] * 100
        energies += np.random.randn(200)
        self.model.fit(inputs=structures, outputs=energies)
        # Given this is a fairly simple model, we should get close to exact.
        self.assertEqual(round(self.model.predict([na2o])[0]), 4, 3)

if __name__ == "__main__":
    unittest.main()
