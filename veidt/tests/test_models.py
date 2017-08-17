# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import os
import json

import numpy as np
from pymatgen import Structure

from veidt.descriptors import DistinctSiteProperty
from veidt.models import NeuralNet, LinearModel


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


class LinearModelTest(unittest.TestCase):

    def setUp(self):
        self.model = LinearModel(
            describer=DistinctSiteProperty(['8c'], ["Z"]))

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

    def test_serialize(self):
        json_str = json.dumps(self.model.as_dict())
        recover = LinearModel.from_dict(json.loads(json_str))
        self.assert_(True)

if __name__ == "__main__":
    unittest.main()
