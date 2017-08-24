# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import os
import json

import numpy as np
import pandas as pd
from pymatgen import Structure

from veidt.abstract import Describer
from veidt.descriptors import DistinctSiteProperty
from veidt.models import NeuralNet, LinearModel
from monty.serialization import MontyEncoder, MontyDecoder


class NeuralNetTest(unittest.TestCase):

    def setUp(self):
        self.model = NeuralNet(
            [25, 5], describer=DistinctSiteProperty(['8c'], ["Z"]))

    def test_fit_evaluate(self):
        li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Li2O.cif"))
        na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Na2O.cif"))
        structures = [li2o] * 100 + [na2o] * 100
        energies = [3] * 100 + [4] * 100
        self.model.fit(inputs=structures, outputs=energies, nb_epoch=100)
        # Given this is a fairly simple model, we should get close to exact.
        self.assertEqual(round(self.model.predict([na2o])[0][0]), 4, 3)

        self.model.save("nntest.h5")
        self.assertTrue(os.path.exists("nntest.h5"))
        os.remove("nntest.h5")

    # def test_serialize(self):
    #     json_str = json.dumps(self.model, cls=MontyEncoder)
    #     recover = LinearModel.from_dict(json.loads(json_str, cls=MontyDecoder))
    #     self.assertIsNotNone(recover)


class LinearModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_train = np.random.rand(10, 2)
        cls.coef = np.random.rand(2)
        cls.intercept = np.random.rand()
        cls.y_train = cls.x_train.dot(cls.coef) + cls.intercept

    def setUp(self):
        class DummyDescriber(Describer):

            def describe(self, obj):
                pass

            def describe_all(self, n):
                return pd.DataFrame(n)

        self.lm = LinearModel(DummyDescriber())

    def test_fit_predict(self):
        self.lm.fit(inputs=self.x_train, outputs=self.y_train)
        x_test = np.random.rand(10, 2)
        y_test = x_test.dot(self.coef) + self.intercept
        y_pred = self.lm.predict(x_test)
        np.testing.assert_array_almost_equal(y_test, y_pred)
        np.testing.assert_array_almost_equal(self.coef, self.lm.coef)
        self.assertAlmostEqual(self.intercept, self.lm.intercept)

    def test_evaluate_fit(self):
        self.lm.fit(inputs=self.x_train, outputs=self.y_train)
        y_pred = self.lm.evaluate_fit()
        np.testing.assert_array_almost_equal(y_pred, self.y_train)

    def test_serialize(self):
        json_str = json.dumps(self.lm.as_dict())
        recover = LinearModel.from_dict(json.loads(json_str))
        self.assertIsNotNone(recover)

if __name__ == "__main__":
    unittest.main()
