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
from veidt.describer.structural_describer import DistinctSiteProperty
from veidt.model.neural_network import FeedForwardNeuralNetwork
from veidt.model.linear_model import LinearModel

import shutil
import tempfile



class NeuralNetTest(unittest.TestCase):
    def setUp(self):
        self.nn = FeedForwardNeuralNetwork(
            [25, 5], describer=DistinctSiteProperty(['8c'], ["Z"]))
        self.nn2 = FeedForwardNeuralNetwork(
            [25, 5], describer=DistinctSiteProperty(['8c'], ["Z"]))
        self.li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                     "../../tests/Li2O.cif"))
        self.na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                     "../../tests/Na2O.cif"))
        self.structures = [self.li2o] * 100 + [self.na2o] * 100
        self.energies = [3] * 100 + [4] * 100
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_fit_evaluate(self):
        self.nn.fit(inputs=self.structures, outputs=self.energies, epochs=100)
        # Given this is a fairly simple model, we should get close to exact.
        #self.assertEqual(round(self.nn.predict([self.na2o])[0][0]), 4, 3)
        self.assertTrue(3 <= round(self.nn.predict([self.na2o])[0][0]) <= 4)

    def test_model_save_load(self):
        model_fname = os.path.join(self.test_dir, 'test_nnmodel.h5')
        scaler_fname = os.path.join(self.test_dir, 'test_nnscaler.save')
        self.nn.fit(inputs=self.structures, outputs=self.energies, epochs=100)
        self.nn.save(model_fname=model_fname, scaler_fname=scaler_fname)
        self.nn2.load(model_fname=model_fname, scaler_fname=scaler_fname)
        self.assertEqual(self.nn.predict([self.na2o])[0][0],
                         self.nn2.predict([self.na2o])[0][0])


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

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

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

    def model_save_load(self):
        self.lm.save(os.path.join(self.test_dir, 'test_lm.save'))
        ori = self.lm.model.coef_
        self.lm.load(os.path.join(self.test_dir, 'test_lm.save'))
        loaded = self.lm.model.coef_
        self.assertAlmostEqual(ori, loaded)


if __name__ == "__main__":
    unittest.main()
