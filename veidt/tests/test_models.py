# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import
import shutil, tempfile
import unittest
import os
import json

import numpy as np
from pymatgen import Structure

from veidt.descriptors import DistinctSiteProperty
from veidt.models import NeuralNet, LinearModel


class NeuralNetTest(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNet(
            [20], describer=DistinctSiteProperty(['8c'], ["Z"]))
        self.nn2 = NeuralNet(
            [20], describer=DistinctSiteProperty(['8c'], ["Z"]))
        self.li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Li2O.cif"))
        self.na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Na2O.cif"))
        self.structures = [self.li2o] * 100 + [self.na2o] * 100
        self.energies = [3] * 100 + [4] * 100
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_fit_evaluate(self):

        self.nn.fit(inputs=self.structures, outputs=self.energies, epochs=100)
        # Given this is a fairly simple model, we should get close to exact.
        self.assertEqual(round(self.nn.predict([self.na2o])[0][0]), 4, 3)

    def test_model_save_load(self):
        model_fname = os.path.join(self.test_dir, 'test_nnmodel.h5')
        scaler_fname = os.path.join(self.test_dir, 'test_nnscaler.save')
        self.nn.fit(inputs=self.structures, outputs=self.energies, epochs=100)
        self.nn.model_save(model_fname=model_fname, scaler_fname=scaler_fname)
        self.nn2.model_load(model_fname=model_fname, scaler_fname=scaler_fname)
        self.assertEqual(self.nn.predict([self.na2o])[0][0], self.nn2.predict([self.na2o])[0][0])


class LinearModelTest(unittest.TestCase):
    def setUp(self):
        self.lm = LinearModel(
            describer=DistinctSiteProperty(['8c'], ["Z"]))
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_fit_evaluate(self):
        li2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Li2O.cif"))
        na2o = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                "Na2O.cif"))
        structures = [li2o] * 100 + [na2o] * 100
        energies = [3] * 100 + [4] * 100
        energies += np.random.randn(200)
        self.lm.fit(inputs=structures, outputs=energies)
        # Given this is a fairly simple model, we should get close to exact.
        self.assertEqual(round(self.lm.predict([na2o])[0]), 4, 3)

    def test_serialize(self):
        json_str = json.dumps(self.lm.as_dict())
        recover = LinearModel.from_dict(json.loads(json_str))
        self.assert_(True)

    def model_save_load(self):
        self.lm.model_save(os.path.join(self.test_dir, 'test_lm.save'))
        ori = self.lm.model.coef_
        load_m = self.lm.model_load(os.path.join(self.test_dir, 'test_lm.save'))
        loaded = self.lm.model.coef_
        self.assertAlmostEqual(ori, loaded)



if __name__ == "__main__":
    unittest.main()
