# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import tempfile
import os
import shutil
import random
import json

import numpy as np
from monty.os.path import which
from monty.serialization import loadfn
from pymatgen import Structure
from veidt.potential.nnp import NNPotential

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.dirname(__file__), 'datapool.json'))

class NNPitentialTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(CWD)
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.potential = NNPotential(name='test')
        self.test_pool = test_datapool
        self.test_structures = []
        self.test_energies = []
        self.test_forces = []
        self.test_stresses = []
        for d in self.test_pool:
            self.test_structures.append(d['structure'])
            self.test_energies.append(d['outputs']['energy'])
            self.test_forces.append(d['outputs']['forces'])
            self.test_stresses.append(d['outputs']['virial_stress'])
        self.test_struct = d['structure']

    def test_write_read_cfgs(self):
        self.potential.write_cfgs('input.data', cfg_pool=self.test_pool)
        datapool, df = self.potential.read_cfgs('input.data')
        self.assertEqual(len(self.test_pool), len(datapool))
        for data1, data2 in zip(self.test_pool, datapool):
            struct1 = data1['structure']
            struct2 = Structure.from_dict(data2['structure'])
            self.assertTrue(struct1 == struct2)
            energy1 = data1['outputs']['energy']
            energy2 = data2['outputs']['energy']
            self.assertTrue(abs(energy1 - energy2) < 1e-3)
            forces1 = np.array(data1['outputs']['forces'])
            forces2 = data2['outputs']['forces']
            np.testing.assert_array_almost_equal(forces1, forces2)

    @unittest.skipIf(not which('RuNNerMakesym'), 'No RuNNerMakesym cmd found.')
    def test_generate_eta(self):
        for num_symm2 in range(2, 10):
            r_etas = self.potential.generate_eta(dmin=1.8, r_cut=4.0,
                                                 num_symm2=num_symm2)
            self.assertEqual(len(r_etas), num_symm2)

    @unittest.skipIf(not which('RuNNer'), 'No RuNNer cmd found.')
    @unittest.skipIf(not which('RuNNerMakesym'), 'No RuNNerMakesym cmd found.')
    def test_train(self):
        hidden_layers = [15, 15]
        activations = ['t'] * len(hidden_layers) + ['l']
        r_etas = self.potential.generate_eta(dmin=1.8, r_cut=4.0,
                                             num_symm2=5)
        a_etas = [0.01, 0.05]
        self.potential.train(train_structures=self.test_structures,
                             energies=self.test_energies,
                             forces=self.test_forces,
                             stresses=self.test_stresses,
                             atom_energy=-4.14, r_cut=4.0, r_etas=r_etas, a_etas=a_etas,
                             hidden_layers=hidden_layers, activations=activations,
                             scale_feature=False, scale_min_short_atomic=0,
                             scale_max_short_atomic=1, short_energy_error_threshold=0.0,
                             short_force_error_threshold=0.0, short_energy_fraction=1,
                             short_force_fraction=1, test_fraction=0.2,
                             force_update_scaling=-1, repeated_energy_update=False,
                             epochs=1)
        self.assertTrue(self.potential.lowest_energy)
        self.assertTrue(self.potential.train_energy_rmse)
        self.assertTrue(self.potential.train_forces_rmse)
        self.assertTrue(self.potential.validation_energy_rmse)
        self.assertTrue(self.potential.validation_forces_rmse)

    @unittest.skipIf(not which('RuNNer'), 'No RuNNer cmd found.')
    @unittest.skipIf(not which('RuNNerMakesym'), 'No RuNNerMakesym cmd found.')
    def test_evaluate(self):
        hidden_layers = [15, 15]
        activations = ['t'] * len(hidden_layers) + ['l']
        r_etas = self.potential.generate_eta(dmin=1.8, r_cut=4.0,
                                             num_symm2=5)
        a_etas = [0.01, 0.05]
        self.potential.train(train_structures=self.test_structures,
                             energies=self.test_energies,
                             forces=self.test_forces,
                             stresses=self.test_stresses,
                             atom_energy=-4.14, r_cut=4.0, r_etas=r_etas, a_etas=a_etas,
                             hidden_layers=hidden_layers, activations=activations,
                             scale_feature=False, scale_min_short_atomic=0,
                             scale_max_short_atomic=1, short_energy_error_threshold=0.0,
                             short_force_error_threshold=0.0, short_energy_fraction=1,
                             short_force_fraction=1, test_fraction=0.2,
                             force_update_scaling=-1, repeated_energy_update=False,
                             epochs=1)
        df_orig, df_tar = \
            self.potential.evaluate(test_structures=self.test_structures,
                                    ref_energies=self.test_energies,
                                    ref_forces=self.test_forces,
                                    ref_stresses=self.test_stresses,
                                    atom_energy=-4.14, r_cut=4.0, r_etas=r_etas, a_etas=a_etas,
                                    hidden_layers=hidden_layers, activations=activations,
                                    scale_feature=False, scale_min_short_atomic=0,
                                    scale_max_short_atomic=1, short_energy_error_threshold=0.0,
                                    short_force_error_threshold=0.0, short_energy_fraction=1,
                                    short_force_fraction=1, test_fraction=0.2,
                                    force_update_scaling=-1, repeated_energy_update=False)
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

if __name__ == '__main__':
    unittest.main()