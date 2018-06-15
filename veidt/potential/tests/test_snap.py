# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import tempfile
import os
import re
import shutil
import json

import numpy as np
from monty.io import zopen
from monty.os.path import which
from monty.serialization import loadfn
from pymatgen import Structure
from veidt.potential.snap import SNAPotential
from veidt.model.linear_model import LinearModel
from veidt.describer.atomic_describer import BispectrumCoefficients


CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.dirname(__file__), 'datapool.json'))

class SNAPotentialTest(unittest.TestCase):

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
        profile = {'Mo': {'r': 0.6, 'w': 1.}}
        self.describer = BispectrumCoefficients(rcutfac=4.6, twojmax=6,
                                                element_profile=profile,
                                                pot_fit=True)
        model = LinearModel(describer=self.describer)
        self.potential = SNAPotential(model=model, name='test')
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

    def test_train(self):
        self.potential.train(structures=self.test_structures,
                             energies=self.test_energies,
                             forces=self.test_forces,
                             stresses=self.test_stresses)
        self.assertEqual(len(self.potential.model.coef),
                         len(self.describer.subscripts) + 1)

    def test_evaluate(self):
        self.potential.train(structures=self.test_structures,
                             energies=self.test_energies,
                             forces=self.test_forces,
                             stresses=self.test_stresses)
        df_orig, df_tar = self.potential.evaluate(test_structures=self.test_structures,
                                                  ref_energies=self.test_energies,
                                                  ref_forces=self.test_forces,
                                                  ref_stresses=self.test_stresses)
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

    @unittest.skipIf(not which('lmp_serial'), 'No LAMMPS cmd found.')
    def test_predict(self):
        self.potential.train(structures=self.test_structures,
                             energies=self.test_energies,
                             forces=self.test_forces,
                             stresses=self.test_stresses)
        energy, forces, stress = self.potential.predict(self.test_struct)
        self.assertEqual(len(forces), len(self.test_struct))
        self.assertEqual(len(stress), 6)

if __name__ == '__main__':
    unittest.main()