# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import os
import shutil
import tempfile

import unittest
from monty.serialization import loadfn
from veidt.potential.processing import pool_from, convert_docs

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.dirname(__file__), 'datapool.json'))

class PorcessingTest(unittest.TestCase):

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

    def test_pool_from(self):
        test_pool = pool_from(self.test_structures, self.test_energies,
                              self.test_forces, self.test_stresses)
        for p1, p2 in zip(test_pool, self.test_pool):
            self.assertEqual(p1['outputs']['energy'], p2['outputs']['energy'])
            self.assertEqual(p1['outputs']['forces'], p2['outputs']['forces'])
            self.assertEqual(p1['outputs']['virial_stress'],
                             p2['outputs']['virial_stress'])

    def test_convert_docs(self):
        tag_structures, df = convert_docs(self.test_pool)
        test_energies = df[df['dtype'] == 'energy']['y_orig']
        self.assertEqual(test_energies, self.test_energies)
        test_forces = df[df['dtype'] == 'force']
        for force1, force2 in zip(test_forces, self.test_forces):
            self.assertEqual(force1, force2)

if __name__ == '__main__':
    unittest.main()