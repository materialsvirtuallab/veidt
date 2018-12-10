# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import tempfile
import os
import shutil

import yaml
import numpy as np
from monty.os.path import which
from pymatgen import Structure, Lattice, Element
from veidt.potential.snap import SNAPotential
from veidt.model.linear_model import LinearModel
from veidt.describer.atomic_describer import BispectrumCoefficients
from veidt.potential.lammps.calcs import \
    SpectralNeighborAnalysis, EnergyForceStress, ElasticConstant, LatticeConstant

CWD = os.getcwd()
with open(os.path.join(os.path.dirname(__file__), 'coeff.yaml')) as f:
    coeff, intercept = yaml.load(f)

class SpectralNeighborAnalysisTest(unittest.TestCase):

    @staticmethod
    def test_bs_subscripts():

        def from_lmp_doc(twojmax, diagonal):
            js = []
            for j1 in range(0, twojmax + 1):
                if diagonal == 2:
                    js.append([j1, j1, j1])
                elif diagonal == 1:
                    for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                        js.append([j1, j1, j])
                elif diagonal == 0:
                    for j2 in range(0, j1 + 1):
                        for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                            js.append([j1, j2, j])
                elif diagonal == 3:
                    for j2 in range(0, j1 + 1):
                        for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                            if j >= j1:
                                js.append([j1, j2, j])
            return js

        for d in range(4):
            for tjm in range(11):
                np.testing.assert_equal(
                    SpectralNeighborAnalysis.get_bs_subscripts(tjm, d),
                    from_lmp_doc(tjm, d))

    @unittest.skipIf(not which('lmp_serial'), 'No LAMMPS cmd found.')
    def test_calculate(self):
        s1 = Structure.from_spacegroup(227, Lattice.cubic(5.46873),
                                       ['Si'],
                                       [[0, 0, 0]])
        profile1 = dict(Si=dict(r=0.5, w=1.9), C=dict(r=0.5, w=2.55))
        diag1 = np.random.randint(4)
        tjm1 = np.random.randint(1, 11)
        calculator1 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm1,
                                               element_profile=profile1,
                                               diagonalstyle=diag1)
        sna1, snad1, snav1, elem1 = calculator1.calculate([s1])[0]
        n1 = calculator1.n_bs
        self.assertAlmostEqual(sna1[0][0], 585.920)
        self.assertEqual(sna1.shape, (len(s1), n1))
        self.assertEqual(snad1.shape, (len(s1), n1 * 3 * len(profile1)))
        self.assertEqual(snav1.shape, (len(s1), n1 * 6 * len(profile1)))
        self.assertEqual(len(np.unique(elem1)), 1)

        s2 = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                       ['Na', 'Cl'],
                                       [[0, 0, 0], [0, 0, 0.5]])
        profile2 = dict(Na=dict(r=0.3, w=0.9),
                        Cl=dict(r=0.7, w=3.0))
        diag2 = np.random.randint(4)
        tjm2 = np.random.randint(1, 11)
        calculator2 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm2,
                                               element_profile=profile2,
                                               diagonalstyle=diag2)
        sna2, snad2, snav2, elem2 = calculator2.calculate([s2])[0]
        n2 = calculator2.n_bs
        self.assertAlmostEqual(sna2[0][0], 525.858)
        self.assertEqual(sna2.shape, (len(s2), n2))
        self.assertEqual(snad2.shape, (len(s2), n2 * 3 * len(profile2)))
        self.assertEqual(snav2.shape, (len(s2), n2 * 6 * len(profile2)))
        self.assertEqual(len(np.unique(elem2)), len(profile2))

        s3 = Structure.from_spacegroup(221, Lattice.cubic(3.88947),
                                       ['Ca', 'Ti', 'O'],
                                       [[0.5, 0.5, 0.5],
                                        [0, 0, 0],
                                        [0, 0, 0.5]])
        profile3 = dict(Ca=dict(r=0.4, w=1.0),
                        Ti=dict(r=0.3, w=1.5),
                        O=dict(r=0.75, w=3.5))
        diag3 = np.random.randint(4)
        tjm3 = np.random.randint(1, 11)
        calculator3 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm3,
                                               element_profile=profile3,
                                               diagonalstyle=diag3)
        sna3, snad3, snav3, elem3 = calculator3.calculate([s3])[0]
        n3 = calculator3.n_bs
        self.assertAlmostEqual(sna3[0][0], 25506.3)
        self.assertEqual(sna3.shape, (len(s3), n3))
        self.assertEqual(snad3.shape, (len(s3), n3 * 3 * len(profile3)))
        self.assertEqual(snav3.shape, (len(s3), n3 * 6 * len(profile3)))
        self.assertEqual(len(np.unique(elem3)), len(profile3))

class EnergyForceStressTest(unittest.TestCase):

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

        element_profile = {'Ni': {'r': 0.5, 'w': 1}}
        describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8,
                                           element_profile=element_profile,
                                           pot_fit=True)
        model = LinearModel(describer=describer)
        model.model.coef_ = coeff
        model.model.intercept_ = intercept
        snap = SNAPotential(model=model)
        snap.specie = Element('Ni')
        self.struct = Structure.from_spacegroup('Fm-3m',
                                                Lattice.cubic(3.506),
                                                ['Ni'], [[0, 0, 0]])
        self.ff_settings = snap

    @unittest.skipIf(not which('lmp_serial'), 'No LAMMPS cmd found.')
    def test_calculate(self):
        calculator = EnergyForceStress(ff_settings=self.ff_settings)
        energy, forces, stresses = calculator.calculate([self.struct])[0]
        self.assertTrue(abs(energy - (-23.1242962)) < 1e-2)
        np.testing.assert_array_almost_equal(forces,
                                             np.zeros((len(self.struct), 3)))
        self.assertEqual(len(stresses), 6)


class ElasticConstantTest(unittest.TestCase):

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

        element_profile = {'Ni': {'r': 0.5, 'w': 1}}
        describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8,
                                           element_profile=element_profile,
                                           pot_fit=True)
        model = LinearModel(describer=describer)
        model.model.coef_ = coeff
        model.model.intercept_ = intercept
        snap = SNAPotential(model=model)
        snap.specie = Element('Ni')
        self.ff_settings = snap

    @unittest.skipIf(not which('lmp_serial'), 'No LAMMPS cmd found.')
    def test_calculate(self):
        calculator = ElasticConstant(ff_settings=self.ff_settings,
                                     lattice='fcc', alat=3.506)
        C11, C12, C44, bulkmodulus = calculator.calculate()
        self.assertTrue(abs(C11 - 276) / 276 < 0.1)
        self.assertTrue(abs(C12 - 159) / 159 < 0.1)
        self.assertTrue(abs(C44 - 132) / 132 < 0.1)


class LatticeConstantTest(unittest.TestCase):

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

        element_profile = {'Ni': {'r': 0.5, 'w': 1}}
        describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8,
                                           element_profile=element_profile,
                                           pot_fit=True)
        model = LinearModel(describer=describer)
        model.model.coef_ = coeff
        model.model.intercept_ = intercept
        snap = SNAPotential(model=model)
        snap.specie = Element('Ni')
        self.struct = Structure.from_spacegroup('Fm-3m',
                                                Lattice.cubic(3.506),
                                                ['Ni'], [[0, 0, 0]])
        self.ff_settings = snap

    @unittest.skipIf(not which('lmp_serial'), 'No LAMMPS cmd found.')
    def test_calculate(self):
        calculator = LatticeConstant(ff_settings=self.ff_settings)
        a, b, c = self.struct.lattice.abc
        calc_a, calc_b, calc_c = calculator.calculate([self.struct])[0]
        np.testing.assert_almost_equal(calc_a, a, decimal=2)
        np.testing.assert_almost_equal(calc_b, b, decimal=2)
        np.testing.assert_almost_equal(calc_c, c, decimal=2)


if __name__ == '__main__':
    unittest.main()