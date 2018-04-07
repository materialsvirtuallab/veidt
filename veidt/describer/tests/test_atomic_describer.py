# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import random

import numpy as np
from monty.os.path import which
from pymatgen import Lattice, Structure, Element

from veidt.describer.atomic_describer import BispectrumCoefficients
from veidt.describer.atomic_describer import CoulombMatrix


class BispectrumCoefficientsTest(unittest.TestCase):

    @staticmethod
    def test_subscripts():

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

        profile = {"Mo": {"r": 0.5, "w": 1}}
        for d in range(4):
            for tjm in range(11):
                bc = BispectrumCoefficients(1.0, twojmax=tjm,
                                            element_profile=profile,
                                            diagonalstyle=d)
                np.testing.assert_equal(bc.subscripts, from_lmp_doc(tjm, d))

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found")
    def test_describe(self):
        s = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                      ["Na", "Cl"],
                                      [[0, 0, 0], [0, 0, 0.5]])
        profile = dict(Na=dict(r=0.5, w=1.0),
                       Cl=dict(r=0.5, w=0.8))
        bc = BispectrumCoefficients(rcutfac=5, twojmax=4,
                                    element_profile=profile,
                                    diagonalstyle=3)
        df = bc.describe(s)
        self.assertAlmostEqual(df.loc[0, "0-0-0"], 62.9328)
        self.assertTupleEqual(df.shape, (len(s), len(bc.subscripts)))

        s *= [2, 2, 2]
        structures = [s] * 10
        for s in structures:
            n = np.random.randint(4)
            inds = np.random.randint(16, size=n)
            s.remove_sites(inds)

        df_all = bc.describe_all(structures)
        i = random.randint(0, 9)
        df_s = df_all.xs(i, level="input_index")
        self.assertEqual(df_s.shape[0], len(structures[i]))
        self.assertTrue(df_s.equals(bc.describe(structures[i])))


class CoulomMatrixTest(unittest.TestCase):

    def setUp(self):

        self.s1 = Structure.from_spacegroup(225,
                                            Lattice.cubic(5.69169),
                                            ["Na", "Cl"],
                                            [[0, 0, 0], [0, 0, 0.5]])
        self.s2 = Structure.from_dict({'@class': 'Structure',
                                       '@module': 'pymatgen.core.structure',
                                       'charge': None,
                                       'lattice': {'a': 5.488739045730133,
                                                   'alpha': 60.0000000484055,
                                                   'b': 5.488739048031658,
                                                   'beta': 60.00000003453459,
                                                   'c': 5.48873905,
                                                   'gamma': 60.000000071689925,
                                                   'matrix': [[4.75338745, 0.0, 2.74436952],
                                                              [1.58446248, 4.48153667, 2.74436952],
                                                              [0.0, 0.0, 5.48873905]],
                                                   'volume': 116.92375473740876},
                                       'sites': [{'abc': [0.5, 0.5, 0.5],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [3.168924965, 2.240768335, 5.488739045]},
                                                 {'abc': [0.5, 0.5, 0.0],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [3.168924965, 2.240768335, 2.74436952]},
                                                 {'abc': [0.0, 0.5, 0.5],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [0.79223124, 2.240768335, 4.116554285]},
                                                 {'abc': [0.5, 0.0, 0.5],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [2.376693725, 0.0, 4.116554285]},
                                                 {'abc': [0.875, 0.875, 0.875],
                                                  'label': 'Lu',
                                                  'properties': {'coordination_no': 16, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Lu', 'occu': 1}],
                                                  'xyz': [5.54561868875, 3.9213445862499996, 9.60529332875]},
                                                 {'abc': [0.125, 0.125, 0.125],
                                                  'label': 'Lu',
                                                  'properties': {'coordination_no': 16, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Lu', 'occu': 1}],
                                                  'xyz': [0.79223124125, 0.56019208375, 1.37218476125]}]})

    def test_coulomb_mat(self):
        cm = CoulombMatrix()
        cmat = cm.describe(self.s1).as_matrix().reshape(self.s1.num_sites, self.s1.num_sites)
        na = Element('Na')
        cl = Element('Cl')
        dist = self.s1.distance_matrix
        self.assertEqual(cmat[0][0], (na.Z ** 2.4) * 0.5)
        self.assertEqual(cmat[4][4], (cl.Z ** 2.4) * 0.5)
        self.assertEqual(cmat[0][1], (na.Z * na.Z) / dist[0][1])

    def test_sorted_coulomb_mat(self):
        cm = CoulombMatrix(sorted=True)
        c = cm.coulomb_mat(self.s2)
        cmat = cm.describe(self.s2).as_matrix().reshape(self.s2.num_sites, self.s2.num_sites)
        norm_order_ind = np.argsort(np.linalg.norm(c, axis=1))
        for i in range(cmat.shape[1]):
            self.assertTrue(np.all(cmat[i] == c[norm_order_ind[i]]))

    def test_random_coulom_mat(self):
        cm = CoulombMatrix(randomized=True, random_seed=7)
        c = cm.coulomb_mat(self.s2)
        cmat = cm.describe(self.s2).as_matrix().reshape(self.s2.num_sites, self.s2.num_sites)
        cm2 = CoulombMatrix(randomized=True, random_seed=8)
        cmat2 = cm2.describe(self.s2).as_matrix().reshape(self.s2.num_sites, self.s2.num_sites)
        self.assertEqual(np.all(cmat == cmat2), False)
        for i in range(cmat.shape[1]):
            self.assertTrue(cmat[i] in c[i])

    def test_describe_all(self):
        cm = CoulombMatrix()
        c = cm.describe_all([self.s1, self.s2])
        c1 = cm.describe(self.s1)
        c2 = cm.describe(self.s2)
        self.assertTrue(np.all(c[0].dropna() == c1[0]))
        self.assertTrue(np.all(c[1].dropna() == c2[0]))


if __name__ == "__main__":
    unittest.main()
