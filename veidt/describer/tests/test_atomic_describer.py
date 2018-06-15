# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
import random

import numpy as np
from monty.os.path import which
from pymatgen import Lattice, Structure, Element

from veidt.describer.atomic_describer import BispectrumCoefficients



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





if __name__ == "__main__":
    unittest.main()