import unittest
import pandas as pd
import numpy as np
from pymatgen import Structure, Element
from copy import deepcopy
from veidt.elsie.spectra_similarity import *
from pymatgen.analysis.xas.spectrum import XANES

test_df = pd.read_json('test_df.json')
Al2O3_stru = Structure.from_file('alpha_Al2O3.cif')
LiCoO2_stru = Structure.from_file('LiCoO2_mp24850.cif')


class SpectraSimilarityTest(unittest.TestCase):
    def setUp(self):
        Al2O3_spectrum = np.array(test_df.loc[test_df['Formula'] == 'Al2O3']['Exp.spectrum'].values[0])
        LiCoO2_spectrum = np.array(test_df.loc[test_df['Formula'] == 'LiCoO2']['Exp.spectrum'].values[0])
        self.Al2O3_xane_1 = XANES(Al2O3_spectrum[:, 0], Al2O3_spectrum[:, 1], Al2O3_stru, Element('Al'), 'K')

        self.Al2O3_xane_2 = deepcopy(self.Al2O3_xane_1)
        self.Al2O3_xane_2.x = self.Al2O3_xane_2.x - 40

        self.Al2O3_xane_left_shift5 = deepcopy(self.Al2O3_xane_1)
        self.Al2O3_xane_left_shift5.x = self.Al2O3_xane_left_shift5.x - 5

        self.Al2O3_xane_right_shift5 = deepcopy(self.Al2O3_xane_1)
        self.Al2O3_xane_right_shift5.x = self.Al2O3_xane_right_shift5.x + 5

        self.LiCoO2_xane = XANES(LiCoO2_spectrum[:, 0], LiCoO2_spectrum[:, 1], LiCoO2_stru, Element('Co'), 'K')

    def test_energy_range_warning(self):
        with self.assertWarnsRegex(UserWarning, 'less than 30 meV'):
            Al2O3_SpecSimi = SpectraSimilarity(self.Al2O3_xane_1, self.Al2O3_xane_2)
            self.assertTrue(Al2O3_SpecSimi.valid_comparison)

        with self.assertWarnsRegex(UserWarning, r'no overlap .* match'):
            Co_Al_SpecSimi = SpectraSimilarity(self.Al2O3_xane_1, self.LiCoO2_xane)
            self.assertFalse(Co_Al_SpecSimi.valid_comparison)

    def test_spectra_lower_extend(self):
        temp_spect1, temp_spect2 = spectra_lower_extend(deepcopy(self.Al2O3_xane_1),
                                                        deepcopy(self.Al2O3_xane_left_shift5))

        self.assertTrue(np.allclose(temp_spect1.y[np.where((temp_spect1.x - 1550) <= 5)],
                                    temp_spect1.y[np.where(temp_spect1.x == 1550)][0]))
        self.assertTrue(temp_spect1.x[0] == temp_spect2.x[0])
        self.assertTrue((temp_spect1.x[-1] - temp_spect2.x[-1]) == 5)

        temp_spect1, temp_spect2 = spectra_lower_extend(deepcopy(self.Al2O3_xane_1),
                                                        deepcopy(self.Al2O3_xane_right_shift5))
        self.assertTrue(np.allclose(temp_spect2.y[np.where((temp_spect2.x - 1555) <= 5)],
                                    temp_spect2.y[np.where(temp_spect2.x == 1555)][0]))
        self.assertTrue(temp_spect1.x[0] == temp_spect2.x[0])
        self.assertTrue((temp_spect2.x[-1] - temp_spect1.x[-1]) == 5)

    def test_absorption_onset_shift(self):
        shifted_spect1, shifted_spect2, shifted_energy, abs_onset = absorption_onset_shift(self.Al2O3_xane_1,
                                                                                           self.Al2O3_xane_left_shift5,
                                                                                           0.1)
        self.assertEqual(shifted_energy, -5.0)
        self.assertAlmostEqual(abs_onset, 1565.4000000000001)
        shifted_spect3, shifted_spect4, shifted_energy, abs_onset = absorption_onset_shift(self.Al2O3_xane_1,
                                                                                           self.Al2O3_xane_right_shift5,
                                                                                           0.1)
        self.assertEqual(shifted_energy, 5.0)
        self.assertAlmostEqual(abs_onset, 1565.4000000000001)
        self.assertTrue(np.allclose(shifted_spect1.x, shifted_spect3.x))

    def test_get_shifted_similarity(self):
        # Test whether the get_shifted_similarity function could found the right energy variation scale
        shifted_spect1, shifted_spect2, energy_diff, abs_onset = absorption_onset_shift(self.Al2O3_xane_1,
                                                                                        self.Al2O3_xane_1, 0.1)
        self.assertEqual(energy_diff, 0.0)

        spect_start_scale_onset = self.Al2O3_xane_1.x[np.argmax(self.Al2O3_xane_1.x > abs_onset)]
        spect_end_scale = max(self.Al2O3_xane_1.x)
        spect_scale_energy_den = (self.Al2O3_xane_1.x > abs_onset).sum()
        spect_energy_before_onset = self.Al2O3_xane_1.x[:np.argmax(self.Al2O3_xane_1.x > abs_onset)]

        spect_squeeze_scaled_energy = np.linspace(spect_start_scale_onset, spect_end_scale - 2, spect_scale_energy_den)
        spect_squeeze_scaled_energy = np.hstack((spect_energy_before_onset, spect_squeeze_scaled_energy))
        squeezed_spect = Spectrum(spect_squeeze_scaled_energy, self.Al2O3_xane_1.y)
        squeezed_simi = SpectraSimilarity(self.Al2O3_xane_1, squeezed_spect)
        squeezed_simi_max = squeezed_simi.get_shifted_similarity('PearsonCorrMeasure', energy_variation=[-2, 2, 0.01])
        self.assertAlmostEqual(squeezed_simi_max, 1.0, 4)
        # Due to the interpolation variation, the squeeze energy scale calculated will show some deviation
        self.assertTrue(np.allclose(squeezed_simi.max_scale_energy, 2, 1e-2))

        spect_broaden_scaled_energy = np.linspace(spect_start_scale_onset, spect_end_scale + 2, spect_scale_energy_den)
        spect_broaden_scaled_energy = np.hstack((spect_energy_before_onset, spect_broaden_scaled_energy))
        broaden_spect = Spectrum(spect_broaden_scaled_energy, self.Al2O3_xane_1.y)
        broaden_simi = SpectraSimilarity(self.Al2O3_xane_1, broaden_spect)
        broaden_simi_max = broaden_simi.get_shifted_similarity('PearsonCorrMeasure', energy_variation=[-2, 2, 0.01])
        self.assertAlmostEqual(broaden_simi_max, 1.0, 4)
        self.assertEqual(broaden_simi.max_scale_energy, -2)

        squeezed_simi = SpectraSimilarity(self.Al2O3_xane_1, squeezed_spect)
        squeezed_simi_max = squeezed_simi.get_shifted_similarity('Cosine', energy_variation=[-2, 2, 0.01],
                                                                 spect_preprocess=(('areanorm', 'sigmoid', 'intnorm')))
        self.assertAlmostEqual(squeezed_simi_max, 1.0, 4)
        # Due to the interpolation variation, the squeeze energy scale calculated will show some deviation
        self.assertTrue(np.allclose(squeezed_simi.max_scale_energy, 2, 1e-2))

        self_simi = SpectraSimilarity(self.Al2O3_xane_1, self.Al2O3_xane_1)
        self_simi.get_shifted_similarity('PearsonCorrMeasure', energy_variation=[-2, 2, 0.01],
                                         spect_preprocess=(('areanorm', 'sigmoid', 'intnorm')))
        self.assertAlmostEqual(self_simi.max_scale_energy, 0.0)

        broaden_simi = SpectraSimilarity(self.Al2O3_xane_1, broaden_spect)
        broaden_simi_max = broaden_simi.get_shifted_similarity('Euclidean', energy_variation=[-2, 2, 0.01],
                                                               spect_preprocess=(('areanorm', 'sigmoid', 'intnorm')))
        self.assertAlmostEqual(broaden_simi_max, 1.0, 4)
        self.assertEqual(broaden_simi.max_scale_energy, -2)
