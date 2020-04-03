import unittest, json, os
import numpy as np
from pymatgen import Structure, Element
from copy import deepcopy
from veidt.elsie.spectra_similarity import *
from pymatgen.analysis.xas.spectrum import XAS

Al2O3_cif = os.path.join(os.path.dirname(__file__), 'alpha_Al2O3.cif')
LiCoO2_cif = os.path.join(os.path.dirname(__file__), 'LiCoO2_mp24850.cif')
Al2O3_spect_file = os.path.join(os.path.dirname(__file__), 'Al2O3_spectrum.json')
LiCoO2_spect_file = os.path.join(os.path.dirname(__file__), 'LiCoO2_spectrum.json')

Al2O3_stru = Structure.from_file(Al2O3_cif)
LiCoO2_stru = Structure.from_file(LiCoO2_cif)
with open(LiCoO2_spect_file, 'r') as f:
    LiCoO2_spect = json.load(f)
with open(Al2O3_spect_file, 'r') as f:
    Al2O3_spect = json.load(f)


class SpectraSimilarityTest(unittest.TestCase):
    def setUp(self):
        self.Al2O3_xane_1 = XAS(Al2O3_spect['x'], Al2O3_spect['y'], Al2O3_stru, Element('Al'), 'K', spectrum_type='XANES')

        self.Al2O3_xane_2 = deepcopy(self.Al2O3_xane_1)
        self.Al2O3_xane_2.x = self.Al2O3_xane_2.x - 40

        self.Al2O3_xane_left_shift5 = deepcopy(self.Al2O3_xane_1)
        self.Al2O3_xane_left_shift5.x = self.Al2O3_xane_left_shift5.x - 5

        self.Al2O3_xane_right_shift5 = deepcopy(self.Al2O3_xane_1)
        self.Al2O3_xane_right_shift5.x = self.Al2O3_xane_right_shift5.x + 5

        self.LiCoO2_xane = XAS(LiCoO2_spect['x'], LiCoO2_spect['y'], LiCoO2_stru, Element('Co'), 'K', spectrum_type='XANES')

    def test_energy_range_warning(self):
        #with self.assertWarnsRegex(UserWarning, 'less than 30 meV'):
        Al2O3_SpecSimi = SpectraSimilarity(self.Al2O3_xane_1, self.Al2O3_xane_2)
        self.assertTrue(Al2O3_SpecSimi.valid_comparison)

        # with self.assertWarnsRegex(UserWarning, r'no overlap .* match'):
        Co_Al_SpecSimi = SpectraSimilarity(self.Al2O3_xane_1, self.LiCoO2_xane)
        self.assertFalse(Co_Al_SpecSimi.valid_comparison)
        self.assertTrue(Co_Al_SpecSimi.get_shifted_similarity('Cosine') == 0)
        self.assertTrue(
            Co_Al_SpecSimi.get_shifted_similarity('Cosine',
                spect_preprocess=['areanorm', 'sigmoid', 'intnorm']) == 0)

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

        self.assertRaisesRegex(ValueError, "intensity threshold", absorption_onset_shift, sp1=self.Al2O3_xane_1,
                               sp2=self.Al2O3_xane_2,
                               intensity_threshold=2)

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

    def test_cross_correlation_shift(self):
        shifted_spect1, shifted_spect2, shifted_energy = signal_corre_shift(self.Al2O3_xane_1,
                                                                            self.Al2O3_xane_left_shift5)
        self.assertTrue(np.allclose(shifted_energy, -5.0, 1e-2))

        shifted_spect1, shifted_spect2, shifted_energy = signal_corre_shift(self.Al2O3_xane_1,
                                                                            self.Al2O3_xane_right_shift5)
        self.assertTrue(np.allclose(shifted_energy, 5.0, 1e-2))

        spect_simi1 = SpectraSimilarity(self.Al2O3_xane_1, self.Al2O3_xane_left_shift5)
        cosine_simi = spect_simi1.get_shifted_similarity('Cosine', spect_preprocess=(('intnorm')),
                                                         algo='cross_correlate')
        pearson_simi = spect_simi1.get_shifted_similarity('PearsonCorrMeasure', spect_preprocess=(('intnorm')),
                                                          algo='cross_correlate')
        with self.assertRaises(AttributeError):
            spect_simi1.abs_onset
            self.assertAlmostEqual(cosine_simi, 1, 5)
            self.assertAlmostEqual(pearson_simi, 1, 5)
            self.assertTrue(np.allclose(spect_simi1.shifted_energy, -5, 1e-2))

        spect_simi2 = SpectraSimilarity(self.Al2O3_xane_1, self.Al2O3_xane_right_shift5)
        cosine_simi = spect_simi2.get_shifted_similarity('Cosine', spect_preprocess=(('intnorm')),
                                                         algo='cross_correlate')
        pearson_simi = spect_simi2.get_shifted_similarity('PearsonCorrMeasure', spect_preprocess=(('intnorm')),
                                                          algo='cross_correlate')
        with self.assertRaises(AttributeError):
            spect_simi2.abs_onset
            self.assertAlmostEqual(cosine_simi, 1, 5)
            self.assertAlmostEqual(pearson_simi, 1, 5)
            self.assertTrue(np.allclose(spect_simi2.shifted_energy, 5, 1e-2))

    def test_preset_shift(self):
        shifted_spect1, shifted_spect2, shifted_energy = preset_value_shift(self.Al2O3_xane_1,
                                                                            self.Al2O3_xane_left_shift5, -5)

        self.assertEqual(shifted_energy, -5.0)
        self.assertTrue(np.all(shifted_spect1.x == shifted_spect2.x))
        self.assertTrue(np.all(shifted_spect1.y == shifted_spect2.y))

        shifted_spect1, shifted_spect2, shifted_energy = preset_value_shift(self.Al2O3_xane_1,
                                                                            self.Al2O3_xane_right_shift5, 5)
        self.assertEqual(shifted_energy, 5.0)
        self.assertTrue(np.all(shifted_spect1.x == shifted_spect2.x))
        self.assertTrue(np.all(shifted_spect1.y == shifted_spect2.y))

        spect_simi_preset = SpectraSimilarity(self.Al2O3_xane_1, self.Al2O3_xane_right_shift5)
        self.assertRaisesRegex(ValueError, "The energy shift value has not been set", spect_simi_preset._spectrum_shift,
                               algo='user_specify')

        spect_simi_preset._spectrum_shift(algo='user_specify', preset_shift=5)
        self.assertTrue(np.all(spect_simi_preset.shifted_sp2.x == spect_simi_preset.sp1.x))
        cosine_simi = spect_simi_preset.get_shifted_similarity('Cosine', spect_preprocess=(('intnorm')),
                                                               algo='user_specify', preset_shift=5)
        spect_simi_2 = SpectraSimilarity(self.Al2O3_xane_1, self.Al2O3_xane_left_shift5)
        cosine_simi_2 = spect_simi_2.get_shifted_similarity('Cosine', spect_preprocess=(('intnorm')),
                                                            algo='user_specify', preset_shift=-5)

        self.assertAlmostEqual(cosine_simi, 1)
        self.assertAlmostEqual(cosine_simi_2, 1)

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

if __name__ == "__main__":
    unittest.main()

