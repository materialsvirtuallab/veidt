from veidt.ELSIE.preprocessing import Preprocessing
from pymatgen.core.spectrum import Spectrum
import numpy as np
import unittest
from copy import deepcopy


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.x_range = np.linspace(-np.pi, np.pi, 201)
        self.spectrum = Spectrum(self.x_range, np.sin(self.x_range))

    def test_derivative(self):
        deri_preprocessing = Preprocessing(deepcopy(self.spectrum))
        deri_preprocessing.first_derivative()
        self.assertTrue(np.allclose(deri_preprocessing.spectrum.y, np.cos(self.x_range)[:-1], atol=2e-2))

        deri_2nd_pre = Preprocessing(deepcopy(self.spectrum))
        deri_2nd_pre.second_derivative()
        self.assertTrue(np.allclose(deri_2nd_pre.spectrum.y, np.negative(np.sin(self.x_range))[:-2], atol=3.5e-2))

    def test_weighted_derivative(self):
        deri_preprocessing = Preprocessing(Spectrum(self.x_range, 2 * self.x_range))
        deri_preprocessing.weighted_first_derivative()
        self.assertTrue(np.allclose(deri_preprocessing.spectrum.y, 4 * self.x_range[:-1]))

        deri_preprocessing = Preprocessing(Spectrum(self.x_range, 2 * self.x_range))
        deri_preprocessing.weighted_second_derivative()
        self.assertTrue(np.allclose(deri_preprocessing.spectrum.y, 0))

    def test_normalization(self):
        normal_pre = Preprocessing(Spectrum(self.x_range, np.abs(np.sin(self.x_range))))
        normal_pre.intensity_normalize()
        self.assertAlmostEqual(np.sum(normal_pre.spectrum.y), 1)

        normal_max = Preprocessing(Spectrum(self.x_range, np.abs(np.sin(self.x_range))))
        normal_max.maximum_intensity_norm()
        self.assertEqual(max(normal_max.spectrum.y), 1)

        vector_norm = Preprocessing(Spectrum(self.x_range, np.abs(np.sin(self.x_range))))
        vector_norm.vector_norm_normalize()
        self.assertTrue(np.allclose(vector_norm.spectrum.y.sum() * np.linalg.norm(np.abs(np.sin(self.x_range))),
                                    np.abs(np.sin(self.x_range)).sum()))

        area_norm = Preprocessing(Spectrum(self.x_range, np.abs(np.sin(self.x_range))))
        area_norm.area_normalize()
        self.assertTrue(np.allclose(np.trapz(area_norm.spectrum.y, area_norm.spectrum.x), 1))

    def test_spec_preprocessing(self):
        x_range = np.linspace(0, np.pi / 2, 100)
        to_compare = np.cos(x_range)[:-1]
        to_compare /= to_compare.sum()
        spec_preprocessing = Preprocessing(Spectrum(x_range, np.sin(x_range)))
        spec_preprocessing.spectrum_process(('1st_der', 'intnorm'))

        self.assertAlmostEqual(spec_preprocessing.spectrum.y.sum(), 1)
        self.assertTrue(np.allclose(spec_preprocessing.spectrum.y, to_compare, atol=1.5e-4))

    def test_property(self):
        spec_preprocessing = Preprocessing(self.spectrum)
        self.assertTrue(set(spec_preprocessing.preprocessing_method) == set(
            ['first_derivative', 'second_derivative', 'vector_norm_normalize', 'maximum_intensity_norm',
             'area_normalize',
             'snv_norm', 'square_root_squashing', 'sigmoid_squashing', 'weighted_first_derivative',
             'weighted_second_derivative', 'intensity_normalize']))
