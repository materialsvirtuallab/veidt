# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import numpy as np


class Preprocessing(object):
    """
    Preprocessing class used for spectrum preprocessing.
    """

    def __init__(self, spectrum):
        """
        Create an Preprocessing object
        Args:
            spectrum (pymatgen.core.spectrum.Spectrum): Spectrum object used to initialize
             preprocessing class.
        """
        self.spectrum = spectrum
        self.process_tag = []
        self.proc_dict = {
            '1st_der': 'first_derivative',
            '2nd_der': 'second_derivative',
            'vecnorm': 'vector_norm_normalize',
            'maxnorm': 'maximum_intensity_norm',
            'areanorm': 'area_normalize',
            'snvnorm': 'snv_norm',
            'square': 'square_root_squashing',
            'sigmoid': 'sigmoid_squashing',
            '1st_wt': 'weighted_first_derivative',
            '2nd_wt': 'weighted_second_derivative',
            'intnorm': 'intensity_normalize'
        }

    @property
    def preprocessing_method(self):
        """
        Returns: a list of available preprocessing methods
        """
        return list(self.proc_dict.values())

    def first_derivative(self):
        """
        Return first derivative as spectrum
        """
        deriv_x, deriv_y = self.derivative_spect(self.spectrum, 1)
        self.spectrum.x, self.spectrum.y = np.copy(deriv_x), np.copy(deriv_y)

    def second_derivative(self):
        """
        Return second derivative as spectrum
        """
        deriv_x, deriv_y = self.derivative_spect(self.spectrum, 2)
        self.spectrum.x, self.spectrum.y = np.copy(deriv_x), np.copy(deriv_y)

    def weighted_first_derivative(self):
        """
        Return weighted first derivative spectrum as spectrum
        """
        deriv_x, deriv_y = self.derivative_spect(self.spectrum, 1)
        self.spectrum.x, self.spectrum.y = deriv_x, np.multiply(self.spectrum.y[:-1], deriv_y)

    def weighted_second_derivative(self):
        """
        Return weighted second derivative spectrum as spectrum
        """
        deriv_x, deriv_y = self.derivative_spect(self.spectrum, 2)
        self.spectrum.x, self.spectrum.y = deriv_x, np.multiply(self.spectrum.y[:-2], deriv_y)

    def intensity_normalize(self):
        """
        Normalize with respect to the intensity sum
        """
        self.spectrum.normalize('sum')

    def maximum_intensity_norm(self):
        """
        Normalize with respect to the maximum intensity
        """
        self.spectrum.normalize('max')

    def vector_norm_normalize(self):
        """
        Normalize with respect to the norm of the spectrum as a vector
        """
        spect_norm = np.linalg.norm(self.spectrum.y)
        self.spectrum.y /= spect_norm

    def area_normalize(self):
        """
        Normalize the peak intensity using under curve area, i.e. normalized curve's under curve
        area should equals 1
        """
        under_curve_area = np.trapz(self.spectrum.y, self.spectrum.x)
        self.spectrum.y /= under_curve_area

    def snv_norm(self):
        """
        Normalize with repect to the variance of the spectrum intensity and return abs. spectrum
        """
        inten_mean = np.mean(self.spectrum.y)
        inten_std = np.mean(self.spectrum.y)
        normalized_mu = np.divide(np.subtract(self.spectrum.y, inten_mean), inten_std)

        # Since snv norm will return negative absorption value after normalization, need to add
        # the minimum absorption value and shift the baseline back to zero
        min_norm_mu = np.abs(np.min(normalized_mu))
        normalized_mu = np.add(normalized_mu, min_norm_mu)
        self.spectrum.y = normalized_mu

    def square_root_squashing(self):
        """
        Squashing the spectrum using square root of the spectrum
        """
        squashed_mu = np.sqrt(np.abs(self.spectrum.y))
        self.spectrum.y = squashed_mu

    def sigmoid_squashing(self):
        """
        Squashing the spectrum using the sigmoid funtion, i.e. squashed_y = (1 - cos(pi*spectrum.y))/2
        """
        squashed_mu = np.divide(np.subtract(1, np.cos(np.pi * self.spectrum.y)), 2)
        self.spectrum.y = squashed_mu

    def derivative_spect(self, spect1, order):
        """
        Calculate derivative of a given spectrum, to keep returned spectrum dimension consistent, endpoints are
        not pad with endvalues
        Args:
            spect1: Given spectrum with spect1.x corresponding to energy. spect1.y corresponding to absorption
            order: The number of times the spectrum are differenced
        Returns: Differenciated x and y

        """
        deriv_x = np.copy(spect1.x)
        deriv_y = np.copy(spect1.y)

        def first_derivative(x, y):
            derivative = np.diff(y) / np.diff(x)
            return x[:-1], derivative

        while order >= 1:
            deriv_x, deriv_y = first_derivative(deriv_x, deriv_y)
            order -= 1

        return deriv_x, deriv_y

    def spectrum_process(self, process_seq):
        """
        Preprocess the self.spectrum object using the preprocess method listed in process_seq
        Args:
            process_seq (list/tuple/string): preprocessing methods
        Returns:

        """
        if (process_seq is not None) and (isinstance(process_seq, list) or isinstance(process_seq, tuple)):
            for pro in process_seq:
                getattr(self, self.proc_dict[pro])()
                self.process_tag.append(pro)

        if (process_seq is not None) and isinstance(process_seq, str):
            getattr(self, self.proc_dict[process_seq])()
            self.process_tag.append(process_seq)
