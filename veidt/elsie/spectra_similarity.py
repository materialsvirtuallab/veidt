# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from monty.json import MSONable
import numpy as np
import warnings
from pymatgen.core.spectrum import Spectrum
from copy import deepcopy
from veidt.elsie.preprocessing import Preprocessing
from veidt.elsie import similarity_measures
from scipy.interpolate import interp1d
from scipy import signal


class SpectraSimilarity(MSONable):
    def __init__(self, sp1, sp2, interp_points=200):
        """
        Initialization SpectrumSimilarity object to determine the similarity
        between two spectra. Both spectra object should be follow pymatgen.

        Args:
            sp1: Spectrum object 1. Given spectrum to match, usually
                collected from exp.
            sp2: Spectrum object 2. Candidate spectrum, usually
                computational reference spectrum.
            interp_points: Number of points used for spectrum interpolation
                throughout comparison
        """
        self.sp1 = sp1
        self.sp2 = sp2
        self.shifted_sp1 = None
        self.shifted_sp2 = None
        self.interp_points = interp_points
        self._energy_validation()

    def _energy_validation(self):
        """
        Valid the overlap absorption range of both spectra. Warning will raise
        in the following two cases:
        1. If the overlap energy range is less than 30 meV,
        2. If there is no overlap energy, self.valid_comparison set to false
        """
        min_energy_1, max_energy_1 = np.min(self.sp1.x), np.max(self.sp1.x)
        min_energy_2, max_energy_2 = np.min(self.sp2.x), np.max(self.sp2.x)
        max_min_energy = max(min_energy_1, min_energy_2)
        min_max_energy = min(max_energy_1, max_energy_2)

        if (min_energy_2 > max_energy_1) or (min_energy_1 > max_energy_2):
            warning_msg = "Candidate spectrum has no overlap with given spectrum to match"
            warnings.warn(warning_msg)
            self.valid_comparison = False
        elif (min_max_energy - max_min_energy) < 30:
            warning_msg = "Candidate and given spectra's overlap absorption energy is less than 30 meV"
            warnings.warn(warning_msg)
            self.valid_comparison = True
        else:
            self.valid_comparison = True

    def _spectrum_shift(self, algo='threshold_shift', intensity_threshold=0.06, preset_shift=None):
        """
        Shift self.sp2 with respect to self.spec1. Self.spec1 will be
        untouched.

        Args:
            algo: Algorithm used to determine the energy shift between two
                spectra. Currently available types are:
                "threshold_shift": Use the onset of absorption. Onset energy
                are determined by the intensity_threshold.
                "cross_correlate": Use the cross correlation function between
                two spectra to determine the shift energy.
                "user_specify": User specify the shift energy between the two
                                spectra. The shift energy value should be set
                                through the preset_shift.
            intensity_threshold: The absorption peak intensity threshold used
                to determine the absorption onset, default set to 0.1
            preset_shift: The energy shift value between the two spectra.
                    preset_shift > 0 means sp2 needs to shift left w.r.t sp1
        """
        if algo == 'user_specify':
            if preset_shift is None:
                raise ValueError('The energy shift value has not been set')
            self.shifted_sp1, self.shifted_sp2, self.shifted_energy = \
                preset_value_shift(self.sp1, self.sp2, preset_shift)

        if algo == 'threshold_shift':
            self.sp1, self.sp2 = spectra_lower_extend(self.sp1, self.sp2)
            self.shifted_sp1, self.shifted_sp2, self.shifted_energy, self.abs_onset = \
                absorption_onset_shift(self.sp1, self.sp2, intensity_threshold)
        elif algo == 'cross_correlate':
            self.sp1, self.sp2 = spectra_lower_extend(self.sp1, self.sp2)
            self.shifted_sp1, self.shifted_sp2, self.shifted_energy = \
                signal_corre_shift(self.sp1, self.sp2)

    def get_shifted_similarity(self, similarity_metric, energy_variation=None,
                               spect_preprocess=None, **kwargs):
        """
        Calculate the similarity between two shifted spectra
        Args:
            similarity_metric (string): The similarity metric used for comparison.
            energy_variation (list): Energy variation value used to squeeze or broaden the candidate
                spectrum (sp2) beyonds spectrum shift onset point. E.g., [-2, 2, 0.1]
                specifies sp2's spectrum energy (Es) beyonds onset point will scale from Es - 2 to Es + 2
                at 0.1 interval. Maximum similarity and its' corresponding scale energy will be returned.
            spect_preprocess (list/tuple): Preprocessing steps need to taken for each spectrum

        """

        if not self.valid_comparison:
            return 0

        if (self.shifted_sp1 is None) and (self.shifted_sp2 is None):
            self._spectrum_shift(**kwargs)
        simi_class = getattr(similarity_measures, similarity_metric)

        if energy_variation is not None:
            sp2_energy_scale_onset = self.shifted_sp2.x[np.argmax(
                self.shifted_sp2.x > self.abs_onset)]
            sp2_energy_scale_end = max(self.shifted_sp2.x)
            sp2_scale_energy_den = (self.shifted_sp2.x > self.abs_onset).sum()

            max_simi = float("-inf")
            for scale_energy in np.arange(
                    energy_variation[0], energy_variation[1], energy_variation[2]):
                sp2_scaled_energy = np.linspace(
                    sp2_energy_scale_onset,
                    sp2_energy_scale_end + scale_energy,
                    sp2_scale_energy_den)
                shifted_sp2_scaled_energy = np.hstack(
                    (self.shifted_sp2.x[:np.argmax(
                        self.shifted_sp2.x > self.abs_onset)],
                     sp2_scaled_energy))
                if shifted_sp2_scaled_energy.shape != self.shifted_sp2.x.shape:
                    raise ValueError('The scaled energy grid density is '
                                     'different from pre-scaled')
                scaled_shifted_sp2 = Spectrum(shifted_sp2_scaled_energy,
                                              self.shifted_sp2.y)

                # Interpolate and calculate the similarity between
                # scaled_shifted_sp2 and shifted_sp1
                overlap_energy = energy_overlap(self.shifted_sp1,
                                                scaled_shifted_sp2)
                overlap_energy_grid = np.linspace(
                    overlap_energy[0], overlap_energy[1], self.interp_points)
                shifted_sp1_interp = spectra_energy_interpolate(
                    self.shifted_sp1, overlap_energy_grid)
                scaled_shifted_sp2_interp = spectra_energy_interpolate(
                    scaled_shifted_sp2, overlap_energy_grid)

                pre_shifted_sp1_interp = Preprocessing(shifted_sp1_interp)

                pre_shifted_sp1_interp.spectrum_process(['intnorm'])
                pre_scaled_shifted_sp2_interp = Preprocessing(scaled_shifted_sp2_interp)

                pre_scaled_shifted_sp2_interp.spectrum_process(['intnorm'])
                shifted_sp1_interp = pre_shifted_sp1_interp.spectrum
                scaled_shifted_sp2_interp = pre_scaled_shifted_sp2_interp.spectrum

                similarity_obj = simi_class(shifted_sp1_interp.y,
                                            scaled_shifted_sp2_interp.y)

                try:
                    similarity_value = similarity_obj.similarity_measure()
                except Exception:
                    warnings.warn("Cannot generate valid similarity value for the two spectra")
                    similarity_value = np.NaN

                if similarity_value > max_simi:
                    max_simi = similarity_value
                    self.interp_shifted_sp1 = shifted_sp1_interp
                    self.interp_scaled_shift_sp2 = scaled_shifted_sp2_interp
                    # max_scale_energy<0 means the sp2 should be squeeze for
                    # maximum matching
                    self.max_scale_energy = scale_energy

            if spect_preprocess is not None:
                pre_shifted_sp1_interp = Preprocessing(
                    self.interp_shifted_sp1)
                pre_scaled_shifted_sp2_interp = Preprocessing(
                    self.interp_scaled_shift_sp2)

                pre_shifted_sp1_interp.spectrum_process(spect_preprocess)
                pre_scaled_shifted_sp2_interp.spectrum_process(
                    spect_preprocess)

                shifted_sp1_interp = pre_shifted_sp1_interp.spectrum
                scaled_shifted_sp2_interp = pre_scaled_shifted_sp2_interp.spectrum
                similarity_obj = simi_class(shifted_sp1_interp.y,
                                            scaled_shifted_sp2_interp.y)
                max_simi = similarity_obj.similarity_measure()

            return max_simi

        elif energy_variation is None:
            overlap_energy = energy_overlap(self.shifted_sp1,
                                            self.shifted_sp2)
            overlap_energy_grid = np.linspace(
                overlap_energy[0], overlap_energy[1], self.interp_points)
            shifted_sp1_interp = spectra_energy_interpolate(
                self.shifted_sp1, overlap_energy_grid)
            shifted_sp2_interp = spectra_energy_interpolate(
                self.shifted_sp2, overlap_energy_grid)

            if spect_preprocess is not None:
                pre_shifted_sp1_interp = Preprocessing(shifted_sp1_interp)
                pre_shifted_sp2_interp = Preprocessing(shifted_sp2_interp)

                pre_shifted_sp1_interp.spectrum_process(spect_preprocess)
                pre_shifted_sp2_interp.spectrum_process(spect_preprocess)

                shifted_sp1_interp = pre_shifted_sp1_interp.spectrum
                shifted_sp2_interp = pre_shifted_sp2_interp.spectrum

            similarity_obj = simi_class(shifted_sp1_interp.y,
                                        shifted_sp2_interp.y)

            try:
                similarity_value = similarity_obj.similarity_measure()
            except Exception:
                warnings.warn("Cannot generate valid similarity value "
                              "for the two spectra")
                similarity_value = 0

            return similarity_value


def energy_overlap(sp1, sp2):
    """
    Calculate the overlap energy range of two spectra, i.e. lower bound is the
    maximum of two spectra's minimum energy.
    Upper bound is the minimum of two spectra's maximum energy

    Args:
        sp1: Spectrum object 1
        sp2: Spectrum object 2

    Returns:
        Overlap energy range

    """
    overlap_range = [max(sp1.x.min(), sp2.x.min()), min(sp1.x.max(),
                                                        sp2.x.max())]
    return overlap_range


def spectra_energy_interpolate(sp1, energy_range):
    """
    Use Scipy's interp1d and returns spectrum object with absorption value
    interpolated with given energy_range

    Args:
        sp1: Spectrum object 1
        energy_range: new energy range used in interpolate

    Returns:
        Spectrum object with given energy range and interpolated absorption value

    """
    interp = interp1d(sp1.x, sp1.y)
    interp_spect = interp(energy_range)
    sp1.x = np.array(energy_range)
    sp1.y = interp_spect
    return sp1


def spectra_lower_extend(sp1, sp2):
    """
    Extend the energy range of spectra and ensure both spectra has same lower
    bound in energy. The spectrum with higher low energy
    bound with be extended, the first absorption value will be used for
    absorption extension.

    Args:
        sp1: Spectrum object 1
        sp2: Spectrum object 2

    Returns:
        Two Spectrum objects with same lower energy bound

    """
    min_energy = min(sp1.x.min(), sp2.x.min())

    if sp1.x.min() > min_energy:
        # Calculate spectrum point density used for padding
        sp1_den = np.ptp(sp1.x) / sp1.ydim[0]
        extend_spec1_energy = np.linspace(min_energy, sp1.x.min(),
                                          retstep=sp1_den)[0][:-1]
        sp1_ext_energy = np.hstack((extend_spec1_energy, sp1.x))
        sp1_ext_mu = np.lib.pad(sp1.y, (len(extend_spec1_energy), 0), 'constant',
                                constant_values=(sp1.y[0], 0))
        sp1.x = sp1_ext_energy
        sp1.y = sp1_ext_mu

    elif sp2.x.min() > min_energy:
        sp2_den = np.ptp(sp2.x) / sp2.ydim[0]
        extend_spec2_energy = np.linspace(min_energy, sp2.x.min(),
                                          retstep=sp2_den)[0][:-1]
        sp2_ext_energy = np.hstack((extend_spec2_energy, sp2.x))
        sp2_ext_mu = np.lib.pad(sp2.y, (len(extend_spec2_energy), 0),
                                'constant',
                                constant_values=(sp2.y[0], 0))
        sp2.x = sp2_ext_energy
        sp2.y = sp2_ext_mu

    return sp1, sp2


def absorption_onset_shift(sp1, sp2, intensity_threshold):
    """
    Shift spectrum 2 with respect to spectrum 1 using the difference between
    two spectra's onset of absorption.
    The onset is determined by ascertaining the lowest incident energy at which
    the spectra's absorption intensity reaches the 'intensity_threshold' of the
    peak intensity.

    Args:
        sp1: Spectrum object 1
        sp2: Spectrum object 2
        intensity_threshold: The absorption peak intensity threshold used to
            determine the absorption onset. Must be a float between 0 and 1.

    Returns:
        shifted_sp1: Spectrum object 1
        shifted_sp2: Spectrum object with absorption same as sp2 and
            shifted energy range
        energy_diff: Energy difference between sp1 and sp2,
            energy_diff > 0 mean sp2 needs to shift left

    """
    if not 0 <= float(intensity_threshold) <= 1:
        raise ValueError("The intensity threshold must be between 0 and 1")

    sp1_inten_thres = max(sp1.y) * float(intensity_threshold)
    sp2_inten_thres = max(sp2.y) * float(intensity_threshold)

    threpoint_1_energy = sp1.x[np.argmax(sp1.y > sp1_inten_thres)]
    threpoint_2_energy = sp2.x[np.argmax(sp2.y > sp2_inten_thres)]

    energy_diff = threpoint_2_energy - threpoint_1_energy

    # sp2 need to shift left
    if energy_diff >= 0:
        sp2_new_energy = sp2.x - energy_diff
        sp2_new_mu = sp2.y

    # sp2 need to shift right
    elif energy_diff < 0:
        sp2_new_energy = sp2.x - energy_diff
        sp2_new_mu = sp2.y

    shifted_sp1 = Spectrum(sp1.x, sp1.y)
    shifted_sp2 = Spectrum(sp2_new_energy, sp2_new_mu)

    return shifted_sp1, shifted_sp2, energy_diff, threpoint_1_energy


def signal_corre_shift(sp1, sp2):
    """
    Using the cross correlation function between two spectra to determine the shift energy.
    Args:
        sp1: Spectrum object 1
        sp2: Spectrum object 2

    Returns:
        energy_diff: Energy difference between sp1 and sp2,
            energy_diff > 0 means sp2 needs to shift left

    """

    overlap_energy = energy_overlap(sp1, sp2)
    # Energy grid interpolate point density: 0.01 eV
    overlap_energy_grid = np.linspace(overlap_energy[0], overlap_energy[1],
                                      int(float(overlap_energy[1] - overlap_energy[0]) / 0.01))

    interp_sp1 = spectra_energy_interpolate(Spectrum(sp1.x, sp1.y), overlap_energy_grid)
    interp_sp2 = spectra_energy_interpolate(Spectrum(sp2.x, sp2.y), overlap_energy_grid)

    if not np.allclose(interp_sp1.x, interp_sp2.x, 1e-5):
        raise ValueError("Two scaled spectra's energy grid densities are different")

    sp2_shift_index = np.argmax(signal.correlate(interp_sp2.y, interp_sp1.y))

    # sp2 need to shift left
    if sp2_shift_index > interp_sp2.x.shape[0]:
        left_shift_index = sp2_shift_index - interp_sp2.x.shape[0]
        energy_diff = interp_sp2.x[left_shift_index] - interp_sp2.x.min()

    # sp2 need to shift right
    elif sp2_shift_index < interp_sp2.x.shape[0]:
        right_shift_index = interp_sp2.x.shape[0] - sp2_shift_index
        energy_diff = -(interp_sp2.x[right_shift_index] - interp_sp2.x.min())

    else:
        energy_diff = 0

    shifted_sp1 = Spectrum(sp1.x, sp1.y)
    shifted_sp2 = Spectrum(sp2.x - energy_diff, sp2.y)

    return shifted_sp1, shifted_sp2, energy_diff


def preset_value_shift(sp1, sp2, preset_shift):
    """
    Using the preset value to shift the two spectra
    Args:
        sp1: Spectrum object 1
        sp2: Spectrum object 2
        preset_shift: Preset energy shift value between two spectra,
            energy_diff > 0 means sp2 needs to shift left

    """

    shifted_sp1 = Spectrum(sp1.x, sp1.y)
    shifted_sp2 = Spectrum(sp2.x - preset_shift, sp2.y)
    return shifted_sp1, shifted_sp2, preset_shift
