from monty.json import MSONable
import numpy as np
import warnings
from pymatgen.core.spectrum import Spectrum
from copy import deepcopy
from veidt.ELSIE.preprocessing import Preprocessing
from veidt.ELSIE import similarity_measures
from scipy.interpolate import interp1d


class SpectraSimilarity(MSONable):
    def __init__(self, spect1, spect2, interp_points=200):
        """
        Initialization SpectrumSimilarity object to determine the similarity between two spectra.
        Both spectra object should be follow pymatgen
        Args:
            spect1: Spectrum object 1. Given spectrum to match, usually collected from exp.
            spect2: Spectrum object 2. Candidate spectrum, usually computational reference spectrum.
            interp_points: Number of points used for spectrum interpolation throughout comparison
        """

        self.spect1 = spect1
        self.spect2 = spect2
        self.shifted_spect1 = None
        self.shifted_spect2 = None
        self.interp_points = interp_points
        self._energy_validation()

    def _energy_validation(self):
        """
        Valid the overlap absorption range of both spectra. Warning will raise in the following two cases:
        1. If the overlap energy range is less than 30 meV,
        2. If there is no overlap energy, self.valid_comparison set to false
        """
        min_energy_1, max_energy_1 = np.min(self.spect1.x), np.max(self.spect1.x)
        min_energy_2, max_energy_2 = np.min(self.spect2.x), np.max(self.spect2.x)
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

    def _spectrum_shift(self, algo='threshold_shift', intensity_threshold=0.06):
        """
        Shift self.spect2 with respect to self.spec1. Self.spec1 will be untouched.
        Args:
            algo: Algorithm used to determine the energy shift between two spectra. Currently available types are:
                "threshold_shift": Use the onset of absorption. Onset energy are determined by the intensity_threshold.
                "cross_correlate": Use the cross correlation function between two spectra to determine the shift energy.
                "cross_FFT": Use FFT function to calculate the cross correlation function between two spectra and determine
                            energy shift
            intensity_threshold: The absorption peak intensity threshold used to determine the absorption onset, default
                                set to 0.1
            **kwargs: Other parameters
        """
        self.spect1, self.spect2 = spectra_lower_extend(self.spect1, self.spect2)

        if algo == 'threshold_shift':
            self.shifted_spect1, self.shifted_spect2, self.shifted_energy, self.abs_onset = absorption_onset_shift(
                self.spect1, self.spect2, intensity_threshold)

    def get_shifted_similarity(self, similarity_metric, energy_variation=None, spect_preprocess=None):

        if (self.shifted_spect1 is None) and (self.shifted_spect2 is None):
            self._spectrum_shift()
        simi_class = getattr(similarity_measures, similarity_metric)

        if energy_variation is not None:
            spect2_energy_scale_onset = self.shifted_spect2.x[np.argmax(self.shifted_spect2.x > self.abs_onset)]
            spect2_energy_scale_end = max(self.shifted_spect2.x)
            spect2_scale_energy_den = (self.shifted_spect2.x > self.abs_onset).sum()

            max_simi = float("-inf")
            for scale_energy in np.arange(energy_variation[0], energy_variation[1], energy_variation[2]):
                spect2_scaled_energy = np.linspace(spect2_energy_scale_onset, spect2_energy_scale_end + scale_energy,
                                                   spect2_scale_energy_den)
                shifted_spect2_scaled_energy = np.hstack(
                    (self.shifted_spect2.x[:np.argmax(self.shifted_spect2.x > self.abs_onset)], spect2_scaled_energy))
                if shifted_spect2_scaled_energy.shape != self.shifted_spect2.x.shape:
                    raise ValueError('The scaled energy grid density is different from pre-scaled')
                scaled_shifted_spect2 = Spectrum(shifted_spect2_scaled_energy, self.shifted_spect2.y)

                # Interpolate and calculate the similarity between scaled_shifted_spect2 and shifted_spect1
                overlap_energy = energy_overlap(self.shifted_spect1, scaled_shifted_spect2)
                overlap_energy_grid = np.linspace(overlap_energy[0], overlap_energy[1], self.interp_points)
                shifted_spect1_interp = spectra_energy_interpolate(self.shifted_spect1, overlap_energy_grid)
                scaled_shifted_spect2_interp = spectra_energy_interpolate(scaled_shifted_spect2, overlap_energy_grid)

                similarity_obj = simi_class(shifted_spect1_interp.y, scaled_shifted_spect2_interp.y)

                try:
                    similarity_value = similarity_obj.similarity_measure()
                except:
                    warnings.warn("Cannot generate valid similarity value for the two spectra")
                    similarity_value = np.NaN

                if similarity_value > max_simi:
                    max_simi = similarity_value
                    self.interp_shifted_spect1 = shifted_spect1_interp
                    self.interp_scaled_shift_spect2 = scaled_shifted_spect2_interp
                    self.max_scale_energy = scale_energy  # max_scale_energy<0 means the spect2 should be squeeze for maximum matching

            if spect_preprocess is not None:
                pre_shifted_spect1_interp = Preprocessing(self.interp_shifted_spect1)
                pre_scaled_shifted_spect2_interp = Preprocessing(self.interp_scaled_shift_spect2)

                pre_shifted_spect1_interp.spectrum_process(spect_preprocess)
                pre_scaled_shifted_spect2_interp.spectrum_process(spect_preprocess)

                shifted_spect1_interp = pre_shifted_spect1_interp.spectrum
                scaled_shifted_spect2_interp = pre_scaled_shifted_spect2_interp.spectrum
                similarity_obj = simi_class(shifted_spect1_interp.y, scaled_shifted_spect2_interp.y)
                max_simi = similarity_obj.similarity_measure()

            return max_simi

        elif energy_variation is None:
            overlap_energy = energy_overlap(self.shifted_spect1, self.shifted_spect2)
            overlap_energy_grid = np.linspace(overlap_energy[0], overlap_energy[1], self.interp_points)
            shifted_spect1_interp = spectra_energy_interpolate(self.shifted_spect1, overlap_energy_grid)
            shifted_spect2_interp = spectra_energy_interpolate(self.shifted_spect2, overlap_energy_grid)

            if spect_preprocess is not None:
                pre_shifted_spect1_interp = Preprocessing(shifted_spect1_interp)
                pre_shifted_spect2_interp = Preprocessing(shifted_spect2_interp)

                pre_shifted_spect1_interp.spectrum_process(spect_preprocess)
                pre_shifted_spect2_interp.spectrum_process(spect_preprocess)

                shifted_spect1_interp = pre_shifted_spect1_interp.spectrum
                shifted_spect2_interp = pre_shifted_spect2_interp.spectrum

            similarity_obj = simi_class(shifted_spect1_interp.y, shifted_spect2_interp.y)

            try:
                similarity_value = similarity_obj.similarity_measure()
            except:
                warnings.warn("Cannot generate valid similarity value for the two spectra")
                similarity_value = np.NaN

            return similarity_value


def energy_overlap(spect1, spect2):
    """
    Calculate the overlap energy range of two spectra, i.e. lower bound is the maximum of two spectra's minimum energy.
    Upper bound is the minimum of two spectra's maximum energy
    Args:
        spect1: Spectrum object 1
        spect2: Spectrum object 2

    Returns:
        Overlap energy range

    """
    overlap_range = [max(spect1.x.min(), spect2.x.min()), min(spect1.x.max(), spect2.x.max())]
    return overlap_range


def spectra_energy_interpolate(spect1, energy_range):
    """
    Use Scipy's interp1d and returns spectrum object with absorption value interpolated with given energy_range
    Args:
        spect1: Spectrum object 1
        energy_range: new energy range used in interpolate

    Returns:
        Spectrum object with given energy range and interpolated absorption value

    """
    interp = interp1d(spect1.x, spect1.y)
    interp_spect = interp(energy_range)
    spect1.x = np.array(energy_range)
    spect1.y = interp_spect
    return spect1


def spectra_lower_extend(spect1, spect2):
    """
    Extend the energy range of spectra and ensure both spectra has same lower bound in energy. The spectrum with higher low energy
    bound with be extended, the first absorption value will be used for absorption extension.
    Args:
        spect1: Spectrum object 1
        spect2: Spectrum object 2

    Returns:
        Two Spectrum objects with same lower energy bound

    """
    min_energy = min(spect1.x.min(), spect2.x.min())

    if spect1.x.min() > min_energy:
        # Calculate spectrum point density used for padding
        spect1_den = np.ptp(spect1.x) / spect1.ydim[0]
        extend_spec1_energy = np.linspace(min_energy, spect1.x.min(), retstep=spect1_den)[0][:-1]
        spect1_ext_energy = np.hstack((extend_spec1_energy, spect1.x))
        spect1_ext_mu = np.lib.pad(spect1.y, (len(extend_spec1_energy), 0), 'constant',
                                   constant_values=(spect1.y[0], 0))
        spect1.x = spect1_ext_energy
        spect1.y = spect1_ext_mu

    elif spect2.x.min() > min_energy:
        spect2_den = np.ptp(spect2.x) / spect2.ydim[0]
        extend_spec2_energy = np.linspace(min_energy, spect2.x.min(), retstep=spect2_den)[0][:-1]
        spect2_ext_energy = np.hstack((extend_spec2_energy, spect2.x))
        spect2_ext_mu = np.lib.pad(spect2.y, (len(extend_spec2_energy), 0), 'constant',
                                   constant_values=(spect2.y[0], 0))
        spect2.x = spect2_ext_energy
        spect2.y = spect2_ext_mu

    return spect1, spect2


def absorption_onset_shift(spect1, spect2, intensity_threshold):
    """
    Shift spectrum 2 with respect to spectrum 1 using the difference between two spectra's onset of absorption.
    The onset is determined by ascertaining the lowest incident energy at which the spectra's absorption intensity
    reaches the 'intensity_threshold' of the peak intensity.
    Args:
        spect1: Spectrum object 1
        spect2: Spectrum object 2
        intensity_threshold: The absorption peak intensity threshold used to determine the absorption onset. Must
                            be a float number between 0 and 1.

    Returns:
        shifted_spect1: Spectrum object 1
        shifted_spect2: Spectrum object with absorption same as spect2 and shifted energy range
        energy_diff: Energy difference between spect1 and spect2, energy_diff > 0 mean spect2 needs to shift left

    """
    if (not 0 <= float(intensity_threshold) <= 1):
        raise ValueError("The intensity threshold must be a value between 0 and 1")

    spect_1_inten_thres = max(spect1.y) * float(intensity_threshold)
    spect_2_inten_thres = max(spect2.y) * float(intensity_threshold)

    threpoint_1_energy = spect1.x[np.argmax(spect1.y > spect_1_inten_thres)]
    threpoint_2_energy = spect2.x[np.argmax(spect2.y > spect_2_inten_thres)]

    energy_diff = threpoint_2_energy - threpoint_1_energy

    ##spect2 need to shift left
    if energy_diff >= 0:
        spect_2_new_energy = spect2.x - energy_diff
        spect_2_new_mu = spect2.y

    ##Spect2 need to shift right
    elif energy_diff < 0:
        spect_2_new_energy = spect2.x - energy_diff
        spect_2_new_mu = spect2.y

    shifted_spect1 = deepcopy(spect1)
    shifted_spect2 = Spectrum(spect_2_new_energy, spect_2_new_mu)

    return shifted_spect1, shifted_spect2, energy_diff, threpoint_1_energy
