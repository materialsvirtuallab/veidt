# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
from numbers import Number
import numpy as np
from scipy.interpolate import interp1d
import warnings
import os
import json
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data")
CNUM_MODEL_NAME_TEMPLATE = 'RandomForest_{}_c_num.sav'
CMOTIF_MODEL_NAME_TEMPLATE = 'RandomForest_{}_c_env_ex_{}.sav'


class CenvPrediction(object):

    def __init__(self, xanes_spectrum, energy_reference, energy_range, edge_energy=None,
                 spectrum_interpolation=True):
        """
        Args:
            xanes_spectrum: veidt.rfxas.core.XANES object
            energy_reference (str): Energy reference mode, choose from 'lowest' or 'E0'. 'lowest' mode for
                            using the lowest energy of spectrum as the starting point of the to be characterized
                            spectrum. 'E0' mode for using the edge energy as the reference point to generate to be
                            characterized spectrum energy range.
            energy_range:
            edge_energy:
            spectrum_interpolation:
        """

        self.xanes_spectrum = xanes_spectrum
        self.absorption_specie = self.xanes_spectrum.absorption_specie
        self.energy_reference = energy_reference
        self.energy_range = energy_range
        if isinstance(self.energy_range, list):
            self.energy_lower_bound = self.energy_range[0]
            self.energy_higher_bound = self.energy_range[-1]

        self._parameter_validation()

        if energy_reference == 'E0' and edge_energy is None:
            warning_msg = "Using edge energy of xanes_spectrum object, be cautious about how the object's edge energy is determined"
            warnings.warn(warning_msg)
            self.edge_energy = self.xanes_spectrum.e0
        elif energy_reference == 'E0' and edge_energy:
            self.edge_energy = edge_energy
        elif self.energy_reference == 'lowest':
            self.edge_energy = self.xanes_spectrum.e0

        if spectrum_interpolation:
            self._energy_interp()
        else:
            self.interp_spectrum = self.xanes_spectrum.y
            self.interp_spectrum_reshape = np.array(self.interp_spectrum).reshape(1, -1)
            self.interp_energy = self.xanes_spectrum.x

    def cenv_prediction(self):
        self._cnum_prediction()
        self._cmotif_prediction()

    def _cnum_prediction(self):
        cnum_pred_ele_json = os.path.join(DATA_DIR, 'cnum_predict_elements.json')
        with open(cnum_pred_ele_json, 'r') as fp:
            cnum_pred_elements = json.load(fp)

        if self.absorption_specie not in cnum_pred_elements:
            warning_msg = 'Coordination number prediction model for {} is unavailable.'.format(self.absorption_specie)
            warnings.warn(warning_msg)
            self.pred_cnum_ranklist = 'cnum undetermined'
        else:
            cnum_model_name = CNUM_MODEL_NAME_TEMPLATE.format(self.absorption_specie)
            cnum_model_path = os.path.join(MODEL_DIR, 'cnum', cnum_model_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                cnum_model_loaded = joblib.load(cnum_model_path)
            self.pred_cnum_ranklist = cnum_model_loaded.predict(self.interp_spectrum_reshape)[0]

    def _cmotif_prediction(self):
        cmotif_pred_ele_json = os.path.join(DATA_DIR, 'cmotif_predict_elements.json')
        with open(cmotif_pred_ele_json, 'r') as fp:
            cmotif_ele_env_dict = json.load(fp)
        ele_env_valid_prediction = cmotif_ele_env_dict[self.absorption_specie]

        if self.pred_cnum_ranklist == 'cnum undetermined':
            self.pred_cenv = 'cenv undetermined'
        else:
            pred_cnum_ranklist = self.pred_cnum_ranklist
            # Using predicted cnum ranklist to predict cenv ranklist
            spectral_env_pred = []
            for indi_pred_cnum in pred_cnum_ranklist.split('-'):
                cmotif_pred_cenv = 'ex_{}'.format(indi_pred_cnum)

                # No available coord. motif prediction model for this particular coord. num. of this element
                if cmotif_pred_cenv not in ele_env_valid_prediction:
                    pseudo_cmotif_label = '{} coord. motif undetermined'.format(indi_pred_cnum)
                    spectral_env_pred.append(pseudo_cmotif_label)
                else:
                    cmotif_model_name = CMOTIF_MODEL_NAME_TEMPLATE.format(self.absorption_specie, indi_pred_cnum)
                    cmotif_model_path = os.path.join(MODEL_DIR, 'cmotif', cmotif_model_name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        cmotif_model_loaded = joblib.load(cmotif_model_path)
                        pred_motif_ranklist = cmotif_model_loaded.predict(self.interp_spectrum_reshape)[0]
                        pred_cnum_cmotif_concat = '-'.join([indi_pred_cnum, pred_motif_ranklist])
                        spectral_env_pred.append(pred_cnum_cmotif_concat)

            self.pred_cenv = spectral_env_pred

    def _energy_interp(self):
        # if energy_reference is 'lowest' and energy range is proper passed
        if self.energy_reference == 'lowest':
            x_axis_energy = self.xanes_spectrum.x
            y_spectrum = self.xanes_spectrum.y
            x_axis_energy_start = np.min(x_axis_energy)
            x_axis_energy_end = x_axis_energy_start + self.energy_range
            x_axis_energy_end_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_end)
            x_axis_linspace = np.linspace(x_axis_energy_start, x_axis_energy[x_axis_energy_end_index], num=200)
            f = interp1d(x_axis_energy, y_spectrum, kind='cubic', bounds_error=False, fill_value=0)
            self.interp_spectrum = f(x_axis_linspace)
            normalize_factor = np.max(self.interp_spectrum, axis=0)
            self.interp_spectrum /= normalize_factor
            self.interp_energy = x_axis_linspace
            self.interp_spectrum_reshape = np.array(self.interp_spectrum).reshape(1, -1)

        elif self.energy_reference == 'E0':
            x_axis_energy = self.xanes_spectrum.x
            y_spectrum = self.xanes_spectrum.y
            x_axis_energy_start = self.edge_energy + self.energy_lower_bound
            x_energy_start_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_start)
            x_axis_energy_end = self.edge_energy + self.energy_higher_bound
            x_energy_end_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_end)
            x_axis_linspace = np.linspace(x_axis_energy[x_energy_start_index], x_axis_energy[x_energy_end_index],
                                          num=200)
            f = interp1d(x_axis_energy, y_spectrum, kind='cubic', bounds_error=False, fill_value=0)
            self.interp_spectrum = f(x_axis_linspace)
            normalize_factor = np.max(self.interp_spectrum, axis=0)
            self.interp_spectrum /= normalize_factor
            self.interp_spectrum_reshape = np.array(self.interp_spectrum).reshape(1, -1)
            self.interp_energy = x_axis_linspace

    def _parameter_validation(self):
        if self.energy_reference not in ['lowest', 'E0']:
            raise ValueError('Invalid energy reference option, energy_reference should either be "lowest" or "E0"')

        if self.energy_reference == 'lowest':
            if not isinstance(self.energy_range, Number):
                raise ValueError(
                    'Energy range needs to be a number when the energy reference point is the starting energy of the spectrum')

            if self.energy_range < 0:
                raise ValueError(
                    'Energy range needs to be larger than 0. Invalid energy range error.'
                )

        if self.energy_reference == 'E0':
            if not isinstance(self.energy_range, list):
                raise ValueError(
                    'Energy range needs to be a list contains lower energy bound and higher energy bound refer to energy reference point'
                )

            if self.energy_lower_bound > 0:
                raise ValueError(
                    'Energy lower bound needs to be less than zero.'
                )

            if self.energy_higher_bound < 0:
                raise ValueError(
                    'Energy higher bound needs to be larger than zero.'
                )


def find_nearest_energy_index(energy_array, energy_value):
    energy_array = np.asarray(energy_array)
    energy_index = (np.abs(energy_array - energy_value)).argmin()
    return energy_index
