# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
from numbers import Number
import numpy as np
from scipy.interpolate import interp1d
import warnings


class CenvPrediction(object):

    def __init__(self, xanes_spectrum, energy_reference, energy_range, edge_energy=None):

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
        elif self.energy_reference=='lowest':
            self.edge_energy = self.xanes_spectrum.e0

        self._energy_interp()

    def _energy_interp(self):
        #if energy_reference is 'lowest' and energy range is proper passed
        if self.energy_reference=='lowest':
            x_axis_energy = self.xanes_spectrum.x
            y_spectrum = self.xanes_spectrum.y
            x_axis_energy_start = np.min(x_axis_energy)
            x_axis_energy_end = x_axis_energy_start + self.energy_range
            x_axis_energy_end_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_end)
            x_axis_linspace = np.linspace(x_axis_energy_start, x_axis_energy[x_axis_energy_end_index], num=200)
            f = interp1d(x_axis_energy, y_spectrum, kind='cubic', bounds_error=False, fill_value=0)
            self.interp_spectrum = f(x_axis_linspace)
            self.interp_energy = x_axis_linspace
        elif self.energy_reference == 'E0':
            x_axis_energy = self.xanes_spectrum.x
            y_spectrum = self.xanes_spectrum.y
            x_axis_energy_start = self.edge_energy + self.energy_lower_bound
            x_energy_start_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_start)
            x_axis_energy_end = self.edge_energy + self.energy_higher_bound
            x_energy_end_index = find_nearest_energy_index(x_axis_energy, x_axis_energy_end)
            x_axis_linspace = np.linspace(x_axis_energy[x_energy_start_index], x_axis_energy[x_energy_end_index], num=200)
            f = interp1d(x_axis_energy, y_spectrum, kind='cubic', bounds_error=False, fill_value=0)
            self.interp_spectrum = f(x_axis_linspace)
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

            if self.energy_lower_bound >0:
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
