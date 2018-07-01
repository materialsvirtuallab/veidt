# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import re
import os
import glob
import random
import logging
import subprocess
import itertools
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from monty.io import zopen
from monty.os.path import which
from monty.tempfile import ScratchDir
from monty.serialization import loadfn
from pymatgen import Structure, Lattice, Element, units
from veidt.potential.abstract import Potential
from veidt.potential.processing import pool_from, convert_docs
from veidt.potential.lammps.calcs import EnergyForceStress

module_dir = os.path.dirname(__file__)
NNinput_params = loadfn(os.path.join(module_dir, 'params', 'NNinput.json'))

class NNPotential(Potential):
    """
    This class implements Neural Network Potential.
    """
    bohr_to_angstrom = units.bohr_to_angstrom
    eV_to_Ha = units.eV_to_Ha

    def __init__(self, name=None):
        """

        Args:
            name (str): Name of force field.
        """
        self.name = name if name else "NNPotential"
        self.specie = None
        self.shortest_distance = np.inf
        self.lowest_energy = np.inf
        self.weights = []
        self.bs = []
        self.epochs = 0
        self.params = None
        self.scaling_params = None

    def _line_up(self, structure, energy, forces, virial_stress):
        """
        Convert input structure, energy, forces, virial_stress to
        proper configuration format for RuNNer usage. Note that
        RuNNer takes bohr as length unit and Hatree as energy unit.

        Args:
            structure (Structure): Pymatgen Structure object.
            energy (float): DFT-calculated energy of the system.
            forces (list): The forces should have dimension
                (num_atoms, 3).
            virial_stress (list): stress should has 6 distinct
                elements arranged in order [xx, yy, zz, xy, yz, xz].

        Returns:
        """
        if len(structure.symbol_set) > 1:
            raise ValueError("Structure is not unary.")

        inputs = OrderedDict(Size=structure.num_sites, \
                             SuperCell=structure.lattice, \
                             AtomData=(structure, forces), \
                             Energy=energy, \
                             Stress=virial_stress)

        lines = ['begin']

        if 'SuperCell' in inputs:
            bohr_matrix = inputs['SuperCell'].matrix / self.bohr_to_angstrom
            for vec in bohr_matrix:
                lines.append('lattice {:>15.6f}{:>15.6f}{:>15.6f}'.format(*vec))
        if 'AtomData' in inputs:
            format_float = \
                'atom{:>16.9f}{:>16.9f}{:>16.9f}{:>4s}{:>15.9f}{:>15.9f}{:>15.9f}{:>15.9f}{:>15.9f}'
            for i, (site, force) in enumerate(zip(structure, forces)):
                lines.append(format_float.format(*site.coords / self.bohr_to_angstrom, \
                                site.species_string, 0.0, 0.0,
                                *np.array(force) * self.eV_to_Ha * self.bohr_to_angstrom))
        if 'Energy' in inputs:
            lines.append('energy  {:f}'.format(energy * self.eV_to_Ha))

        lines.append('charge  {:f}'.format(structure.charge))
        lines.append('end')

        return '\n'.join(lines)

    def write_cfgs(self, filename, cfg_pool):

        lines = []
        for dataset in cfg_pool:
            if isinstance(dataset['structure'], dict):
                structure = Structure.from_dict(dataset['structure'])
            else:
                structure = dataset['structure']
            energy = dataset['outputs']['energy']
            forces = dataset['outputs']['forces']
            virial_stress = dataset['outputs']['virial_stress']

            lines.append(self._line_up(structure, energy, forces, virial_stress))

            dist = np.unique(structure.distance_matrix.ravel())[1]
            if self.shortest_distance > dist:
                self.shortest_distance = dist

        self.specie = Element(structure.symbol_set[0])

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        return filename

    def write_input(self, atom_energy=None, mode=1, **kwargs):
        """
        Write input_nn file to train the Neural Network model.

        Args:
            atom_energy (float): Atomic reference energy to remove before
                training (unit: eV).
            mode (int): Mode to execute the RuNNer. (1=calculate symmetry functions,
                2=fitting mode, 3=predicition mode)
            kwargs:
                describer:
                    r_cut (float): Cutoff distance (unit: Å).
                    r_etas (1D array): η in radial function.
                    rss (1D array): Rs in radial function.
                    a_etas (1D array): η in angular function.
                    zetas (1D array): ζ in angular function.
                    lambdas (1D array): λ in angular function. Default to
                        (1, -1).
                model:
                    hidden_layers (list): List of ints contains the number of
                        nodes in each hidden layer.
                    activations (list): List of strings contains the activation
                        function in each hidden layer and the output layer.
                        (t=tanh, s=sigmoid, g=gaussian, c=cosine, l=linear)
                    epochs (int): Training epochs. Default to 100.
                    weights_min (float): Minimum value for initial random short
                        range weights. Default to -1.0.
                    weights_max (float): Maximum value for initial random short
                        range weights. Default to 1.0.
                    test_fraction (float): threshold for splitting between
                        training and test set.
                    scale_min_short_atomic (float): Minimum value of scaled
                        symmetry functions features. Default to 0.0.
                    scale_max_short_atomic (float): Maximum value of scaled
                        symmetry functions features. Default to 1.0.
                    fitting_unit (string): unit for error output (eV or Ha).
                        Default to eV.
                    force_update_scaling (float): Scaling factor for force
                        update. Default to 1.0.
                    optmode_short_energy (int): Optimization modefor short
                        range energies (1=Kalman filter, 2=conjugate gradient,
                        3=steepest descent). Default to 1.
                    optmode_short_force (int): Optimization modefor short
                        range forces (1=Kalman filter, 2=conjugate gradient,
                        3=steepest descent). Default to 1.
                    short_energy_error_threshold (float): Threshold of
                        adaptive Kalman filter for short range energy.
                        Default to 0.0.
                    short_force_error_threshold (float): Threshold of
                        adaptive Kalman filter for short range force.
                        Default to 0.3.
                    kalman_lambda_short (float): Kalman parameters.
                    kalman_nue_short (float): Kalman parameters.
                    short_energy_fraction (float): Weights of energy
                        used for training. Default to 1.
                    short_force_fraction (float): Weights of force
                        used for training. Default to 0.1.
                    short_force_group (int): Group forces for update.
        """
        atom_energy = atom_energy if atom_energy else self.lowest_energy

        filename = 'input.nn'

        PARAMS = {'general': ['nn_type_short 1', 'number_of_elements 1'],
                  'describer': ['rcut', 'r_etas', 'rss', 'a_etas', 'zetas', 'lambdas'],
                  'model': {'structure': ['hidden_layers', 'activations'],
                            'scale': ['scale_min_short_atomic',
                                      'scale_max_short_atomic'],
                            'fitting': ['fitting_unit', 'force_update_scaling',
                                        'optmode_short_energy', 'optmode_short_force',
                                        'short_energy_error_threshold',
                                        'short_force_error_threshold',
                                        'kalman_lambda_short', 'kalman_nue_short',
                                        'short_energy_fraction', 'short_force_group',
                                        'short_force_fraction', 'epochs',
                                        'weights_min', 'weights_max',
                                        'test_fraction', 'points_in_memory'],
                            'analyze': ['analyze_error', 'print_mad',
                                        'analyze_composition', 'repeated_energy_update'],
                            'output': ['write_weights_epoch 1', 'write_trainpoints',
                                       'write_trainforces', 'calculate_forces',
                                       'calculate_stress']}}

        if 'general' in PARAMS:
            lines = PARAMS.get('general')
            lines.append(' '.join(['runner_mode', str(mode)]))
            lines.append(' '.join(['elements', self.specie.name]))
            atom_energy = atom_energy * self.eV_to_Ha
            lines.extend(['remove_atom_energies', \
                          ' '.join(['atom_energy', self.specie.name, str(atom_energy)])])
            # bond_threshold = self.shortest_distance / self.bohr_to_angstrom
            lines.append(' '.join(['bond_threshold', '0.5']))

        if 'describer' in PARAMS:
            type2_format = 'symfunction_short {central_atom}  2 {neighbor_atom}' \
                           '    {r_eta}    {rs}    {rcut}'
            type3_format = 'symfunction_short {central_atom}  3 {neighbor_atom1} ' \
                           '{neighbor_atom2}    {a_eta}   {lambd}   {zeta}   {rcut}'
            central_atom = self.specie.name
            neighbor_atom1 = self.specie.name
            neighbor_atom2 = self.specie.name
            r_cut = kwargs.get('r_cut') if kwargs.get('r_cut') \
                                    else NNinput_params.get('describer').get('r_cut')
            r_cut /= self.bohr_to_angstrom
            r_etas = kwargs.get('r_etas') if kwargs.get('r_etas') \
                                    else NNinput_params.get('describer').get('r_etas')
            rss = kwargs.get('rss') if kwargs.get('rss') \
                                    else NNinput_params.get('describer').get('rss')
            rss = np.array(rss) / self.bohr_to_angstrom

            a_etas = kwargs.get('a_etas') if kwargs.get('a_etas') \
                                    else NNinput_params.get('describer').get('a_etas')
            zetas = kwargs.get('zetas') if kwargs.get('zetas')\
                                    else NNinput_params.get('describer').get('zetas')
            lambdas = NNinput_params.get('describer').get('lambdas')

            for r_eta, rs in itertools.product(r_etas, rss):
                lines.append(type2_format.format(central_atom=central_atom,
                                                 neighbor_atom=neighbor_atom1,
                                                 r_eta=r_eta, rs=rs, rcut=r_cut))

            for a_eta, lambd, zeta in itertools.product(a_etas, lambdas, zetas):
                lines.append(type3_format.format(central_atom=central_atom,
                                                 neighbor_atom1=neighbor_atom1,
                                                 neighbor_atom2=neighbor_atom2,
                                                 a_eta=a_eta, lambd=lambd,
                                                 zeta=zeta, rcut=r_cut))

            self.num_features = len(r_etas) * len(rss) \
                                    + len(a_etas) * len(zetas) * len(lambdas)

        if 'model' in PARAMS:
            lines.append('use_short_nn')
            lines.append('use_short_forces')
            hidden_layers = kwargs.get('hidden_layers') if kwargs.get('hidden_layers') \
                    else NNinput_params.get('model').get('structure').get('hidden_layers')
            self.layer_sizes = [self.num_features] + hidden_layers + [1]

            hidden_layers = [str(i) for i in hidden_layers]
            activations = kwargs.get('activations') if kwargs.get('activations') \
                    else NNinput_params.get('model').get('structure').get('activations')
            lines.append('global_hidden_layers_short {}'.format(len(hidden_layers)))
            lines.append('global_nodes_short {}'.format(' '.join(hidden_layers)))
            lines.append('global_activation_short {}'.format(' '.join(activations)))

            if 'scale' in PARAMS.get('model'):
                lines.extend(['mix_all_points', 'scale_symmetry_functions'])
                for key in PARAMS.get('model').get('scale'):
                    value = kwargs.get(key) if kwargs.get(key) \
                            else NNinput_params.get('model').get('scale').get(key)
                    lines.append(' '.join([key, str(value)]))
            if 'fitting' in PARAMS.get('model'):
                for key in PARAMS.get('model').get('fitting'):
                    value = kwargs.get(key) if kwargs.get(key) \
                            else NNinput_params.get('model').get('fitting').get(key)
                    if key == 'epochs':
                        self.epochs = value
                    lines.append(' '.join([key, str(value)]))
            if 'analyze' in PARAMS.get('model'):
                lines.extend(PARAMS.get('model').get('analyze'))
            if 'output' in PARAMS.get('model'):
                lines.extend(PARAMS.get('model').get('output'))

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        return filename

    def read_cfgs(self, filename='output.data'):
        """
        Args:
            filename (str): The configuration file to be read.
        """
        data_pool = []
        with zopen(filename, 'rt') as f:
            lines = f.read()

        block_pattern = re.compile('begin\n(.*?)end', re.S)
        lattice_pattern = re.compile('lattice(.*?)\n')
        position_pattern = re.compile('atom(.*?)\n')
        energy_pattern = re.compile('energy(.*?)\n')

        for block in block_pattern.findall(lines):
            d = {'outputs':{}}
            lattice_str = lattice_pattern.findall(block)
            lattice = Lattice(np.array([latt.split() for latt in lattice_str],
                                        dtype=np.float) * self.bohr_to_angstrom)
            position_str = position_pattern.findall(block)
            positions = pd.DataFrame([pos.split() for pos in position_str])
            positions.columns = \
                ['x', 'y', 'z', 'specie', 'charge', 'atomic_energy', 'fx', 'fy', 'fz']
            coords = np.array(positions.loc[:, ['x', 'y', 'z']], dtype=np.float)
            coords = coords * self.bohr_to_angstrom
            species = np.array(positions['specie'])
            forces = np.array(positions.loc[:, ['fx', 'fy', 'fz']], dtype=np.float)
            forces = forces / self.eV_to_Ha / self.bohr_to_angstrom
            energy_str = energy_pattern.findall(block)[0]
            energy = float(energy_str.lstrip()) / self.eV_to_Ha
            struct = Structure(lattice=lattice, species=species, coords=coords,
                               coords_are_cartesian=True)
            d['structure'] = struct.as_dict()
            d['outputs']['energy'] = energy
            d['outputs']['forces'] = forces
            d['num_atoms'] = len(struct)

            data_pool.append(d)
        _, df = convert_docs(docs=data_pool)
        return data_pool, df

    def write_param(self):
        """
        Write optimized weights file to perform energy and force prediction.
        """
        if self.params is None or self.scaling_params is None:
            raise RuntimeError("The parameters should be provided.")
        weights_filename = '.'.join(['weights', self.suffix, 'data'])
        format_str_weight = '{:>18s}{:>2s}{:>10s}{:>6s}{:>6s}{:>6s}{:>6s}'
        format_str_bs = '{:>18s}{:>2s}{:>10s}{:>6s}{:>6}'
        lines = []
        for i in range(self.params.shape[0]):
            if self.params.iloc[i]['type'] == 'a':
                lines.append(format_str_weight.format(*self.params.iloc[i]))
            else:
                lines.append(format_str_bs.format(*self.params.iloc[i]))

        with open(weights_filename, 'w') as f:
            f.writelines('\n'.join(lines))

        scaling_filename = 'scaling.data'
        format_str_scaling = '{:>4s}{:>5s}{:>19s}{:>18s}{:>18s}'
        format_str_trivial = '{:>20s}{:>20s}'
        scaling_lines = []
        for i in range(self.num_features):
            scaling_lines.append(format_str_scaling.format(*self.scaling_params.iloc[i]))
        scaling_lines.append(format_str_trivial.format(*self.scaling_params.iloc[-1]))
        with open(scaling_filename, 'w') as f:
            f.writelines('\n'.join(scaling_lines))

        return weights_filename, scaling_filename

    def train(self, train_structures, energies=None, forces=None,
                                    stresses=None, **kwargs):
        """
        Training data with moment tensor method.

        Args:
            train_structures ([Structure]): The list of Pymatgen Structure object.
                energies ([float]): The list of total energies of each structure
                in structures list.
            energies ([float]): List of total energies of each structure in
                structures list.
            forces ([np.array]): List of (m, 3) forces array of each structure
                with m atoms in structures list. m can be varied with each
                single structure case.
            stresses (list): List of (6, ) virial stresses of each
                structure in structures list.
            kwargs: Parameters in write_input method.
        """
        if not which('RuNNer'):
            raise RuntimeError("RuNNer has not been found.")

        for energy, structure in zip(energies, train_structures):
            if self.lowest_energy > (energy / len(structure)):
                self.lowest_energy = energy / len(structure)

        train_pool = pool_from(train_structures, energies, forces, stresses)
        atoms_filename = 'input.data'

        with ScratchDir('.'):
            atoms_filename = self.write_cfgs(filename=atoms_filename, cfg_pool=train_pool)

            for i in range(1, 3):
                input_filename = self.write_input(mode=i, **kwargs)
                p = subprocess.Popen(['RuNNer'], stdout=subprocess.PIPE)
                stdout = p.communicate()[0]

                rc = p.returncode
                if rc != 0:
                    error_msg = 'RuNNer exited with return code %d' % rc
                    msg = stdout.decode("utf-8").split('\n')[:-1]
                    try:
                        error_line = [i for i, m in enumerate(msg)
                                      if m.startswith('ERROR')][0]
                        error_msg += ', '.join([e for e in msg[error_line:]])
                    except:
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)

            weights_filename_pattern = '*{}*'.format(str(self.epochs) + '.short')
            weights_filename = glob.glob(weights_filename_pattern)[0]
            self.suffix = weights_filename.split('.')[2]
            with open(weights_filename) as f:
                lines = f.readlines()

            params = pd.DataFrame([line.split() for line in lines])
            params.columns = ['value', 'type', '', 'ahead_index', 'ahead_node',
                              'behind_index', 'behind_node']
            self.params = params

            for layer_index in range(1, len(self.layer_sizes)):
                weights_group = params[(params['ahead_index'] == str(layer_index - 1)) \
                              & (params['behind_index'] == str(layer_index))]

                weights = np.reshape(np.array(weights_group['value'], dtype=np.float),
                                     (self.layer_sizes[layer_index - 1],
                                      self.layer_sizes[layer_index]))
                self.weights.append(weights)

                bs_group = params[(params['type'] == 'b') &
                                  (params['ahead_index'] == str(layer_index))]
                bs = np.array(bs_group['value'], dtype=np.float)
                self.bs.append(bs)

            with open('scaling.data') as f:
                scaling_lines = f.readlines()
            scaling_params = pd.DataFrame([line.split() for line in scaling_lines])
            scaling_params.column = ['', '', 'min', 'max', 'average']
            self.scaling_params = scaling_params

        return rc

    def evaluate(self, test_structures, ref_energies, ref_forces,
                                ref_stresses, **kwargs):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potential.

        Args:
            test_structures ([Structure]): List of Pymatgen Structure Objects.
            ref_energies ([float]): List of DFT-calculated total energies of
                each structure in structures list.
            ref_forces ([np.array]): List of DFT-calculated (m, 3) forces of
                each structure with m atoms in structures list. m can be varied
                with each single structure case.
            ref_stresses (list): List of DFT-calculated (6, ) viriral stresses
                of each structure in structures list.
            kwargs: Parameters of write_input method.
        """
        if not which('RuNNer'):
            raise RuntimeError("RuNNer has not been found.")

        original_file = 'input.data'
        predict_file = 'output.data'

        predict_pool = pool_from(test_structures, ref_energies,
                                 ref_forces, ref_stresses)
        with ScratchDir('.'):
            _, _ = self.write_param()
            original_file = self.write_cfgs(original_file, cfg_pool=predict_pool)
            _, df_orig = self.read_cfgs(original_file)

            _ = self.write_input(mode=3, **kwargs)

            dfs = []
            for data in predict_pool:
                _ = self.write_cfgs(original_file, cfg_pool=[data])
                p = subprocess.Popen(['RuNNer'], stdout=subprocess.PIPE)
                stdout = p.communicate()[0]

                rc = p.returncode
                if rc != 0:
                    error_msg = 'RuNNer exited with return code %d' % rc
                    msg = stdout.decode("utf-8").split('\n')[:-1]
                    try:
                        error_line = [i for i, m in enumerate(msg)
                                      if m.startswith('ERROR')][0]
                        error_msg += ', '.join([e for e in msg[error_line:]])
                    except:
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)

                _, df = self.read_cfgs(predict_file)
                dfs.append(df)
            df_predict = pd.concat(dfs, ignore_index=True)

        return df_orig, df_predict

    def predict(self, structure):
        pass