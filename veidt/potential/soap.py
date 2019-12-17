# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import re
import os
import yaml
import random
import logging
import subprocess
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict

import numpy as np
from monty.io import zopen
from monty.os.path import which
from monty.tempfile import ScratchDir
from monty.serialization import loadfn
from pymatgen import Structure, Lattice, Element
from veidt.potential.abstract import Potential
from veidt.potential.processing import pool_from, convert_docs
from veidt.potential.lammps.calcs import EnergyForceStress

module_dir = os.path.dirname(__file__)
soap_params = loadfn(os.path.join(module_dir, 'params', 'soap.json'))


class SOAPotential(Potential):
    """
    This class implements Smooth Overlap of Atomic Position potential.
    """
    pair_style = 'pair_style        quip'
    pair_coeff = 'pair_coeff        * * {} {} {}'

    def __init__(self, name=None, param=None):
        """

        Args:
            name (str): Name of force field.
            param (dict): The parameter configuration of potential.
        """
        self.name = name if name else "SOAPotential"
        self.param = param if param else {}
        self.specie = None

    def _line_up(self, structure, energy, forces, virial_stress):
        """
        Convert input structure, energy, forces, virial_stress to
        proper configuration format for MLIP usage.

        Args:
            structure (Structure): Pymatgen Structure object.
            energy (float): DFT-calculated energy of the system.
            forces (list): The forces should have dimension
                (num_atoms, 3).
            virial_stress (list): stress should has 6 distinct
                elements arranged in order [xx, yy, zz, xy, yz, xz].

        Returns:
        """
        full_virial_stress = [virial_stress[0], virial_stress[3], virial_stress[5],
                              virial_stress[3], virial_stress[1], virial_stress[4],
                              virial_stress[5], virial_stress[4], virial_stress[2]]

        inputs = OrderedDict(Size=structure.num_sites,
                             SuperCell=structure.lattice,
                             AtomData=(structure, forces),
                             Energy=energy,
                             Stress=full_virial_stress)

        lines = []
        if 'Size' in inputs:
            lines.append(str(inputs['Size']))

        description = []
        if 'Energy' in inputs:
            description.append('dft_energy={}'.format(inputs['Energy']))
        if 'Stress' in inputs:
            description.append('dft_virial={%s}' %
                               '\t'.join(list(map(lambda f: str(f), inputs['Stress']))))
        if 'SuperCell' in inputs:
            SuperCell_str = list(map(lambda f: str(f), inputs['SuperCell'].matrix.ravel()))
            description.append('Lattice="{}"'.format('     '.join(SuperCell_str)))
        description.append("Properties=species:S:1:pos:R:3:Z:I:1:dft_force:R:3")
        lines.append(' '.join(description))

        if 'AtomData' in inputs:
            format_str = '{:<10s}{:>16f}{:>16f}{:>16f}{:>8d}{:>16f}{:>16f}{:>16f}'
            for i, (site, force) in enumerate(zip(structure, forces)):
                lines.append(format_str.format(site.species_string,
                                               *site.coords, site.specie.Z, *force))
        return '\n'.join(lines)

    def write_cfgs(self, filename, cfg_pool):
        if not filename.endswith('.xyz'):
            raise RuntimeError('The extended xyz file should end with ".xyz"')

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

        self.specie = Element(structure.symbol_set[0])

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        return filename

    def read_cfgs(self, filename, predict=False):
        """
        Args:
            filename (str): The configuration file to be read.
        """
        type_convert = {'R': np.float32, 'I': np.int, 'S': np.str}
        data_pool = []
        with zopen(filename, 'rt') as f:
            lines = f.read()
        repl = re.compile('AT ')
        lines = repl.sub('', string=lines)

        block_pattern = re.compile('(\n[0-9]+\n|^[0-9]+\n)(.+?)(?=\n[0-9]+\n|$)', re.S)
        lattice_pattern = re.compile('Lattice="(.+)"')
        # energy_pattern = re.compile('dft_energy=(-?[0-9]+.[0-9]+)', re.I)
        energy_pattern = re.compile(r'(?<=\S{3}\s|dft_)energy=(-?[0-9]+.[0-9]+)')
        # stress_pattern = re.compile('dft_virial={(.+)}')
        stress_pattern = re.compile(r'dft_virial=({|)(.+?)(}|) \S.*')
        properties_pattern = re.compile(r'properties=(\S+)', re.I)
        # position_pattern = re.compile('\n(.+)', re.S)
        position_pattern = re.compile('\n(.+?)(?=\nE.*|\n\n.*|$)', re.S)
        # formatify = lambda string: [float(s) for s in string.split()]

        for (size, block) in block_pattern.findall(lines):
            d = {'outputs': {}}
            size = int(size)
            lattice_str = lattice_pattern.findall(block)[0]
            lattice = Lattice(list(map(lambda s: float(s), lattice_str.split())))
            # energy_str = energy_pattern.findall(block)[0]
            energy_str = energy_pattern.findall(block)[-1]
            energy = float(energy_str)
            # stress_str = stress_pattern.findall(block)[0]
            stress_str = stress_pattern.findall(block)[0][1]
            virial_stress = np.array(list(map(lambda s: float(s), stress_str.split())))
            virial_stress = [virial_stress[i] for i in [0, 4, 8, 1, 5, 6]]
            properties = properties_pattern.findall(block)[0].split(":")
            labels_columns = OrderedDict()
            labels = defaultdict()
            for i in range(0, len(properties), 3):
                labels_columns[properties[i]] = [int(properties[i + 2]), properties[i + 1]]
            position_str = position_pattern.findall(block)[0].split('\n')
            position = np.array([p.split() for p in position_str])
            column_index = 0
            for key in labels_columns:
                num_columns, dtype = labels_columns[key]
                labels[key] = position[:, column_index: column_index + num_columns].astype(type_convert[dtype])
                column_index += num_columns
            struct = Structure(lattice=lattice, species=labels['species'].ravel(),
                               coords=labels['pos'], coords_are_cartesian=True)
            if predict:
                forces = labels['force']
            else:
                forces = labels['dft_force']
            d['structure'] = struct.as_dict()
            d['outputs']['energy'] = energy
            assert size == struct.num_sites
            d['num_atoms'] = size
            d['outputs']['forces'] = forces
            d['outputs']['virial_stress'] = virial_stress

            data_pool.append(d)
        _, df = convert_docs(docs=data_pool)
        return data_pool, df

    def train(self, train_structures, energies=None, forces=None, stresses=None,
              default_sigma=[0.0005, 0.1, 0.05, 0.01],
              use_energies=True, use_forces=True, use_stress=False, **kwargs):
        """
        Training data with gaussian process regression.

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
            default_sigma (list): Error criteria in energies, forces, stress
                and hessian. Should have 4 numbers.
            use_energies (bool): Whether to use dft total energies for training.
                Default to True.
            use_forces (bool): Whether to use dft atomic forces for training.
                Default to True.
            use_stress (bool): Whether to use dft virial stress for training.
                Default to False.

            kwargs:
                l_max (int): Parameter to configure GAP. The band limit of
                    spherical harmonics basis function. Default to 12.
                n_max (int): Parameter to configure GAP. The number of radial basis
                    function. Default to 10.
                atom_sigma (float): Parameter to configure GAP. The width of gaussian
                    atomic density. Default to 0.5.
                zeta (float): Present when covariance function type is do product.
                    Default to 4.
                cutoff (float): Parameter to configure GAP. The cutoff radius.
                    Default to 4.0.
                cutoff_transition_width (float): Parameter to configure GAP.
                    The transition width of cutoff radial. Default to 0.5.
                delta (float): Parameter to configure Sparsification.
                    The signal variance of noise. Default to 1.
                f0 (float): Parameter to configure Sparsification.
                    The signal mean of noise. Default to 0.0.
                n_sparse (int): Parameter to configure Sparsification.
                    Number of sparse points.
                covariance_type (str): Parameter to configure Sparsification.
                    The type of convariance function. Default to dot_product.
                sparse_method (str): Method to perform clustering in sparsification.
                    Default to 'cur_points'.

                sparse_jitter (float): Intrisic error of atomic/bond energy,
                    used to regularise the sparse covariance matrix.
                    Default to 1e-8.
                e0 (float): Atomic energy value to be subtracted from energies
                    before fitting. Default to 0.0.
                e0_offset (float): Offset of baseline. If zero, the offset is
                    the average atomic energy of the input data or the e0
                    specified manually. Default to 0.0.
        """
        if not which('teach_sparse'):
            raise RuntimeError("teach_sparse has not been found.\n",
                               "Please refer to https://github.com/libAtoms/QUIP for ",
                               "further detail.")
        atoms_filename = 'train.xyz'
        xml_filename = 'train.xml'
        train_pool = pool_from(train_structures, energies, forces, stresses)

        exe_command = ["teach_sparse"]
        exe_command.append('at_file={}'.format(atoms_filename))
        gap_configure_params = ['l_max', 'n_max', 'atom_sigma', 'zeta', 'cutoff',
                                'cutoff_transition_width', 'delta', 'f0', 'n_sparse',
                                'covariance_type', 'sparse_method']
        preprocess_params = ['sparse_jitter', 'e0', 'e0_offset']
        target_for_training = ['use_energies', 'use_forces', 'use_stress']
        if len(default_sigma) != 4:
            raise ValueError("The default sigma is supposed to have 4 numbers.")

        gap_command = ['soap']
        for param_name in gap_configure_params:
            param = kwargs.get(param_name) if kwargs.get(param_name) \
                else soap_params.get(param_name)
            gap_command.append(param_name + '=' + '{}'.format(param))
        exe_command.append("gap=" + "{" + "{}".format(' '.join(gap_command)) + "}")

        for param_name in preprocess_params:
            param = kwargs.get(param_name) if kwargs.get(param_name) \
                else soap_params.get(param_name)
            exe_command.append(param_name + '=' + '{}'.format(param))

        default_sigma = [str(f) for f in default_sigma]
        exe_command.append("default_sigma={%s}" % (' '.join(default_sigma)))

        if use_energies:
            exe_command.append('energy_parameter_name=dft_energy')
        if use_forces:
            exe_command.append('force_parameter_name=dft_force')
        if use_stress:
            exe_command.append('virial_parameter_name=dft_virial')
        exe_command.append('gp_file={}'.format(xml_filename))

        with ScratchDir('.'):
            self.write_cfgs(filename=atoms_filename, cfg_pool=train_pool)

            p = subprocess.Popen(exe_command, stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'QUIP exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            def get_xml(xml_file):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                potential_label = root.tag
                gpcoordinates = list(root.iter('gpCoordinates'))[0]
                param_file = gpcoordinates.get('sparseX_filename')
                param = np.loadtxt(param_file)
                return tree, param, potential_label

            tree, param, potential_label = get_xml(xml_filename)
            self.param['xml'] = tree
            self.param['param'] = param
            self.param['potential_label'] = potential_label

        return rc

    def write_param(self, xml_filename='soap.xml'):
        """
        Write xml file to perform lammps calculation.

        Args:
            xml_filename (str): Filename to store xml formatted parameters.
        """
        if not self.param:
            raise RuntimeError("The xml and parameters should be provided.")
        tree = self.param.get('xml')
        root = tree.getroot()
        gpcoordinates = list(root.iter('gpCoordinates'))[0]
        param_filename = "{}.soapparam".format(self.name)
        gpcoordinates.set('sparseX_filename', param_filename)
        np.savetxt(param_filename, self.param.get('param'))
        tree.write(xml_filename)
        pair_coeff = self.pair_coeff.format(xml_filename,
                                            '\"Potential xml_label={}\"'.format(self.param.get('potential_label')),
                                            self.specie.Z)
        ff_settings = [self.pair_style, pair_coeff]
        return ff_settings

    def evaluate(self, test_structures, ref_energies=None, ref_forces=None,
                 ref_stresses=None, predict_energies=True,
                 predict_forces=True, predict_stress=False):
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
            predict_energies (bool): Whether to predict energies of configurations.
            predict_forces (bool): Whether to predict forces of configurations.
            predict_stress (bool): Whether to predict virial stress of
                configurations.
        """
        if not which('quip'):
            raise RuntimeError("quip has not been found.\n",
                               "Please refer to https://github.com/libAtoms/QUIP for ",
                               "further detail.")
        xml_file = 'predict.xml'
        original_file = 'original.xyz'
        predict_file = 'predict.xyz'
        predict_pool = pool_from(test_structures, ref_energies,
                                 ref_forces, ref_stresses)

        with ScratchDir('.'):
            _ = self.write_param(xml_file)
            original_file = self.write_cfgs(original_file, cfg_pool=predict_pool)
            _, df_orig = self.read_cfgs(original_file)

            exe_command = ["quip"]
            exe_command.append("atoms_filename={}".format(original_file))
            exe_command.append("param_filename={}".format(xml_file))
            if predict_energies:
                exe_command.append("energy=T")
            if predict_forces:
                exe_command.append("forces=T")
            if predict_stress:
                exe_command.append("virial=T")

            p = subprocess.Popen(exe_command, stdout=open(predict_file, 'w'))
            stdout = p.communicate()[0]
            rc = p.returncode

            _, df_predict = self.read_cfgs(predict_file, predict=True)

        return df_orig, df_predict

    def predict(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """
        calculator = EnergyForceStress(self)
        energy, forces, stress = calculator.calculate(structures=[structure])[0]
        return energy, forces, stress

    def save(self, filename='param.yaml'):
        """
        Save parameters of the potential.

        Args:
            filename (str): The file to store parameters of potential.

        Returns:
            (str)
        """
        with open(filename, 'w') as f:
            yaml.dump(self.param, f)

        return filename

    @staticmethod
    def from_file(filename):
        """
        Initialize potential with parameters file.

        ARgs:
            filename (str): The file storing parameters of potential.

        Returns:
            SOAPotential
        """
        with open(filename) as f:
            param = yaml.load(f)
        return SOAPotential(param=param)
