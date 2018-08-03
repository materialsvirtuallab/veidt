# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import re
import os
import random
import pandas as pd
from collections import OrderedDict, defaultdict
from scipy.spatial.distance import pdist, squareform

import numpy as np
from monty.io import zopen
from pymatgen import Structure, Lattice
from veidt.potential.processing import convert_docs, pool_from, MonteCarloSampler
from veidt.potential.abstract import Potential
from veidt.describer.atomic_describer import AGNIFingerprints
from veidt.potential.lammps.calcs import EnergyForceStress
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

class AGNIPotential(Potential):
    """
    This class implements Adaptive generalizable neighborhood
    informed potential.
    """
    pair_style = 'pair_style        agni'
    pair_coeff = 'pair_coeff        * * {} {}'
    def __init__(self, name=None):
        """

        Args:
            name (str): Name of force field.
            describer (AGNIFingerprints): Describer to describe
                structures.
        """
        self.name = name if name else 'AGNIPotential'
        self.param = OrderedDict(element=None, interaction=None,
                                 Rc=None, eta=None, sigma=None)
        self.xU = None
        self.yU = None
        self.alphas = None
        self.specie = None

    def from_file(self, filename):
        """
        Get the AGNIPotential parameters from file.

        Args:
            filename (str): Filename to be read.

        Returns:
            (list)
        """
        with zopen(filename) as f:
            lines = f.read()

        generation = int(re.search('generation (.*)', lines).group(1))
        num_elements =  int(re.search('n_elements (.*)', lines).group(1))
        element_type = re.search('\nelement(.*)\n', lines).group(1)
        interaction = re.search('interaction (.*)\n', lines).group(1)
        cutoff = float(re.search('Rc (.*)\n', lines).group(1))
        eta_str = re.search('eta (.*)\n', lines).group(1)
        etas = 1 / np.sqrt(list(map(lambda s: float(s), eta_str.split())))
        sigma = float(re.search('sigma(.*)\n', lines).group(1))
        lambd = float(re.search('lambda(.*)\n', lines).group(1))
        # b = float(re.search('\nb(.*)\n', lines).group(1))
        num_references = int(re.search('n_train(.*)\n', lines).group(1))

        self.param['element'] = element_type
        self.param['interaction'] = element_type
        self.param['Rc'] = cutoff
        self.param['eta'] = list(map(lambda s: float(s), eta_str.split()))
        self.param['sigma'] = sigma
        self.param['n_train'] = num_references
        self.param['lambda'] = lambd

        pattern = re.compile('endVar\s*(.*?)(?=\n$|$)', re.S)
        map_format = lambda string: [float(s) for s in string.split()]
        references_params = np.array(list(map(map_format,
                                   pattern.findall(lines)[0].split('\n'))))
        assert len(references_params) == num_references
        indices = references_params[:, 0]
        xU = references_params[:, 1:-2]
        yU = references_params[:, -2]
        alphas = references_params[:, -1]
        self.xU = xU
        self.yU = yU
        self.alphas = alphas

    def sample(self, datapool=None, r_cut=8, eta_size=8, num_samples=3000,
                            num_attempts=10000, t_init=1000, t_final=100):
        """
        Use metropolis sampling method to select uniform and diverse enough
        data subsets to represent the whole structure-derived features.

        Args:
            data_pool (list): Data pool to be sampled.
            r_cut (float): Cutoff radius to generate high-dimensional
                agni features.
            eta_size (int): The size of dimensions of features, range
                in logarithmic grid between 0.8A - 8A.
            num_samples (int): Number of datasets sampled from datapool.
            num_attempts (int): Number of sampling attempts.
            t_init (int): Initial temperature.
            t_final (int): Final temperature.

        Returns:
            (features, targets)
        """
        def cost_function(x):
            dist_matrix = squareform(pdist(x))
            np.fill_diagonal(dist_matrix, np.Infinity)
            return np.sum(1. / dist_matrix ** 2)

        etas = np.exp(np.linspace(np.log(8), np.log(0.8), eta_size, dtype=np.float16))
        fingerprint = AGNIFingerprints(r_cut=r_cut, etas=etas)
        self.describer = fingerprint

        structures = []
        concat_forces = []
        for dataset in datapool:
            if isinstance(dataset['structure'], dict):
                structure = Structure.from_dict(dataset['structure'])
            else:
                structure = dataset['structure']
            structures.append(structure)
            concat_forces.append(np.array(dataset['outputs']['forces']).ravel())
        self.param['Rc'] = r_cut
        self.param['element'] = structure.symbol_set[0]
        self.param['interaction'] = structure.symbol_set[0]
        self.param['eta'] = list(1 / (etas ** 2))
        self.specie = structure.symbol_set[0]
        features = fingerprint.describe_all(structures).values
        targets = np.concatenate(concat_forces)
        assert features.shape[0] == len(targets)

        mcsampler = MonteCarloSampler(datasets=features, num_samples=num_samples,
                                      cost_function=cost_function)
        mcsampler.sample(num_attempts=num_attempts, t_init=t_init, t_final=t_final)
        features_selected = features[mcsampler.index_selected]
        targets_selected = targets[mcsampler.index_selected]
        return features, targets, features_selected, targets_selected

    def fit(self, features, targets, cv=5, alpha=1e-8,
            scoring_criteria='neg_mean_absolute_error', threshold=1e-3):
        """
        Fit the dataset with kernel ridge regression.

        Args:
            features (np.array): features X.
            targets (np.array): targets y.
            cv (int): The numbre of folds in cross validation.
                Default to 5.
            alpha (float): Small positive number.
                Regularization parameter in KRR.
            scoring (str): The scoring strategy to evaluate the
                prediction on test sets. The same as the scoring
                parameter in sklearn.model_selection.GridSearchCV.
                Default to 'neg_mean_absolute_error', i.e. MAE.
            threshold (float): The converged threshold of final
                optimal sigma.

        Returns:
            (float) The optimized sigma.
        """
        st_gamma = -np.inf
        nd_gamma = np.inf
        gamma_trials = np.logspace(-6, 4, 11)
        while(abs(st_gamma - nd_gamma) > threshold):
            kr = GridSearchCV(KernelRidge(kernel='rbf', alpha=alpha,
                              gamma=0.1), cv=cv, param_grid={"gamma": gamma_trials},
                              return_train_score=True)
            kr.fit(features, targets)
            cv_results = pd.DataFrame(kr.cv_results_)
            st_gamma = cv_results['param_gamma'][cv_results['rank_test_score'] \
                                           == 1].iloc[0]
            nd_gamma = cv_results['param_gamma'][cv_results['rank_test_score'] \
                                           == 2].iloc[0]
            gamma_trials = np.linspace(min(st_gamma, nd_gamma), max(st_gamma, nd_gamma), 10)
        gamma = st_gamma

        K = np.exp(-gamma * squareform(pdist(features)) ** 2)
        alphas = np.dot(np.linalg.inv(K \
                        + alpha * np.eye(len(features))), targets)
        kkr = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
        kkr.fit(features, targets)

        self.param['n_train'] = len(features)
        self.param['lambda'] = alpha
        self.param['sigma'] = 1 / np.sqrt(2 * gamma)
        self.xU = features
        self.yU = targets
        self.predictor = kkr
        self.alphas = alphas

        return gamma

    def train(self, train_structures, energies=None, forces=None,
                        stresses=None, **kwargs):
        """
        Training data with agni method.

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
        """
        train_pool = pool_from(train_structures, energies, forces, stresses)
        _, _, features, targets = self.sample(train_pool, **kwargs)
        gamma = self.fit(features, targets)
        return 0

    def write_param(self):
        """
        Write fitted agni parameter file to perform lammps calculation.
        """
        filename = 'ref.agni'
        if not self.param or self.xU is None or self.yU is None \
                or self.alphas is None:
            raise RuntimeError("The parameters should be provided.")

        assert len(self.alphas) == len(self.yU)
        lines = [' '.join([key] + [str(f) for f in value])
                            if isinstance(value, list)
                            else ' '.join([key, str(value)])
                            for key, value in self.param.items()]
        lines.insert(0, 'generation 1')
        lines.insert(1, 'n_elements 1')
        lines.append('endVar\n')

        for index, (x, y, alpha) in enumerate(zip(self.xU, self.yU, self.alphas)):
            index_str = str(index)
            x_str = ' '.join([str(f) for f in x])
            y_str = str(y)
            alpha_str = str(alpha)
            line = '{} {} {} {}'.format(index_str, x_str, y_str, alpha_str)
            lines.append(line)

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        pair_coeff = self.pair_coeff.format(filename, self.specie)
        ff_settings = [self.pair_style, pair_coeff]

        return ff_settings

    def evaluate2(self, test_structures, ref_energies=None,
                    ref_forces=None, ref_stresses=None):
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
        """
        predict_pool = pool_from(test_structures, ref_energies,
                                 ref_forces, ref_stresses)
        _, df_orig = convert_docs(predict_pool)

        efs_calculator = EnergyForceStress(ff_settings=self)
        efs_results = efs_calculator.calculate(test_structures)

        assert len(test_structures) == len(efs_results)

        data_pool = []
        for struct, (energy, forces, stresses) in zip(test_structures, efs_results):
            d = {'outputs': {}}
            d['structure'] = struct.as_dict()
            d['num_atoms'] = len(struct)
            d['outputs']['energy'] = energy
            d['outputs']['forces'] = forces
            d['outputs']['virial_stress'] = stresses

            data_pool.append(d)
        _, df_pred = convert_docs(data_pool)

        return df_orig, df_pred

    def evaluate(self, test_structures, ref_energies, ref_forces, ref_stresses):
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
        """
        predict_pool = pool_from(test_structures, ref_energies,
                                 ref_forces, ref_stresses)
        _, df_orig = convert_docs(predict_pool)

        data_pool = []
        for struct in test_structures:
            d = {'outputs': {}}
            d['structure'] = struct.as_dict()
            d['num_atoms'] = len(struct)
            features = self.describer.describe(struct)
            targets = self.predictor.predict(features.values)
            d['outputs']['energy'] = 0
            d['outputs']['forces'] = targets.reshape((-1, 3))
            d['outputs']['virial_stress'] = [0., 0., 0., 0., 0., 0.]
            data_pool.append(d)
        _, df_pred = convert_docs(data_pool)
        return df_orig, df_pred