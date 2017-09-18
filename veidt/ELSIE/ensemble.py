# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from veidt.ELSIE.spectra_similarity import SpectraSimilarity
import operator
import numpy as np
from copy import deepcopy
import pandas as pd
from collections import defaultdict
from pymatgen.core.spectrum import Spectrum


class EnsembleRank(object):
    """
    Basic ensemble vote algorithm object
    """

    def __init__(self, spect_col_df, spect_column, target_spect, label_column='mp-id'):
        """
        Create an EnsembleRank object
        Args:
            spect_col_df (pandas.DataFrame): A pandas dataframe object that contains at least two columns, one column with unique label, another
                        column corresponding to computational spectra
            spect_column (string): The column in the dataframe that contains all reference spectra the target
                        spectrum needs to compare with
            target_spect (Nx2 array): The target spectrum to be identify
            label_column (string): The column contains unique label/key of each reference spectrum. Used for generating rankings
        """
        self.dataframe = spect_col_df
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.spect_column = spect_column
        self.target_spect = target_spect
        self.label_col = label_column
        self.simple_ensem = SimpleEnsemble(self.target_spect, self.dataframe[self.spect_column].tolist())
        self.dataframe['energy_shift'] = self.simple_ensem.spect_df['energy_shift'].tolist()[:]

    def borda_rank_vote(self, ensemble_pair):
        """
        Use the Borda count method, the Borda count for a spectrum is the sum of the number of spectra ranked
        below it by each individual preprocessing-similarity pair. The Borda count could be referred as a group
        consensus decision
        Args:
            ensemble_pair list(list): A list of preprocessing-similarity pair. Each preprocessing-similarity pair with format:
            [[preproc_1, preproc_2, etc...], similarity_metric].
            Each target spectrum and reference spectrum will be preprocessed using the preprocess methods listed in the list[0].
            Pair-wise similarity between target spectrum and each reference spectrum will be computed using similarity_metric in list[1].

        """

        label_vote_pool = defaultdict(int)

        for proc_comb in ensemble_pair:
            copy_simple_ensem = SimpleEnsemble(self.target_spect, self.dataframe[self.spect_column].tolist())
            copy_simple_ensem.preprocess_similarity_compute(proc_comb[0], proc_comb[1])
            copy_simple_ensem.spect_df.sort_values('Similarity', ascending=True, inplace=True)
            index_rank = copy_simple_ensem.spect_df.index.tolist()

            for index, key in enumerate(index_rank):
                mp_id = self.dataframe.iloc[key][self.label_col]
                label_vote_pool[mp_id] += (index + 1)

        sorted_borda_rank = sorted(label_vote_pool.items(), key=operator.itemgetter(1), reverse=True)
        self.borda_rank = pd.DataFrame.from_records(sorted_borda_rank, columns=[self.label_col, 'borda_rank'])
        self.dataframe = pd.merge(self.dataframe, self.borda_rank, on=self.label_col)

    def calculate_softmax_prob(self, shift_penalty_alpha=0.01):
        """
        Calculate the softmax probability using the computed Borda count and spectrum shift of each spectrum.
        Shift_penalty_alpha is used to penalize the probability of reference spectrum with large spectrum energy shift
        Args:
            shift_penalty_alpha (float): penalize parameter used to adjust the weight of spectrum shift penalization, typical value is
                        between 0.05 to 0.15.

        """

        self.dataframe['exp_normalized_count'] = np.exp(
            self.dataframe['borda_rank'] / self.dataframe['borda_rank'].sum())
        self.dataframe['exp_no_penalty_prob'] = self.dataframe['exp_normalized_count'] / (
            self.dataframe['exp_normalized_count'].sum())
        self.dataframe['abs_shift'] = np.abs(self.dataframe['energy_shift'] - self.dataframe['energy_shift'].mean())
        self.dataframe['neg_shift_alpha'] = np.exp(
            np.negative(shift_penalty_alpha * self.dataframe['abs_shift']) / (self.dataframe['energy_shift'].std()))
        self.dataframe['exp_count_penalty'] = self.dataframe['exp_normalized_count'] * self.dataframe['neg_shift_alpha']
        self.dataframe['exp_prob_penalty'] = self.dataframe['exp_count_penalty'] / (
            self.dataframe['exp_count_penalty'].sum())


class SimpleEnsemble(object):
    """
    Simple ensemble classifier object used for spectrum shift, preprocessing and similarity computation between computed
    reference spectra and target spectrum
    """

    def __init__(self, unknown_spectrum, refdb_spectrum):
        """
        Create a SimpleEnsemble object
        Args:
            unknown_spectrum (Nx2 array): Target spectrum to be compared with reference spectra, N x 2 dimension numpy array.
            refdb_spectrum list(Mx2 array): Reference spectrum, each reference spectrum is an M x 2 dimension numpy array
                with first column corresponding to wavelength and second column corresponding to absorption
        """
        self.u_spect = [unknown_spectrum]
        self.ref_spect = refdb_spectrum
        self.dataframe_init()

    def dataframe_init(self):
        """
        Initialize the comparison pandas dataframe, column 'Unknown_spect' is corresponding to the target spectrum object
        column 'Ref_spect' contains all reference spectrum object

        """
        u_spect_list = self.u_spect * len(self.ref_spect)
        self.spect_df = pd.DataFrame({'Unknown_spect': u_spect_list,
                                      'Ref_spect': self.ref_spect
                                      })
        spect_simi_list = []
        spect_shift_energy = []

        for index, row in self.spect_df.iterrows():
            unknown_spect = Spectrum(row['Unknown_spect'][:, 0], row['Unknown_spect'][:, 1])
            ref_spect = Spectrum(row['Ref_spect'][:, 0], row['Ref_spect'][:, 1])

            spect_simi_obj = SpectraSimilarity(unknown_spect, ref_spect)
            spect_simi_obj._spectrum_shift()
            spect_shift_energy.append(spect_simi_obj.shifted_energy)
            spect_simi_list.append(spect_simi_obj)
        self.spect_df['Spectra_Simi_obj'] = spect_simi_list
        self.spect_df['energy_shift'] = spect_shift_energy

    def preprocess_similarity_compute(self, preprocess, similarity_metric):
        """
        For each row, compute the similarity measure between the target and reference spectra.
        Args:
            preprocess (list/tuple): Preprocessing steps need to taken for each spectrum
            similarity_metric (string): The similarity metric used for comparison.

        Returns:

        """
        self.spect_df['Similarity'] = self.spect_df['Spectra_Simi_obj'].apply(
            lambda x: x.get_shifted_similarity(similarity_metric, spect_preprocess=preprocess))
