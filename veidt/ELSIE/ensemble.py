from veidt.ELSIE.spectra_similarity import SpectraSimilarity
import operator, os, json
import numpy as np
from copy import deepcopy
import pandas as pd
from collections import defaultdict
from pymatgen.core.spectrum import Spectrum


class EnsembleRank(object):
    """
    Basic ensemble vote algorithm object
    """

    def __init__(self, spect_col_df, spect_column, to_identify_spect, mp_id_column='mp-id'):
        """
        Create an EnsembleRank object
        Args:
            spect_col_df: A pandas dataframe object that contains at least two columns, one column with mp-ids, another
                        column corresponding to computational spectra
            spect_column: The column name of the column in the dataframe that contains all reference spectra the target
                        spectrum needs to compare with
            to_identify_spect: The target spectrum to be identify, a pymatgen.core.spectrum object
            mp_id_column: The column name of the dataframe's mp-id column, a unique key to label each reference spectrum
        """
        self.dataframe = spect_col_df
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.spect_column = spect_column
        self.to_identify_spect = to_identify_spect
        self.mp_id_col = mp_id_column
        self.simple_ensem = SimpleEnsemble(self.to_identify_spect, self.dataframe[self.spect_column].tolist())
        self.dataframe['energy_shift'] = self.simple_ensem.spect_df['energy_shift'].tolist()[:]

    def borda_rank_vote(self, ensemble_pair):
        """
        Use the Borda count method, the Borda count for a spectrum is the sum of the number of spectra ranked
        below it by each individual preprocessing-similarity pair. The Borda count could be referred as a group
        consensus decision
        Args:
            ensemble_pair: A list of preprocessing-similarity pair with format: [[preproc_1, preproc_2, etc...], similarity_1].
            Similarity will be calculated using list[1] after preprocessing steps listed in the list[0]

        """

        mp_vote_pool = defaultdict(int)

        for proc_comb in ensemble_pair:
            copy_simple_ensem = deepcopy(self.simple_ensem)
            copy_simple_ensem.preprocess_similarity_compute(proc_comb[0], proc_comb[1])
            copy_simple_ensem.spect_df.sort_values('Similarity', ascending=True, inplace=True)
            index_rank = copy_simple_ensem.spect_df.index.tolist()

            for index, key in enumerate(index_rank):
                mp_id = self.dataframe.iloc[key][self.mp_id_col]
                mp_vote_pool[mp_id] += (index + 1)

        sorted_borda_rank = sorted(mp_vote_pool.items(), key=operator.itemgetter(1), reverse=True)
        self.borda_rank = pd.DataFrame.from_records(sorted_borda_rank, columns=[self.mp_id_col, 'borda_rank'])
        self.dataframe = pd.merge(self.dataframe, self.borda_rank, on=self.mp_id_col)

    def calculate_softmax_prob(self, shift_penalty_alpha=0.05):
        """
        Calculate the softmax probability using the computed Borda count and spectrum shift of each spectrum.
        Shift_penalty_alpha is used to penalize the probability of reference spectrum with large spectrum energy shift
        Args:
            shift_penalty_alpha: penalize parameter used to adjust the weight of spectrum shift penalization, typical value is
                        between 0.05 to 0.15.

        """

        self.dataframe['exp_normalized_count'] = np.exp(
            self.dataframe['borda_rank'] / self.dataframe['borda_rank'].sum())
        self.dataframe['exp_no_penalty_prob'] = self.dataframe['exp_normalized_count'] / (
            self.dataframe['exp_normalized_count'].sum())
        self.dataframe['abs_shift'] = np.abs(self.dataframe['energy_shift'] - self.dataframe['energy_shift'].mean())
        self.dataframe['neg_shift_alpha'] = np.exp(
            np.negative(shift_penalty_alpha * self.dataframe['abs_shift']) / (self.dataframe['energy_shift'].std()))
        #         self.dataframe['neg_shift_alpha'] = np.exp(np.negative(shift_penalty_alpha*self.dataframe['abs_shift'])/(np.std(self.dataframe['energy_shift'])))
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
            unknown_spectrum: Target spectrum to be compared with reference spectra, N x 2 dimension numpy array.
            refdb_spectrum: Reference spectrum, each reference spectrum is an M x 2 dimension numpy array
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
            preprocess: Preprocessing steps need to taken for each spectrum
            similarity_metric: The similarity metric used for comparison.

        Returns:

        """
        self.spect_df['Similarity'] = self.spect_df['Spectra_Simi_obj'].apply(
            lambda x: x.get_shifted_similarity(similarity_metric, spect_preprocess=preprocess))
