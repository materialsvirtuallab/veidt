import numpy as np
from veidt.elsie.ensemble import SimpleEnsemble, EnsembleRank
from copy import deepcopy
import json, os
import unittest
import pandas as pd

valid_ensemble = os.path.join(os.path.dirname(__file__), '..', 'valid_ensemble.json')
with open(valid_ensemble) as df:
    ensemble_dict = json.load(df)
ensemble_list = [[value['Preprocessing'], value['Similarity_metric']] for _, value in ensemble_dict.items()]


class EnsembleRankTest(unittest.TestCase):
    def setUp(self):
        self.x_range = np.linspace(-np.pi * 5, np.pi * 5, 2001)
        self.target_spect = np.column_stack((self.x_range, abs(np.sin(self.x_range))))
        self.spect_to_identify = pd.DataFrame([(x, deepcopy(self.target_spect)) for x in ['A', 'B', 'C', 'D']],
                                              columns=['unique_key', 'ref_spect'])
        self.ensemble_test = EnsembleRank(self.spect_to_identify, 'ref_spect', self.target_spect, 'unique_key')

    def test_borda_rank_vote(self):
        self.ensemble_test.borda_rank_vote(ensemble_list)
        self.assertEqual(np.unique(self.ensemble_test.dataframe['energy_shift']), 0.0)
        self.ensemble_test.calculate_softmax_prob()

    def test_exclude_similarity_zero(self):
        no_overlap = np.column_stack((self.x_range + 100, abs(np.sin(self.x_range))))
        no_overlap = pd.DataFrame([(x, deepcopy(no_overlap)) for x in ['E', 'F', 'G', 'H']],
                                  columns=['unique_key', 'ref_spect'])

        some_overlap = np.column_stack((self.x_range + 20, abs(np.sin(self.x_range))))
        some_overlap = pd.DataFrame([(x, deepcopy(some_overlap)) for x in ['I', 'J', 'K']],
                                    columns=['unique_key', 'ref_spect'])

        test_df = pd.concat([self.spect_to_identify, no_overlap, some_overlap])
        test_df = test_df.sample(frac=1)
        test_df.reset_index(inplace=True, drop=True)
        #The failure of assertWarns seems to be bug of python https://bugs.python.org/issue29620 
        #and cannot be reproduced elsewhere
        #with self.assertWarnsRegex(UserWarning, 'less than 30 meV') and \
        #        self.assertWarnsRegex(UserWarning, r'no overlap .* match'):
        ensemble_test = EnsembleRank(test_df, 'ref_spect', self.target_spect, 'unique_key')
        ensemble_test.borda_rank_vote(ensemble_list)
        ensemble_test.calculate_softmax_prob()
        self.assertTrue(ensemble_test.dataframe.shape[0] == 7)
        self.assertTrue(set(ensemble_test.dataframe['unique_key'] == set(['A', 'B', 'C', 'D', 'I', 'J', 'K'])))


class SimpleEnsembleTest(unittest.TestCase):
    def setUp(self):
        self.x_range = np.linspace(-np.pi * 5, np.pi * 5, 2001)
        self.target_spect = np.column_stack((self.x_range, abs(np.sin(self.x_range))))
        self.ref_spect = [deepcopy(self.target_spect)] * 3
        self.simple_test = SimpleEnsemble(self.target_spect, self.ref_spect)

    def test_preprocess_similarity_compute(self):
        self.simple_test.preprocess_similarity_compute('intnorm', 'Cosine')
        self.assertTrue(np.allclose(self.simple_test.spect_df['Similarity'], 1.0))
