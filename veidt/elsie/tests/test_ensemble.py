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
        self.x_range = np.linspace(-np.pi*5, np.pi*5, 2001)
        self.target_spect = np.column_stack((self.x_range, abs(np.sin(self.x_range))))
        self.spect_to_identify = pd.DataFrame([(x, deepcopy(self.target_spect)) for x in ['A', 'B', 'C', 'D']],
                                              columns=['unique_key', 'ref_spect'])
        self.ensemble_test = EnsembleRank(self.spect_to_identify, 'ref_spect', self.target_spect, 'unique_key')

    def test_borda_rank_vote(self):
        self.ensemble_test.borda_rank_vote(ensemble_list)
        self.assertEqual(np.unique(self.ensemble_test.dataframe['energy_shift']), 0.0)
        self.ensemble_test.calculate_softmax_prob()


class SimpleEnsembleTest(unittest.TestCase):
    def setUp(self):
        self.x_range = np.linspace(-np.pi*5, np.pi*5, 2001)
        self.target_spect = np.column_stack((self.x_range, abs(np.sin(self.x_range))))
        self.ref_spect = [deepcopy(self.target_spect)] * 3
        self.simple_test = SimpleEnsemble(self.target_spect, self.ref_spect)

    def test_preprocess_similarity_compute(self):
        self.simple_test.preprocess_similarity_compute('intnorm', 'Cosine')
        self.assertTrue(np.allclose(self.simple_test.spect_df['Similarity'], 1.0))
