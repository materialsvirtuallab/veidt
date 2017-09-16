import numpy as np
from veidt.ELSIE.ensemble import SimpleEnsemble, EnsembleRank
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
        self.x_range = np.linspace(-np.pi, np.pi, 201)
        self.target_spect = np.column_stack((self.x_range, abs(np.sin(self.x_range))))
        self.spect_to_identify = pd.DataFrame([(x, deepcopy(self.target_spect)) for x in ['A', 'B', 'C', 'D']],
                                              columns=['unique_key', 'ref_spect'])
        self.ensemble_test = EnsembleRank(self.spect_to_identify, 'ref_spect', self.target_spect, 'unique_key')

    def test_borda_rank_vote(self):
        self.ensemble_test.borda_rank_vote(ensemble_list)
        self.assertEqual(np.unique(self.ensemble_test.dataframe['energy_shift']), 0.0)
