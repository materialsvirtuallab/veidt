
from veidt.rfxas.core import XANES
import pandas as pd
import os, unittest
import warnings

comp_test_df_path = os.path.join(os.path.dirname(__file__), 'comp_spectra_test.pkl')
comp_test_df = pd.read_pickle(comp_test_df_path)

class RfxasXANESTest(unittest.TestCase):
    def setUp(self):
        self.test_row = comp_test_df.iloc[0]
        self.test_row_formula = self.test_row['formula']
        self.test_row_ele_group = self.test_row['ele_tm_alka_metalloid']
        self.test_row_xas_id = self.test_row['xas_id']
        self.test_row_absorb_specie = self.test_row['absorbing_species']
        self.test_row_energy_e0 = self.test_row['energy_e0']
        self.test_row_structure = self.test_row['structure']
        self.test_row_x = self.test_row['x_axis_energy_55eV']
        self.test_row_spect = self.test_row['interp_spectrum_55eV']
        self.test_row_add_paras = {
            'composition': self.test_row_formula, 'elemental_group': self.test_row_ele_group,
            'xas_id': self.test_row_xas_id
        }

    def test_raise_warning(self):
        with warnings.catch_warnings(record=True) as w:
            xanes_test = XANES(self.test_row_x, self.test_row_spect, self.test_row_absorb_specie, edge='K',
                               **self.test_row_add_paras)
            self.assertTrue('maximum derivative' in str(w[-1].message))
            self.assertEqual(xanes_test.composition, 'NaB(CO2)4')
            self.assertEqual(len(xanes_test.x), 200)
            self.assertEqual(xanes_test.xas_id, 'mp-559618-4-XANES-K')
            self.assertEqual(xanes_test.elemental_group, 'Carbon')

        with warnings.catch_warnings(record=True) as w:
            xanes_test_2 = XANES(self.test_row_x, self.test_row_spect, self.test_row_absorb_specie, edge='K',
                               e0=self.test_row_energy_e0, **self.test_row_add_paras)
            self.assertEqual(len(w), 0)
            self.assertEqual(xanes_test_2.composition, 'NaB(CO2)4')
            self.assertEqual(len(xanes_test_2.x), 200)
            self.assertEqual(xanes_test_2.e0, 274.98)
            self.assertEqual(xanes_test_2.xas_id, 'mp-559618-4-XANES-K')
            self.assertEqual(xanes_test_2.elemental_group, 'Carbon')