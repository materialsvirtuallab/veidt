import unittest
from pymatgen.core import Structure
import pandas as pd
import numpy as np
from veidt.monte_carlo.ensemble import NVT, NPT, uVT
from veidt.monte_carlo.base import State
from veidt.monte_carlo.state import StaticState
from veidt.monte_carlo.state import AtomNumberState, IsingState
from veidt.monte_carlo.base import StateDict
from veidt.monte_carlo.state import SpinStructure, Chain
from veidt.abstract import Model, Describer


class SimpleLinearModel(Model):
    def __init__(self, describer):
        self.describer = describer

    def fit(self):
        pass

    def predict(self, objs):
        features = self.describer.transform(objs)
        return np.sum(5*features**2, axis=1)


class NaCount(Describer):
    def describe(self, structure):
        return pd.DataFrame({'Na': [np.sum([i.specie.name == 'Na' for i in structure])],
                             'K': [np.sum([i.specie.name == 'K' for i in structure])]})

class Volume(State):
    def __init__(self, state, name='volume'):
        super(Volume, self).__init__(state, name)
        self.label = 'volume'
    def change(self):
        self.state += (np.random.rand(1) - 0.5) * 0.1 * self.state


class TestHamiltonian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_nvt(self):
        structure = Structure.from_file('test_NaCoO2.cif')
        model = SimpleLinearModel(NaCount())
        nvt = NVT(model)
        state_dict = StateDict([StaticState(100, 'temperature'),
                                AtomNumberState(10),
                                IsingState([0]*22+[1, 1])])
        spin_struct = SpinStructure(structure=structure, species_map={1: "Na", 0: "K"},
                                    state_dict=state_dict)
        energy1 = nvt.exponential(spin_struct)
        self.assertEqual(energy1, 2440)
        spin_struct.change()
        energy2 = nvt.exponential(spin_struct)
        self.assertIn(energy2, (2250, 2650))

    def test_npt(self):
        structure = Structure.from_file('test_NaCoO2.cif')
        model = SimpleLinearModel(NaCount())
        npt = NPT(model)
        state_dict = StateDict([StaticState(100, 'temperature'),
                                Volume(10, 'volume'),
                                StaticState(-1, 'pressure'),
                                AtomNumberState(10),
                                IsingState([0, 1]*12)])
        spin_struct = SpinStructure(structure=structure, species_map={1: "Na", 0: "K"},
                                    state_dict=state_dict)
        energy1 = npt.exponential(spin_struct)
        self.assertAlmostEqual(energy1, 1429.80157, 4)
        spin_struct.change()
        energy2 = npt.exponential(spin_struct)
        self.assertLessEqual(energy2, 1429.80157 + 10 + 5)
        self.assertLessEqual(1429.80157 + 10 - 5, energy2)

    def test_uVT(self):

        structure = Structure.from_file('test_NaCoO2.cif')
        model = SimpleLinearModel(NaCount())
        uvt = uVT(model)
        state_dict = StateDict([StaticState(100, 'temperature'),
                                StaticState(10, 'volume'),
                                StaticState(23, 'm'),
                                StaticState(-3, 'mu'),
                                StaticState(-1, 'pressure'),
                                AtomNumberState(10),
                                IsingState([0, 1]*12)])
        spin_struct = SpinStructure(structure=structure, species_map={1: "Na", 0: "K"},
                                    state_dict=state_dict)
        energy1 = uvt.exponential(spin_struct)
        self.assertAlmostEqual(energy1, 1469.6704985, 4)


if __name__ == '__main__':
    unittest.main()

