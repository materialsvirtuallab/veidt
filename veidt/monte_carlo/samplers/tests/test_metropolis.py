import unittest
from pymatgen.core import Structure
from veidt.monte_carlo.base import StateDict
from veidt.monte_carlo.state import StaticState, SingleState, \
    IsingState, SpinStructure
from veidt.monte_carlo.samplers.base import Sampler
from veidt.monte_carlo.samplers.metropolis import Metropolis, proposal
from veidt.abstract import Model, Describer
import pandas as pd
import numpy as np
from veidt.monte_carlo.ensemble import NVT
import os
file_path = os.path.dirname(__file__)

class SimpleLinearModel(Model):
    def __init__(self, describer):
        self.describer = describer

    def fit(self):
        pass

    def predict(self, objs):
        features = self.describer.transform(objs)
        return np.sum(features**2, axis=1)


class NaCount(Describer):
    def describe(self, structure):
        return pd.DataFrame({'Na': [np.sum([i.specie.name == 'Na' for i in structure])],
                             'K': [np.sum([i.specie.name == 'K' for i in structure])]})


class ConstantTemperature(StaticState, SingleState):
    def __init__(self, state, name='temperature'):
        super(ConstantTemperature, self).__init__(state, name)


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.structure = Structure.from_file(os.path.join(file_path, '../../tests/test_NaCoO2.cif'))
        self.state_dict = StateDict([
            ConstantTemperature(3000, 'temperature'), IsingState([0] * 22 + [1, 1])])
        self.spin_struct = SpinStructure(structure=self.structure, species_map={1: "Na", 0: "K"},
                                         state_dict=self.state_dict)

    def test_sampler(self):
        sampler = Sampler(self.spin_struct, ensemble=None)
        self.assertEqual(getattr(sampler, 'temperature'), 3000)
        self.assertIsNone(getattr(sampler, 'atom_number', None))

    def test_proposal(self):
        new_spin_struct = proposal(self.spin_struct)
        self.assertNotEqual(new_spin_struct, self.spin_struct)

    def test_metropolis(self):
        ensemble = NVT(SimpleLinearModel(NaCount()))
        sampler = Metropolis(self.spin_struct, ensemble)
        sampler.sample(1000, verbose=False)
        print(sampler.chain.chain['exponential'][-100:-1:10])
        print(len(sampler.ensemble._cache))
        print(sampler.state_structure.state_dict['ising'])
        print('{}/{} acceptance rate'.format(sampler._acceptance, sampler.n_step))
        self.assertEqual(sampler.chain.chain['exponential'][-1], 288)


if __name__ == '__main__':
    unittest.main()

