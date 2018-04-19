import unittest

from pymatgen.core import Structure
from veidt.monte_carlo.base import StateDict
from veidt.monte_carlo.state import AtomNumberState, IsingState, StaticState
from veidt.monte_carlo.state import SpinStructure, Chain
import os
file_path = os.path.dirname(__file__)


def unequal_site_number(list1, list2):
    return sum([i != j for i, j in zip(list1, list2)])


class TestMonteCarlo(unittest.TestCase):

    def test_ising_state(self):
        ising_state = IsingState([0, 1, 0, 1])
        new_ising_state = ising_state.copy()
        self.assertListEqual(ising_state.state, new_ising_state.state)
        self.assertEqual(ising_state, IsingState([0, 1, 0, 1], 'ising2'))
        self.assertEqual(ising_state.n, 4)
        self.assertListEqual(ising_state.state, [0, 1, 0, 1])
        self.assertEqual(ising_state.name, 'ising')
        ising_state.change()
        self.assertEqual(unequal_site_number(ising_state.state, [0, 1, 0, 1]), 1)

    def test_atom_number_state(self):
        atom_number = AtomNumberState(10)
        self.assertEqual(atom_number.state, 10)
        atom_number.change()
        self.assertIn(atom_number.state, [9, 11])

    def test_spin_structure(self):
        species_map = {0: 'K', 1: 'Na'}
        structure = Structure.from_file(os.path.join(file_path, 'test_NaCoO2.cif'))
        state_dict = StateDict([StaticState(100, 'temperature'),
                                AtomNumberState(10),
                                IsingState([0]*22+[1, 1])])
        spin_struct = SpinStructure(structure, state_dict, species_map)
        self.assertListEqual(spin_struct.state_dict['ising'].state, [0] * 22 + [1, 1])

        orig_specie_list = spin_struct.to_specie_list()

        # test move method
        spin_struct = SpinStructure(structure, state_dict, species_map)
        spin_struct.change()
        self.assertEqual(unequal_site_number(spin_struct.state_dict['ising'].state, [0] * 22 + [1, 1]), 1)
        specie_list = spin_struct.to_specie_list()
        self.assertEqual(unequal_site_number(orig_specie_list, specie_list), 1)

        # test from_states
        spin_struct.from_states(
            StateDict([StaticState(1000, 'temperature'),
                       AtomNumberState(10), IsingState([0]*20+[1, 1] + [0, 0])]))
        self.assertEqual(unequal_site_number(spin_struct.to_specie_list(), orig_specie_list), 4)
        self.assertEqual(unequal_site_number(spin_struct.to_states()['ising'].state, [0]*22 + [1, 1]), 4)
        # test structure to states
        self.assertListEqual(spin_struct.structure_to_states(structure)['ising'].state,
                             [0] * 22 + [1, 1])


    def test_chain(self):
        spin_state = IsingState([0, 1, 0])
        atom_state = AtomNumberState(10)
        state_dict = StateDict([spin_state, atom_state])
        chain = Chain()
        chain.append(state_dict)
        chain.append(StateDict([AtomNumberState(20), IsingState([1, 1, 1])]))
        self.assertListEqual(chain.chain['ising'][0], [0, 1, 0])
        self.assertListEqual(chain.chain['ising'][1], [1, 1, 1])
        self.assertListEqual(chain.chain['atom_number'], [10, 20])
        self.assertIs(spin_state._chain, chain)
        self.assertEqual(spin_state._chain.length, 2)

if __name__ == '__main__':
    unittest.main()

