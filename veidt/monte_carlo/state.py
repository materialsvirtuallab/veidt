import numpy as np
from collections import defaultdict
from .base import State, StateStructure
import itertools


class IsingState(State):
    def __init__(self, spin_list, name='ising'):
        """
        :param spin_list: list, 0 and 1's indicating spin
        :param name: string, a name string for the state
        """
        super(IsingState, self).__init__(spin_list, name)
        self.n = len(self.state)
        self.label = 'energy'

    def change(self):
        """flip one site"""
        index = np.random.choice(self.n)
        self.state[index] = 1 - self.state[index]

    def check_state(self):
        """
        Check if only two states 0, 1 present
        :return: raise error if criterion not satisfied
        """
        if not self.state:
            raise ValueError('Empty spin')
        if set(self.state).union({0, 1}) != {0, 1}:
            raise ValueError('Spin can only be 0 and 1')
        return

    def __str__(self):
        """
        string representation of the state
        :return: string
        """
        return self.name + ' ' + str(self.state)


class SingleState(State):
    """
    One single variable state, such as temperature, pressure etc
    """
    def check_state(self):
        """
        check if only a scalar value is in the state
        :return:
        """
        if np.size(self.state) > 1:
            raise ValueError("Only single state supported!")
        return


class StaticState(State):
    """
    StaticState does not change the state when calling the change method
    """
    def change(self):
        pass


class AtomNumberState(SingleState):
    """
    Number of atom state
    """
    def __init__(self, atom_number, name='atom_number'):
        super(AtomNumberState, self).__init__(atom_number, name)

    def change(self):
        """
        Add or delete one atom
        :return:
        """
        self.state += np.random.choice([1, -1])


class SpinStructure(StateStructure):
    """
    Conversion class between binary spin and structure
    """
    def __init__(self, structure, state_dict, species_map):
        """

        :param structure: pymatgen structure object
        :param state_dict: StateDict object
        :param species_map: dict, map spin to structure, e.g., {1: "Na", 0: "K"}
        """
        super(SpinStructure, self).__init__(structure, state_dict)

        self.species_map = species_map
        keys = self.species_map.keys()
        if len(keys) == 2 and (1 in keys) and (0 in keys):
            pass
        else:
            raise ValueError(('dict arguments must provide mappings from ',
                              '0 and 1 to species'))
        self.site_index = sorted(list(itertools.chain(
            *[self.structure.indices_from_symbol(i) for i in self.species_map.values()])))
        structure_dict = self.to_states()
        if self.state_dict != structure_dict:
            self.from_states(self.state_dict)

    def structure_from_states(self, state_dict):
        """
        Convert IsingState into pymatgen structure
        :param state_dict: StateDict object
        :return: pymatgen structure
        """
        spin_list = state_dict['ising'].state
        new_structure = self.structure.copy()
        if len(self.site_index) != len(spin_list):
            raise ValueError("The spin list has to be the same length as structure sites")
        for i, j in zip(self.site_index, spin_list):
            new_structure[i] = self.species_map[j]
        return new_structure

    def structure_to_states(self, structure):
        """
        Convert pymatgen structure into IsingState
        :param structure: pymatgen structure
        :return: StateDict
        """
        state_vector = \
            [0 if i.specie.name == self.species_map[0] else 1 for i in structure
             if i.specie.name in self.species_map.values()]
        new_state_dict = self.state_dict.copy()
        new_state_dict['ising'] = IsingState(state_vector)
        return new_state_dict

    def to_specie_list(self):
        """
        Convert the spin list to species list using the species_map
        :return: list, a list of species string
        """
        return [i.specie.name for i in self.structure if i.specie.name in self.species_map.values()]

    def copy(self):
        """
        Copy a new StateStructure
        :return: StateStructure
        """
        return self.__class__(self.structure.copy(), self.state_dict.copy(), self.species_map)

    def __str__(self):
        """
        string representation of the StateStructure
        :return: string
        """
        return ' '.join([str(i) + str(j.state) for i, j in self.state_dict.items() if i != 'temperature'])


class Chain(object):
    """
    A chain of states class

    To do, need to check states variable changes over steps"""
    def __init__(self):
        self.chain = defaultdict(list)
        self.length = 0
        self.current_state = None

    def append(self, state_dict):
        """Append new states
        :param state_dict: StateDict object
        """

        # append state variables
        for state_name, state in state_dict.items():
            self.chain[state_name].append(state.state)
            # state can observe the chain change
            state._chain = self

        self.length += 1
        self.current_state = state_dict

    @property
    def state_names(self):
        """
        state name list
        :return: list, state names
        """
        return list(self.chain.keys())
