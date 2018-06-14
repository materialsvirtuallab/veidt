from abc import abstractmethod, ABCMeta
import six
from collections import Iterable, OrderedDict
from copy import copy


class State(six.with_metaclass(ABCMeta)):
    def __init__(self, state, name=None):
        """
        Base class for State.
        :param state: scalar or iterables
        :param name: string name
        """
        self.state = state
        self.check_state()
        self.name = name
        self.label = name

    @abstractmethod
    def change(self):
        """
        Change state to new state
        :return:
        """
        pass

    def check_state(self):
        """
        Check if the state make sense, raise Error if not
        :return:
        """
        pass

    def __eq__(self, other):
        """
        Compare two states
        :param other: other state object
        :return: bool
        """
        if isinstance(self.state, Iterable):
            return all([i == j for i, j in zip(self.state, other.state)])
        else:
            return self.state == other.state

    def copy(self):
        """
        copy a state
        :return: new state object with same state variable
        """
        new_state = self.__class__(copy(self.state), self.name)

        # copy other attributes
        # new_state.__dict__.update({i: j for i, j in self.__dict__.items() if i not in ['state', 'name']})
        return new_state


class StaticState(State):
    """
    StaticState does not change the state when calling the change method
    """
    def change(self):
        pass


class StateDict(OrderedDict):
    """
    A collection of states. Usually one physical system is described by more than
    one state variable
    """
    def __init__(self, states=None, **kwargs):
        if isinstance(states, (list, tuple)):
            super(StateDict, self).__init__({i.name: i for i in states})
        if kwargs is not None:
            for key, value in kwargs.items():
                self.update({key: StaticState(value, name=key)})

    def __eq__(self, other):
        keys = self.keys()
        return all([self[i] == other[i] for i in keys])

    def copy(self):
        """
        Deep copy of a StateDict
        :return: new StateDict object
        """
        new_state_dict = StateDict.fromkeys(self.keys())
        new_state_dict.update({i: self[i].copy() for i in self.keys()})
        return new_state_dict


class StateStructure(six.with_metaclass(ABCMeta)):
    """
    Structure with StateDict to describe the states
    Each structure will be associated with a collection of state and can be converted to or from the states
    """
    def __init__(self, structure, state_dict):
        """

        :param structure: pymatgen structure
        :param state_dict: StateDict object
        """
        self.structure = structure.copy()
        self.state_dict = state_dict.copy()

    @abstractmethod
    def structure_from_states(self, state_dict):
        """
        Convert the state into pymatgen structure
        : param state_dict: StateDict object
        :return: structure corresponding to the state dictionary
        """
        pass

    @abstractmethod
    def structure_to_states(self, structure):
        """
        Convert structure to corresponding state dictionary
        :param structure: pymatgen structure
        :return: state dictionary
        """
        pass

    def to_states(self):
        """
        Convert the object to state dictionary
        :return: StateDict object
        """
        return self.structure_to_states(self.structure)

    def from_states(self, state_dict):
        """
        Convert a state dictionary into structure
        :param state_dict: StateDict object
        :return:
        """
        self.structure = self.structure_from_states(state_dict)
        self.state_dict = state_dict

    def change(self):
        """
        Perform state changes for all items in the state dictionary and
        update the structure
        :return:
        """
        [i.change() for i in self.state_dict.values()]
        self.from_states(self.state_dict)








