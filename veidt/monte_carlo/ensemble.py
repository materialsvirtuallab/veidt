import six
from abc import abstractmethod, ABCMeta
import numpy as np
import math
kb = 8.61733e-5  # eV/K
h = 4.135667e-15  # eV*s
u_to_kg = 1.66054e-27  # kg/u
evs2A_n2_to_kg = 16.0218  # kg/eVs2A_n2
u_to_eV = u_to_kg / evs2A_n2_to_kg


class Ensemble(six.with_metaclass(ABCMeta)):
    def __init__(self, model):
        """

        :param model: veidt model, a model that can predict energy from structure
        """
        self.model = model
        # Different ensembles will have different sampling steps
        self.steps = []
        # cache the self.model prediction results
        self._cache = {}

    def _calc_energy(self, state_structure):
        """
        Calculate the energy using model
        :param state_structure: StateStructure object
        :return: float, energy value
        """
        state_string = str(state_structure)
        energy = self._cache.get(state_string, None)
        if energy is None:
            energy = self.model.predict([state_structure.structure])[0]
            self._cache.update({state_string: energy})
        return energy

    @abstractmethod
    def exponential(self, state_structure):
        """
        Calculate the exponential factor in ensemble distributions
        :param state_structure: StateStructure object
        :return: float, exponential factor in ensemble averaging
        """
        pass

    @abstractmethod
    def hamiltonian(self, state_structure):
        """
        hamiltonian corresponding the the StateStructure in the ensemble
        :param state_structure: StateStructure object
        :return: float, hamiltonian
        """
        pass


class NVT(Ensemble):
    def __init__(self, model):
        super(NVT, self).__init__(model)
        self.step_names = ['energy']

    def exponential(self, state_structure):
        hamil = self.hamiltonian(state_structure)
        return hamil

    def hamiltonian(self, state_structure):
        return self._calc_energy(state_structure)


class NPT(Ensemble):
    def __init__(self, model):
        super(NPT, self).__init__(model)
        self.step_names = ['energy', 'volume']

    def exponential(self, state_structure):
        hamil = self.hamiltonian(state_structure)
        volume = state_structure.state_dict['volume'].state
        atom_number = state_structure.state_dict['atom_number'].state
        temperature = state_structure.state_dict['temperature'].state
        return hamil - \
            atom_number * kb * temperature * np.log(volume)

    def hamiltonian(self, state_structure):
        energy = self._calc_energy(state_structure)
        p = state_structure.state_dict['pressure'].state
        volume = state_structure.state_dict['volume'].state
        return energy + p * volume


class uVT(Ensemble):
    def __init__(self, model):
        super(uVT, self).__init__(model)
        self.step_names = ['energy', 'atom_number']
        self._cache = {}

    def exponential(self, state_structure):
        temperature = state_structure.state_dict['temperature'].state
        m = state_structure.state_dict['m'].state
        volume = state_structure.state_dict['volume'].state
        lamb = self._cache.get('lambda', None)
        n = state_structure.state_dict['atom_number'].state
        if lamb is None:
            lamb = np.sqrt(h**2/(2*np.pi*m*u_to_eV*kb*temperature))  # thermal wavelength Angstrom
            self._cache.update({'lambda': lamb})
        hamil = self.hamiltonian(state_structure)
        return hamil + kb*temperature*(3*n*np.log(lamb) + np.log(math.factorial(n)) - \
                                                 n*np.log(volume))

    def hamiltonian(self, state_structure):
        energy = self._calc_energy(state_structure)
        mu = state_structure.state_dict['mu'].state
        n = state_structure.state_dict['atom_number'].state
        return energy - mu * n
