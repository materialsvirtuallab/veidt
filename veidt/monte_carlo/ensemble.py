from abc import abstractmethod, ABCMeta
import numpy as np

kb = 8.61733e-5  # eV/K
h = 4.135667e-15  # eV*s
u_to_kg = 1.66054e-27  # kg/u
evs2A_n2_to_kg = 16.0218  # kg/eVs2A_n2
u_to_eV = u_to_kg / evs2A_n2_to_kg


class Ensemble(metaclass=ABCMeta):
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

    def d_exponential(self, state_structure, new_state_structure):
        return self.exponential(new_state_structure) - self.exponential(state_structure)


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
        return hamil - atom_number * kb * temperature * np.log(volume)

    def hamiltonian(self, state_structure):
        energy = self._calc_energy(state_structure)
        p = state_structure.state_dict['pressure'].state
        volume = state_structure.state_dict['volume'].state
        return energy + p * volume


class uVT(Ensemble):
    def __init__(self, model, specie='Na'):
        super(uVT, self).__init__(model)
        self.step_names = ['energy']
        self.specie = specie
        self._cache = {}

    def exponential(self, state_structure):
        mu = state_structure.state_dict['mu'].state
        n = self.get_nb_atom(state_structure, self.specie)
        hamil = self.hamiltonian(state_structure)
        return hamil - mu * n

    def d_exponential(self, state_structure, new_state_structure):
        temperature = state_structure.state_dict['temperature'].state
        mu = state_structure.state_dict['mu'].state
        m = state_structure.state_dict['m'].state
        volume = state_structure.state_dict['volume'].state
        lambda_ = self._cache.get('lambda', None)
        n1 = self.get_nb_atom(state_structure, self.specie)
        n2 = self.get_nb_atom(new_state_structure, self.specie)
        e1 = self.hamiltonian(state_structure)
        e2 = self.hamiltonian(new_state_structure)
        # print(n1, n2, e1, e2, e1-mu*n1, e2-mu*n2)
        if lambda_ is None:
            lambda_ = np.sqrt(h ** 2 / (2 * np.pi * m * u_to_eV * kb * temperature))  # thermal wavelength Angstrom
            self._cache.update({'lambda': lambda_})
        du = self.hamiltonian(new_state_structure) - self.hamiltonian(state_structure)
        mu_prime = mu - 3 * kb * temperature * np.log(lambda_)
        # insertion
        if n1 < n2:
            return du - mu_prime - kb * temperature * np.log(volume / (n1 + 1))
        # deletion
        else:
            return du + mu_prime - kb * temperature * np.log(n1 / volume)

    def hamiltonian(self, state_structure):
        return self._calc_energy(state_structure)

    @staticmethod
    def get_nb_atom(state_structure, specie):
        return state_structure.structure.composition.to_data_dict['unit_cell_composition'][specie]


class SemiUVT(Ensemble):
    def __init__(self, model, specie='Na', fu_species=['Co'], per_formula_unit=True):
        super(SemiUVT, self).__init__(model)
        self.step_names = ['energy']
        self.specie = specie
        self._cache = {}
        self.fu_species = fu_species
        self.is_fraction = per_formula_unit

    def exponential(self, state_structure):
        mu = state_structure.state_dict['mu'].state
        x = self.get_fraction_or_n(state_structure, self.specie)
        hamil = self.hamiltonian(state_structure)
        return hamil - mu * x

    def hamiltonian(self, state_structure):
        return self._calc_energy(state_structure) / self.get_formula_unit(state_structure)

    def get_fraction_or_n(self, state_structure, specie):
        n = state_structure.structure.composition.to_data_dict['unit_cell_composition'][specie]
        fu = self.get_formula_unit(state_structure)
        return n / fu

    def get_formula_unit(self, state_structure):
        if self.is_fraction:
            return np.sum([state_structure.structure.composition.to_data_dict['unit_cell_composition'][i]
                           for i in self.fu_species])
        else:
            return 1
