from veidt.monte_carlo.state import StaticState
from veidt.monte_carlo.state import Chain
kb = 8.61733e-5


class Sampler(object):
    def __init__(self, state_structure, ensemble):
        """

        :param state_structure: StateStructure object
        :param ensemble: Ensemble object
        """
        self.state_structure = state_structure
        self.ensemble = ensemble
        self.chain = Chain()

        # Add static state as attributes, e.g., constant temperature etc
        for i, j in self.state_structure.state_dict.items():
            if isinstance(j, StaticState):
                setattr(self, i, j.state)
        self.n_step = 0
        self._acceptance = 0

    def step(self):
        """
        One Monte Carlo step
        :return:
        """
        pass

    def sample(self, n, n_print=10, verbose=True):
        """
        Monte Carlo sampling for n steps, and print the exponential factor
        every n_print steps if verbose is True

        :param n: int, number of samples
        :param n_print: int, print interval
        :param verbose: bool, print indicator
        :return: the chain of samples
        """
        while self.n_step < n:
            self.step(n_print, verbose)
        return self.chain



