from .base import Sampler, kb
import numpy as np
from inspect import signature


class Metropolis(Sampler):
    """
    A metropolis sampler
    """
    def __init__(self, state_structure, ensemble):
        """

        :param state_structure: StateStructure object
        :param ensemble: Ensemble object
        """
        super(Metropolis, self).__init__(state_structure, ensemble)
        self.exponential = self.ensemble.exponential(self.state_structure)
        self.chain.append(self.state_structure.state_dict)
        self.chain.chain['exponential'].append(self.exponential)

    def step(self, n_print, verbose):
        """
        A metropolis step
        :return:
        """
        self._substeps(n_print, verbose)

    def _substeps(self, n_print, verbose):
        """
        Different ensembles will have different sampling steps, this method calls the steps
        sequentially
        :return:
        """
        if verbose:
            if self.n_step % n_print == 0:
                print('{}th sample exponential {}\n'.format(self.n_step + 1, self.exponential))
        for step_name in self.ensemble.step_names:
            temperature = self.state_structure.state_dict['temperature'].state
            new_state_structure = proposal(self.state_structure, step_name)
            # exponential = self.ensemble.exponential(new_state_structure)
            d_exp = self.ensemble.d_exponential(self.state_structure, new_state_structure)
            # print('delta exp: ', d_exp)
            is_accept = accept(d_exp, temperature)
            if is_accept:
                self.state_structure = new_state_structure
                self.exponential = self.ensemble.exponential(new_state_structure)
                self._acceptance += 1
            self.chain.append(self.state_structure.state_dict)
            self.chain.chain['exponential'].append(self.exponential)
            self.n_step += 1


def proposal(state_structure, step_name=None):
    """
    propose a new state
    :param state_structure: initial structure with state
    :param state_name: the state to change
    :return: new_state_structure
    """
    init_sig = signature(state_structure.__init__)
    p_names = [p.name for p in init_sig.parameters.values() if p.name != 'self']
    out_dict = {i: getattr(state_structure, i) for i in p_names}
    out_dict.update({'structure': state_structure.structure.copy(),
                     'state_dict': state_structure.state_dict.copy()})
    new_state_structure = state_structure.__class__(**out_dict)
    if step_name is None:
        new_state_structure.change()
    else:
        [i.change() for i in new_state_structure.state_dict.values() if i.label==step_name]
        #new_state_structure.state_dict[step_name].change()
        new_state_structure.from_states(new_state_structure.state_dict)
    return new_state_structure


def accept(d_exp, temperature):
    """
    :param d_exp: proposal exponential factor minus the old exponential factor, energy unit
    :param temperature:
    :return: bool, acceptance
    """

    if d_exp <= 0:
        return True
    elif np.random.uniform() < np.exp(-d_exp/kb/temperature):
        return True
    else:
        return False
