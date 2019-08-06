# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from pymatgen.core.spectrum import Spectrum
import numpy as np
import warnings


class XANES(Spectrum):
    """
    Basic XANES object
    Args:
        x: A sequence of x-ray energies in eV
        y: A sequence of mu(E)
        absorption_specie (Specie): Specie associated with the XANES.
        structure (Structure): Structure associated with the XANES. Pymatgen structure object
        edge (str):
    """

    XLABEL = 'Energy'
    YLABEL = 'Intensity'

    def __init__(self, x, y, absorption_specie, edge, structure=None, e0=None, **kwargs):
        super().__init__(x, y, absorption_specie, edge, structure, e0, **kwargs)
        self.absorption_specie = absorption_specie
        self.edge = edge
        self.structure = structure
        if e0:
            self.e0 = e0
        else:
            warning_msg = 'Edge energy is determined with maximum derivative. Using this e0 with caution.'
            warnings.warn(warning_msg)
            self.e0 = self.x[np.argmax(np.gradient(self.y) / np.gradient(self.x))]

        for (field, value) in kwargs.items():
            setattr(self, field, value)

    def __str__(self):
        if self.structure:
            return "%s %s Edge for %s: %s" % (
            self.absorption_specie, self.edge,
            self.structure.composition.reduced_formula,
            super().__str__()
        )
        else:
            return "%s %s Edge for %s"%(
            self.absorption_specie, self.edge,
            super().__str__()
            )