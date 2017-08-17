# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from veidt.abstract import Describer


class Generator(Describer):

    def __init__(self, funcs, labels):
        """
        :param funcs [func]: List of functions to map inputs to a
            single output value.
        :param labels [str]: List of strings to label the output
            of each function.
        """
        assert len(funcs) == len(labels), \
            "No. of functions and labels UNEQUAL."
        self.funcs = funcs
        self.labels = labels

    def describe(self, obj):
        """
        Returns description of an object based on all functions.

        :param obj: Object to be described.
        :return: {label: value} dict.
        """
        output = {}
        for f, l in zip(self.funcs, self.labels):
            output[l] = f(obj)
        return output


class DistinctSiteProperty(Describer):
    """
    Constructs a descriptor based on properties of distinct sites in a
    structure. For now, this assumes that there is only one type of species in
    a particular Wyckoff site.
    """
    #todo: generalize to multiple sites with the same Wyckoff.

    def __init__(self, wyckoffs, properties, symprec=0.1):
        """
        :param wyckoffs: List of wyckoff symbols. E.g., ["48a", "24c"]
        :param properties: Sequence of specie properties. E.g., ["atomic_radius"]
        :param symprec: Symmetry precision for spacegroup determination.
        """
        self.wyckoffs = wyckoffs
        self.properties = properties
        self.symprec = symprec

    def describe(self, structure):
        a = SpacegroupAnalyzer(structure, self.symprec)
        symm = a.get_symmetrized_structure()
        data = []
        names = []
        for w in self.wyckoffs:
            site = symm.equivalent_sites[symm.wyckoff_symbols.index(w)][0]
            for p in self.properties:
                data.append(getattr(site.specie, p))
                names.append("%s-%s" % (w, p))
        return pd.Series(data, index=names)

