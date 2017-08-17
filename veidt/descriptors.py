# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import importlib

import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from veidt.abstract import Describer


class Generator(Describer):

    def __init__(self, func_dict):
        """
        :param funcs_dict (Dict): Dict with labels as keys and
            stringified function as values. The functions are
            recovered using eval() method.
        """
        self.func_dict = func_dict

    def describe(self, obj):
        """
        Returns description of an object based on all functions.

        :param obj: Object to be described.
        :return: {label: value} dict.
        """
        def get_func(name):
            try:
                breakdown = name.split(".")
                f_name = breakdown[-1]
                mod_name = ".".join(name.split(".")[:-1])
                mod = importlib.import_module(mod_name)
                func = getattr(mod, f_name)
            except:
                func = eval(name)
            return func

        output = {}
        for k, v in self.func_dict.items():
            func = get_func(v)
            output[k] = func(obj)
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

