# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import get_el_sp

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy

from veidt.abstract import Describer


class MultiDescriber(Describer):
    """
    This is a generic multiple describer that allows one to combine multiple
    describers.
    """

    def __init__(self, describers):
        """
        :param describers: List of describers. Note that the application of the
            Describers is from left to right. E.g., [Describer1(), Describer2()]
            will run Describer1.describe on the input object, and then run
            Describer2 on the output from Describer1.describe. This provides
            a powerful way to combine multiple describers to generate generic
            descriptors and basis functions.
        """
        self.describers = describers

    def describe(self, obj):
        desc = obj
        for d in self.describers:
            desc = d.describe(desc)
        return desc

    def describe_all(self, objs):
        descs = objs
        for d in self.describers:
            descs = d.describe_all(descs)
        return descs


class FuncGenerator(Describer):
    """
    General transformer for arrays. In principle, any numerical
    operations can be done as long as each involved function has a
    NumPy.ufunc implementation, e.g., np.sin, np.exp...
    """

    def __init__(self, func_dict, append=True):
        """
        :param func_dict: Dict with labels as keys and stringified
            function as values. The functions arerecovered from strings
            using eval() built-in function. All functions should be
            pointing to a NumPy.ufunc since the calculations will be
            performed on array-like objects. For functions implemented
            elsewhere other than in NumPy, e.g., functions in
            scipy.special, please make sure the module is imported.
        :param append: Whether return the full DataFrame with inputs.
            Default to True.
        """
        self.func_dict = func_dict
        self.append = append

    def describe(self, df):
        """
        Returns description of an object based on all functions.

        :param df: DataFrame with input data.
        :return: DataFrame with transformed data.
        """
        collector = []
        for k, v in self.func_dict.items():
            data = eval(v)(df)
            if isinstance(data, pd.Series):
                data.name = k
            elif isinstance(data, pd.DataFrame):
                columns = [k + " " + c for c in data.columns]
                data.columns = columns
            collector.append(data)
        new_df = pd.concat(collector, axis=1)
        if self.append:
            new_df = df.join(new_df)
        return new_df


class DistinctSiteProperty(Describer):
    """
    Constructs a descriptor based on properties of distinct sites with different coordination number in a
    structure.
    """
    #todo: generalize to multiple sites with the same Wyckoff.

    def __init__(self,properties,CNs=None, symprec=0.1):
        """
        :param CNs: List of coordination numbers of distintic sites. E.g., [8, 6, 4], return results for full
        possible CNs in the structure if CNs == None :param properties: Sequence of specie properties. E.g.,
        ["atomic_radius"], if "ionic_radius" is in the list, the structure should  be oxideation_state decorated
        otherwise attribute error raised. :param symprec: Symmetry precision for spacegroup determination.
        """
        self.CNs = CNs
        self.properties = properties
        self.symprec = symprec

    def describe(self, structure,exclude_ele=['O']):
        """

        :param structure: Pymatgen Structure Object, if ionic radius is in the property list, structure should be os
                            decorated
        :param exclude_ele: list of elements not to be considered, default ['O']
        :return: DataFrame with properties averaged for each cn.
        """
        a = SpacegroupAnalyzer(structure, self.symprec)
        #symm = a.get_symmetrized_structure()
        data = []
        names = []
        cn_sites = self.get_cn_sites(structure=structure,
                                     exclude_ele=exclude_ele,
                                     maximum_distance_factor=1.5)
        for cn in self.CNs if self.CNs else cn_sites:
            species= [i.specie for i in cn_sites[cn]]
            spe_occu = {spe:species.count(spe) for spe in set(species)}
            for p in self.properties:
                if p == 'X':
                    avg_p = self.get_averaged_X(spe_occu)
                else:
                    avg_p = np.average([getattr(spe,p) for spe,occ in spe_occu.items()],
                                       weights=[occ for spe,occ in spe_occu.items()])
                data.append(avg_p)
                names.append("%s-%s" % (cn, p))
        return pd.Series(data, index=names)


    def get_cn_sites(self,structure,exclude_ele=['O'],maximum_distance_factor=1.5):
        """

        :param structure: Pymatgen structure Object
        :param exclude_ele: list of elements not to be considered, eg ['O']
        :param maximum_distance_factor:
        :return: a dictionary in the format {cn_1:[sites with coordination number of cn_1]}
        """
        lgf = LocalGeometryFinder()
        lgf.setup_parameters(structure_refinement='none')
        lgf.setup_structure(structure)
        se = lgf.compute_structure_environments(maximum_distance_factor=maximum_distance_factor)
        default_strategy = SimplestChemenvStrategy(se)
        cn_sites = {}
        for eqslist in se.equivalent_sites:
            eqslist = [i for i in eqslist if i.specie.symbol not in exclude_ele]
            if not eqslist:
                continue
            site = eqslist[0]
            ces = default_strategy.get_site_coordination_environments(site)
            ce = ces[0]
            cn = int(ce[0].split(':')[1])
            if cn in cn_sites:
                cn_sites[cn].extend(eqslist)

            else:
                cn_sites.update({cn:[site for site in eqslist]})

        return cn_sites

    def get_averaged_X(self,spe_occu):
        """
        Calcualte averaged electronnegtivity of two mixed species
        :param spe_occu: specie in string or dict in the format {el1:amt1,el2:amt2}
        :return:return the mean of electronegtivity from definition (Binding energy)
                cf https://www.wikiwand.com/en/Electronegativity
        """
        # make sure spe does not contain charge
        o = get_el_sp('O2-')
        if len(spe_occu) < 2:
            el = get_el_sp(list(spe_occu.keys())[0])
            return el.X
        else:
            avg_eneg = 0
            factor = sum([v for k, v in spe_occu.items()])
            for s, amt in spe_occu.items():
                el = get_el_sp(s)
                avg_eneg += (amt/factor) * (el.X - o.X) ** 2

            return np.abs(o.X - (np.sqrt(avg_eneg)))



