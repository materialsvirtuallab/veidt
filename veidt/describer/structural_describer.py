from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ..abstract import Describer
import pandas as pd


class DistinctSiteProperty(Describer):
    """
    Constructs a descriptor based on properties of distinct sites in a
    structure. For now, this assumes that there is only one type of species in
    a particular Wyckoff site.
    """
    #todo: generalize to multiple sites with the same Wyckoff.

    def fit(self, structures, target=None):
        return self

    def __init__(self, wyckoffs, properties, symprec=0.1):
        """
        :param wyckoffs: List of wyckoff symbols. E.g., ["48a", "24c"]
        :param properties: Sequence of specie properties. E.g.,
            ["atomic_radius"]. Look at pymatgen.core.periodic_table.Element and
            pymatgen.core.periodic_table.Specie for support properties (there
            are a lot!)
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
        return pd.DataFrame([data], columns=names)
