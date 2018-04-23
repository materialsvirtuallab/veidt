from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ..abstract import Describer
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class DistinctSiteProperty(Describer):
    """
    Constructs a describer based on properties of distinct sites in a
    structure. For now, this assumes that there is only one type of species in
    a particular Wyckoff site.
    """
    # todo: generalize to multiple sites with the same Wyckoff.

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


class GraphicRep(Describer):
    """
    Graphic representation of crystal

    """

    def __init__(self, structures, structure_ids, atomic_describer,
                 max_num_nbr=12, radius=8, dmin=0,
                 step=0.2, var=None, random_sed=9):
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.dmin, self.var, self.step = dmin, var, step
        self.ad = atomic_describer  # Todo: implement the default atomic from the paper and test others
        self.structures = structures

    def gaussian_dis_expand(self, distances, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        filter = np.arange(dmin, dmax + step, step)
        var = var if var else step

        return np.exp(-(distances[..., np.newaxis] - filter) ** 2 / var ** 2)

    def describe(self, structure):
        atom_fea = np.vstack([self.ad.describe(site.specie.element) for site in structure])
        atom_fea = torch.Tensor(atom_fea)

        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        # Sort nbrs for each site in distance
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        # Get two matrix: nbr idx and nbr dis
        nbr_fea_idx = np.array([list(map(lambda x: x[2],
                                         nbr[:self.max_num_nbr]) for nbr in all_nbrs)])
        nbr_fea = np.array([list(map(lambda x: x[1],
                                     nbr[: self.max_num_nbr])) for nbr in all_nbrs])
        nbr_fea = self.gaussian_dis_expand(nbr_fea,
                                           self.dmin, self.radius, self.step, self.var)

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return (atom_fea, nbr_fea, nbr_fea_idx)
