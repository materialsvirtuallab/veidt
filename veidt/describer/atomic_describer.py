# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import itertools
import subprocess
import io

import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import get_el_sp

from veidt.abstract import Describer

class BispectrumCoefficients(Describer):
    """
    Bispectrum coefficients to describe the local environment of each
    atom in a quantitative way.

    """

    def __init__(self, rcutfac, twojmax, element_profile, rfac0=0.99363,
                 rmin0=0, diagonalstyle=3, pot_fit=False):
        """

        Args:
            rcutfac (float): Global cutoff distance.
            twojmax (int): Band limit for bispectrum components.
            element_profile (dict): Parameters (cutoff factor 'r' and
                weight 'w') related to each element, e.g.,
                {'Na': {'r': 0.3, 'w': 0.9},
                 'Cl': {'r': 0.7, 'w': 3.0}}
            rfac0 (float): Parameter in distance to angle conversion.
                Set between (0, 1), default to 0.99363.
            rmin0 (float): Parameter in distance to angle conversion.
                Default to 0.
            diagonalstyle (int): Parameter defining which bispectrum
                components are generated. Choose among 0, 1, 2 and 3,
                default to 3.
            pot_fit (bool): Whether to output in potential fitting
                format. Default to False, i.e., returning the bispectrum
                coefficients for each site.

        """
        from veidt.potential.lammps.calcs import SpectralNeighborAnalysis
        self.calculator = SpectralNeighborAnalysis(rcutfac, twojmax,
                                                   element_profile,
                                                   rfac0, rmin0,
                                                   diagonalstyle)
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.element_profile = element_profile
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        self.diagonalstyle = diagonalstyle
        self.elements = sorted(element_profile.keys(),
                               key=lambda sym: get_el_sp(sym).X)
        self.pot_fit = pot_fit

    @property
    def subscripts(self):
        """
        The subscripts (2j1, 2j2, 2j) of all bispectrum components
        involved.

        """
        return self.calculator.get_bs_subscripts(self.twojmax,
                                                 self.diagonalstyle)

    def describe(self, structure, include_stress=False):
        """
        Returns data for one input structure.

        Args:
            structure (Structure): Input structure.
            include_stress (bool): Whether to include stress descriptors.

        Returns:
            DataFrame.

            In regular format, the columns are the subscripts of
            bispectrum components, while indices are the site indices
            in input structure.

            In potential fitting format, to match the sequence of
            [energy, f_x[0], f_y[0], ..., f_z[N], v_xx, ..., v_xy], the
            bispectrum coefficients are summed up by each specie and
            normalized by a factor of No. of atoms (in the 1st row),
            while the derivatives in each direction are preserved, with
            the columns being the subscripts of bispectrum components
            with each specie and the indices being
            [0, '0_x', '0_y', ..., 'N_z'], and the virial contributions
            (in GPa) are summed up for all atoms for each component in
            the sequence of ['xx', 'yy', 'zz', 'yz', 'xz', 'xy'].

        """
        return self.describe_all([structure], include_stress).xs(0, level='input_index')

    def describe_all(self, structures, include_stress=False):
        """
        Returns data for all input structures in a single DataFrame.

        Args:
            structures (Structure): Input structures as a list.
            include_stress (bool): Whether to include stress descriptors.

        Returns:
            DataFrame with indices of input list preserved. To retrieve
            the data for structures[i], use
            df.xs(i, level='input_index').

        """
        columns = list(map(lambda s: '-'.join(['%d' % i for i in s]),
                           self.subscripts))
        raw_data = self.calculator.calculate(structures)

        def process(output, combine, idx, include_stress):
            b, db, vb, e = output
            df = pd.DataFrame(b, columns=columns)
            if combine:
                df_add = pd.DataFrame({'element': e, 'n': np.ones(len(e))})
                df_b = df_add.join(df)
                n_atoms = df_b.shape[0]
                b_by_el = [df_b[df_b['element'] == e] for e in self.elements]
                sum_b = [df[df.columns[1:]].sum(axis=0) for df in b_by_el]
                hstack_b = pd.concat(sum_b, keys=self.elements)
                hstack_b = hstack_b.to_frame().T / n_atoms
                hstack_b.fillna(0, inplace=True)
                dbs = np.split(db, len(self.elements), axis=1)
                dbs = np.hstack([np.insert(d.reshape(-1, len(columns)),
                                           0, 0, axis=1) for d in dbs])
                db_index = ['%d_%s' % (i, d)
                            for i in df_b.index for d in 'xyz']
                df_db = pd.DataFrame(dbs, index=db_index,
                                     columns=hstack_b.columns)
                if include_stress:
                    vbs = np.split(vb.sum(axis=0), len(self.elements))
                    vbs = np.hstack([np.insert(v.reshape(-1, len(columns)),
                                               0, 0, axis=1) for v in vbs])
                    volume = structures[idx].volume
                    vbs = vbs / volume * 160.21766208  # from eV to GPa
                    vb_index = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
                    df_vb = pd.DataFrame(vbs, index=vb_index,
                                         columns=hstack_b.columns)
                    df = pd.concat([hstack_b, df_db, df_vb])
                else:
                    df = pd.concat([hstack_b, df_db])
            return df

        df = pd.concat([process(d, self.pot_fit, i, include_stress)
                        for i, d in enumerate(raw_data)],
                       keys=range(len(raw_data)), names=["input_index", None])
        return df


class AGNIFingerprints(Describer):
    """
    Fingerprints for AGNI (Adaptive, Generalizable and Neighborhood
    Informed) force field. Elemental systems only.

    """

    def __init__(self, r_cut, etas):
        """

        Args:
            r_cut (float): Cutoff distance.
            etas (numpy.array): All eta parameters in 1D array.
        """
        self.r_cut = r_cut
        self.etas = etas

    def describe(self, structure):
        """
        Calculate fingerprints for all sites in a structure.

        Args:
            structure (Structure): Input structure.

        Returns:
            DataFrame.

        """
        all_neighbors = structure.get_all_neighbors(self.r_cut)
        fingerprints = []
        for i, an in enumerate(all_neighbors):
            center = structure[i].coords
            coords, distances = zip(*[(site.coords, d) for (site, d) in an])
            v = (np.array(coords) - center)[:, :, None]
            d = np.array(distances)[:, None, None]
            e = np.array(self.etas)[None, None, :]
            cf = 0.5 * (np.cos(np.pi * d / self.r_cut) + 1)
            fpi = np.sum(v / d * np.exp(-(d / e) ** 2) * cf, axis=0)
            fingerprints.append(fpi)
        index = ["%d_%s" % (i, d) for i in range(len(structure))
                 for d in "xyz"]
        df = pd.DataFrame(np.vstack(fingerprints), index=index,
                          columns=self.etas)
        return df

    def describe_all(self, structures):
        return pd.concat([self.describe(s) for s in structures],
                         keys=range(len(structures)),
                         names=['input_index', None])