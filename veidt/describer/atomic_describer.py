# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import itertools
import os
import io

import numpy as np
import pandas as pd
from pymatgen import Element
from pymatgen.io.lammps.data import structure_2_lmpdata
from monty.tempfile import ScratchDir

from veidt.abstract import Describer


class BispectrumCoefficients(Describer):
    """
    Bispectrum coefficients to describe the local environment of each
    atom in a quantitative way.

    """

    def __init__(self, rcutfac, twojmax, element_profile, rfac0=0.99363,
                 rmin0=0, diagonalstyle=3):
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
        
        """
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.element_profile = element_profile
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        self.diagonalstyle = diagonalstyle

    @property
    def subscripts(self):
        """
        The subscripts (2j1, 2j2, 2j) of all bispectrum components
        involved.

        """
        subs = itertools.product(range(self.twojmax + 1), repeat=3)
        filters = [lambda x: True if x[0] >= x[1] else False]
        if self.diagonalstyle == 2:
            filters.append(lambda x: True if x[0] == x[1] == x[2] else False)
        else:
            if self.diagonalstyle == 1:
                filters.append(lambda x: True if x[0] == x[1] else False)
            elif self.diagonalstyle == 3:
                filters.append(lambda x: True if x[2] >= x[0] else False)
            elif self.diagonalstyle == 0:
                pass
            j_filter = lambda x: True if \
                x[2] in range(x[0] - x[1], min(self.twojmax, x[0] + x[1]) + 1,
                              2) else False
            filters.append(j_filter)
        for f in filters:
            subs = filter(f, subs)
        return list(subs)

    def _run_lammps(self, structures):
        script = ['units metal',
                  'boundary p p p',
                  'atom_style charge',
                  'box tilt large',
                  'read_data data.sna',
                  'pair_style lj/cut 10',
                  'pair_coeff * * 1 1',
                  'compute sna all sna/atom 1 ',
                  'dump 1 all custom 1 dump.sna c_sna[*]',
                  'run 0']
        args = '{} {} '.format(self.rfac0, self.twojmax)
        elements = [el.symbol for el in sorted(Element(e) for e in
                                               self.element_profile.keys())]
        cutoffs, weights = [], []
        for e in elements:
            cutoffs.append(self.element_profile[e]['r'] * self.rcutfac)
            weights.append(self.element_profile[e]['w'])
        args += ' '.join([str(p) for p in cutoffs + weights])
        args += ' diagonal {} rmin0 {} bzeroflag 0'.format(self.diagonalstyle,
                                                           self.rmin0)
        script[-3] += args

        columns = list(map(lambda s: '-'.join(['%d' % i for i in s]),
                           self.subscripts))
        dfs = []
        with ScratchDir('.'):
            with open('in.sna', 'w') as f:
                f.write('\n'.join(script))
            for s in structures:
                ld = structure_2_lmpdata(s, elements)
                ld.write_file('data.sna')
                os.system('lmp_serial -in in.sna')
                with open('dump.sna') as f:
                    sna_lines = f.readlines()[9:]
                sna = np.loadtxt(io.StringIO(''.join(sna_lines)))
                dfs.append(pd.DataFrame(np.atleast_2d(sna), columns=columns))
        return dfs

    def describe(self, structure):
        """
        Returns data for one input structure.

        Args:
            structure (Structure): Input structure.

        Returns:
            DataFrame. The columns are the subscripts of bispectrum
            components, while indices are the site indices in
            input structure.

        """
        return self._run_lammps([structure])[0]

    def describe_all(self, structures):
        """
        Returns data for all input structures in a single DataFrame.

        Args:
            structures [Structure]: Input structures as a list.

        Returns:
            DataFrame with indices of input list preserved. To retrieve
            the data for structures[i], use
            df.xs(i, level='input_index').

        """
        dfs = self._run_lammps(structures)
        df = pd.concat(dfs, keys=range(len(structures)),
                       names=['input_index', None])
        return df
