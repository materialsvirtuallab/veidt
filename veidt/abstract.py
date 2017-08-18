# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""
Define abstract base classes.
"""

import abc
import warnings

import six
from monty.json import MSONable
import numpy as np
import pandas as pd


class Describer(six.with_metaclass(abc.ABCMeta, MSONable)):
    """
    Base class for a Describer, i.e., something that converts an object to a
    descriptor, typically a numerical representation useful for machine
    learning.
    """

    @abc.abstractmethod
    def describe(self, obj):
        """
        Converts an obj to a descriptor.

        :param obj: Object
        :return: Descriptor for a structure. Recommended format is a pandas
            Series or Dataframe object for easy manipulation. For example, a
            simple site descriptor of the fractional coordinates (this is
            usually a bad descriptor, so it is just for illustration purposes)
            can be generated as::

                print(pd.DataFrame(s.frac_coords, columns=["a", "b", "c"]))
                          a         b         c
                0  0.000000  0.000000  0.000000
                1  0.750178  0.750178  0.750178
                2  0.249822  0.249822  0.249822

            Pandas dataframes can be dumped to a variety of formats (json, csv,
            etc.) easily.
        """
        pass

    def describe_all(self, objs, fmt='list'):
        """
        Convenience method to convert a list of objects to a list of
        descriptors. Default implementation simply loops a call to describe, but
        in some instances, a batch implementation may be more efficient.

        :param objs: List of objects
        :param fmt: (Str) output format, Choose among "list" as regular
            list, "arr" as NumPy array or "df" as pandas DataFrame.
        :return: Concatenated descriptions for all objects.
        """
        descriptions = [self.describe(o) for o in objs]
        concat = None
        if fmt == 'list':
            concat = descriptions
        elif fmt == 'arr':
            dim = len(np.array(descriptions[0]).shape)
            if dim > 1:
                warnings.warn("High dimensional (%d)" % dim +
                              " data to be concatenated, original index of"
                              " input list NOT preserved.")
            concat = np.vstack(descriptions)
        elif fmt == 'df':
            if isinstance(descriptions[0], pd.Series):
                concat = pd.concat(descriptions, axis=1,
                                   keys=range(len(objs))).T
            elif isinstance(descriptions[0], pd.DataFrame):
                concat = pd.concat(descriptions, axis=0,
                                   keys=range(len(objs)),
                                   names=["input_index", None])
        else:
            raise RuntimeError('Unsupported output format. '
                               'Choose among "list" as regular list, '
                               '"arr" as NumPy array or '
                               '"df" as pandas DataFrame.')
        return concat


class Model(six.with_metaclass(abc.ABCMeta, MSONable)):
    """
    Abstract Base class for a Model. Basically, it usually wraps around a deep
    learning package, e.g., the Sequential Model in Keras, but provides for
    transparent conversion of arbitrary input and outputs.
    """

    @abc.abstractmethod
    def fit(self, inputs, outputs):
        """
        Fit the model.

        :param inputs: List of input objects
        :param outputs: List of output objects
        """
        pass

    @abc.abstractmethod
    def predict(self, inputs):
        """
        Predict the value given a set of inputs based on fitted model.

        :param inputs: List of input objects

        :return: List of output objects
        """
        pass
