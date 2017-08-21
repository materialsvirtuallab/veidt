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

    def describe_all(self, objs):
        """
        Convenience method to convert a list of objects to a list of
        descriptors. Default implementation simply loops a call to describe, but
        in some instances, a batch implementation may be more efficient.

        :param objs: List of objects

        :return: Concatenated descriptions for all objects. Recommended format
            is a pandas DataFrame.
        """
        return [self.describe(o) for o in objs]

    def __repr__(self):
        return self.__name__


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

    def __repr__(self):
        return self.__name__
