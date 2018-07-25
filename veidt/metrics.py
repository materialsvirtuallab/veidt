from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
from sklearn.metrics import mean_squared_error, mean_absolute_error

from veidt.utils.general_utils import deserialize_veidt_object
from veidt.utils.general_utils import serialize_veidt_object


def binary_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true).ravel() == np.array(y_pred).ravel())

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error

def serialize(metric):
    return serialize_veidt_object(metric)

def deserialize(config):
    return deserialize_veidt_object(config,
                                    module_objects=globals(),
                                    printable_module_name='metric function')


def get(identifier):
    if isinstance(identifier, dict):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif isinstance(identifier, six.string_types):
        return deserialize(str(identifier))
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)

