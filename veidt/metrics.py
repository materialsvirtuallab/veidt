import numpy as np
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
        config = {'class_name': identifier['class_name'], 'config': identifier['config']}
        return deserialize(config)
    elif isinstance(identifier, str):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
