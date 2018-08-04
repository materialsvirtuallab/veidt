import unittest
import numpy as np
from veidt.metrics import get, serialize, deserialize
import os

file_path = os.path.dirname(__file__)


def test_func():
    return 1


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x1 = np.array([1, 2, 3])
        cls.x2 = np.array([4, 5, 6])
        cls.x3 = np.array([1, 1, 1])
        cls.x4 = np.array([1, 2, 3])

    def test_mae(self):
        mae = get('mae')
        self.assertEqual(mae(self.x1, self.x2), 3)
        self.assertAlmostEqual(mae(self.x3, self.x4), 1)

    def test_mse(self):
        mse = get('mse')
        self.assertEqual(mse(self.x1, self.x2), 9)
        self.assertAlmostEqual(mse(self.x3, self.x4), 5. / 3)

    def test_binary_accuracy(self):
        binary_accuracy = get('binary_accuracy')
        self.assertAlmostEqual(binary_accuracy([0, 1, 0], [1, 1, 1]), 1./3)

        self.assertAlmostEqual(get({"class_name": "binary_accuracy",
                                    "config": {"y_true": [0, 1, 0],
                                               "y_pred": [1, 1, 1]}}), 1/3)
        self.assertEqual(get(test_func)(), 1)
        with self.assertRaises(ValueError) as context:
            get({"class_name":'not existing', "config": "not existing"})

    def test_deserialization(self):
        mae = deserialize("mae")
        self.assertEqual(mae(self.x1, self.x2), 3)

    def test_serialization(self):
        test_func_string = serialize(test_func)
        self.assertEqual(test_func_string, "test_func")

    def test_get(self):
        with self.assertRaises(ValueError):
            dummy_kernel = get([1, 2, 3])

if __name__ == '__main__':
    unittest.main()

