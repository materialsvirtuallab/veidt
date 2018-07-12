import unittest
import numpy as np
from veidt.metrics import get
import os

file_path = os.path.dirname(__file__)


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

if __name__ == '__main__':
    unittest.main()

