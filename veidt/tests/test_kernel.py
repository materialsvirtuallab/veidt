import unittest
import numpy as np
from veidt.kernel import rbf, get_kernel
import os

file_path = os.path.dirname(__file__)


def test_func():
    pass


class TestKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x1 = np.array([[1, 2], [1, 2]])
        cls.x2 = np.array([[2, 3], [2, 3]])
        cls.sigma = 1

    def test_rbf(self):
        self.assertAlmostEqual(np.asscalar(np.sum(rbf(self.x1, self.x2, self.sigma))),
                               4.*np.exp(-2/2))
    def test_get_kernel(self):
        rbf2 = get_kernel('rbf')
        self.assertAlmostEqual(np.asscalar(np.sum(rbf2(self.x1, self.x2, self.sigma))),
                               4. * np.exp(-2 / 2))


if __name__ == '__main__':
    unittest.main()

