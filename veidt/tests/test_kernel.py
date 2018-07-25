import unittest
import numpy as np
from veidt.kernel import rbf, get_kernel
import os

file_path = os.path.dirname(__file__)


def test_func():
    return 1


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
        test_callable = get_kernel(test_func)
        self.assertEqual(1, test_callable())

        test_dict = get_kernel({"class_name": "rbf",
                                 "config": {"x1": np.array([[1, 2], [1, 2]]),
                                            "x2": np.array([[2, 3], [2, 3]]),
                                            "sigma": 1}})
        self.assertAlmostEqual(np.asscalar(np.sum(test_dict)),
                               4. * np.exp(-2 / 2))
        with self.assertRaises(ValueError) as context:
            get_kernel({"class_name": 'none existing',
                        "config": "none existing"})






if __name__ == '__main__':
    unittest.main()

