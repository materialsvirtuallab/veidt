import unittest
import numpy as np
from veidt.utils.general_utils import serialize_veidt_object, deserialize_veidt_object
import os

file_path = os.path.dirname(__file__)


def test_func():
    return 1


class TestGeneralUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x1 = np.array([1, 2, 3])
        cls.x2 = np.array([4, 5, 6])
        cls.x3 = np.array([1, 1, 1])
        cls.x4 = np.array([1, 2, 3])

    def test_serialization(self):
        self.assertEqual(serialize_veidt_object(test_func), "test_func")

    def test_deserialization(self):
        self.assertEqual(1, deserialize_veidt_object('test_func', module_objects=globals())())

if __name__ == '__main__':
    unittest.main()

